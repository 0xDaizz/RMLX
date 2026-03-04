use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::mpsc;

use clap::Args;

use crate::hostfile;
use crate::ssh;

#[derive(Args)]
pub struct LaunchArgs {
    /// Comma-separated hostnames/IPs
    #[arg(long)]
    hosts: Option<String>,

    /// JSON hostfile (entries need 'ssh')
    #[arg(long)]
    hostfile: Option<String>,

    /// Processes per host (default: 1)
    #[arg(short = 'n', long = "repeat-hosts", default_value_t = 1)]
    repeat_hosts: usize,

    /// Backend hint exported as RMLX_BACKEND
    #[arg(long, default_value = "rdma")]
    backend: String,

    /// SSH user override
    #[arg(long)]
    ssh_user: Option<String>,

    /// Extra env KEY=VALUE (repeatable)
    #[arg(long = "env")]
    extra_env: Vec<String>,

    /// Command to run (prefix with --)
    #[arg(trailing_var_arg = true, required = true)]
    command: Vec<String>,
}

struct Slot {
    rank: usize,
    world_size: usize,
    host: String,
    local_slot: usize,
}

/// Validate that a string is a safe shell variable name: `[A-Za-z_][A-Za-z0-9_]*`.
fn is_valid_env_key(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_env_pairs(values: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut env = BTreeMap::new();
    for item in values {
        let eq_pos = item
            .find('=')
            .ok_or_else(|| format!("invalid --env entry: {item:?} (expected KEY=VALUE)"))?;
        let key = item[..eq_pos].trim().to_string();
        if !is_valid_env_key(&key) {
            return Err(format!(
                "invalid --env key {key:?}: must match [A-Za-z_][A-Za-z0-9_]*"
            ));
        }
        let val = item[eq_pos + 1..].to_string();
        env.insert(key, val);
    }
    Ok(env)
}

fn build_slots(hosts: &[String], repeat: usize) -> Vec<Slot> {
    let world_size = hosts.len() * repeat;
    let mut slots = Vec::with_capacity(world_size);
    let mut rank = 0;
    for host in hosts {
        for local_slot in 0..repeat {
            slots.push(Slot {
                rank,
                world_size,
                host: host.clone(),
                local_slot,
            });
            rank += 1;
        }
    }
    slots
}

fn build_remote_command(
    base_cmd: &str,
    slot: &Slot,
    backend: &str,
    coordinator: &str,
    extra_env: &BTreeMap<String, String>,
) -> String {
    let mut env: BTreeMap<&str, String> = BTreeMap::new();
    env.insert("RMLX_RANK", slot.rank.to_string());
    env.insert("RMLX_WORLD_SIZE", slot.world_size.to_string());
    env.insert("RMLX_LOCAL_SLOT", slot.local_slot.to_string());
    env.insert("RMLX_BACKEND", backend.to_string());
    env.insert("RMLX_COORDINATOR", coordinator.to_string());
    for (k, v) in extra_env {
        env.insert(k.as_str(), v.clone());
    }
    let exports: Vec<String> = env
        .iter()
        .map(|(k, v)| format!("{k}={}", ssh::shell_quote(v)))
        .collect();
    format!("{} {base_cmd}", exports.join(" "))
}

enum OutputMsg {
    Line {
        rank: usize,
        host: String,
        text: String,
    },
}

pub fn run(args: LaunchArgs) -> i32 {
    // Parse command (strip leading --)
    let mut command = args.command;
    if command.first().map(|s| s.as_str()) == Some("--") {
        command.remove(0);
    }
    if command.is_empty() {
        eprintln!("error: empty command after --");
        return 2;
    }
    let base_cmd: String = command
        .iter()
        .map(|c| ssh::shell_quote(c))
        .collect::<Vec<_>>()
        .join(" ");

    // Resolve hosts
    let hosts = match hostfile::resolve_hosts(args.hosts.as_deref(), args.hostfile.as_deref()) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };

    if args.repeat_hosts < 1 {
        eprintln!("error: --repeat-hosts must be >= 1");
        return 2;
    }

    let extra_env = match parse_env_pairs(&args.extra_env) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };

    let slots = build_slots(&hosts, args.repeat_hosts);
    let coordinator = hosts[0].clone();

    let (tx, rx) = mpsc::channel::<OutputMsg>();
    let mut procs: Vec<(usize, String, std::process::Child)> = Vec::new();

    // Spawn all processes
    for slot in &slots {
        let slot_cmd =
            build_remote_command(&base_cmd, slot, &args.backend, &coordinator, &extra_env);
        match ssh::spawn_remote(&slot.host, &slot_cmd, args.ssh_user.as_deref()) {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let rank = slot.rank;
                let host = slot.host.clone();

                // Reader thread for stdout
                if let Some(pipe) = stdout {
                    let tx_clone = tx.clone();
                    let host_clone = host.clone();
                    std::thread::spawn(move || {
                        let reader = std::io::BufReader::new(pipe);
                        for text in reader.lines().map_while(Result::ok) {
                            let _ = tx_clone.send(OutputMsg::Line {
                                rank,
                                host: host_clone.clone(),
                                text,
                            });
                        }
                    });
                }

                // Reader thread for stderr
                if let Some(pipe) = stderr {
                    let tx_clone = tx.clone();
                    let host_clone = host.clone();
                    std::thread::spawn(move || {
                        let reader = std::io::BufReader::new(pipe);
                        for text in reader.lines().map_while(Result::ok) {
                            let _ = tx_clone.send(OutputMsg::Line {
                                rank,
                                host: host_clone.clone(),
                                text,
                            });
                        }
                    });
                }

                procs.push((slot.rank, slot.host.clone(), child));
            }
            Err(e) => {
                eprintln!(
                    "error: failed to spawn rank {} on {}: {e}",
                    slot.rank, slot.host
                );
                // Kill already-spawned processes
                for (_, _, ref mut p) in &mut procs {
                    let _ = p.kill();
                }
                return 1;
            }
        }
    }

    // Drop the original sender so rx will close when all threads finish
    drop(tx);

    let mut failures: BTreeMap<usize, i32> = BTreeMap::new();

    // Multiplex output and monitor processes
    loop {
        // Drain available messages
        loop {
            match rx.try_recv() {
                Ok(OutputMsg::Line { rank, host, text }) => {
                    println!("[rank={rank} host={host}] {text}");
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        // Check process statuses
        let mut alive = 0;
        for (rank, _, ref mut child) in &mut procs {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code().unwrap_or(1);
                    if code != 0 && !failures.contains_key(rank) {
                        failures.insert(*rank, code);
                    }
                }
                Ok(None) => alive += 1,
                Err(_) => {}
            }
        }

        // Fail-fast: terminate remaining on first failure
        if !failures.is_empty() {
            for (_, _, ref mut child) in &mut procs {
                if child.try_wait().ok().flatten().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }

        if alive == 0 {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Drain remaining output
    for msg in rx.try_iter() {
        match msg {
            OutputMsg::Line { rank, host, text } => {
                println!("[rank={rank} host={host}] {text}");
            }
        }
    }

    // Collect final exit codes
    for (rank, _, ref mut child) in &mut procs {
        if let Ok(status) = child.wait() {
            let code = status.code().unwrap_or(1);
            if code != 0 {
                failures.entry(*rank).or_insert(code);
            }
        }
    }

    if !failures.is_empty() {
        let ordered: Vec<String> = failures
            .iter()
            .map(|(r, c)| format!("rank {r}: {c}"))
            .collect();
        eprintln!("launch failed ({})", ordered.join(", "));
        return 1;
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_pairs() {
        let pairs = vec!["FOO=bar".into(), "BAZ=1".into()];
        let env = parse_env_pairs(&pairs).unwrap();
        assert_eq!(env["FOO"], "bar");
        assert_eq!(env["BAZ"], "1");
    }

    #[test]
    fn test_parse_env_pairs_no_eq() {
        let pairs = vec!["NOEQ".into()];
        assert!(parse_env_pairs(&pairs).is_err());
    }

    #[test]
    fn test_parse_env_pairs_empty_key() {
        let pairs = vec!["=val".into()];
        assert!(parse_env_pairs(&pairs).is_err());
    }

    #[test]
    fn test_parse_env_pairs_injection_key() {
        let pairs = vec!["BAD;echo INJECTED=1".into()];
        assert!(parse_env_pairs(&pairs).is_err());
    }

    #[test]
    fn test_is_valid_env_key() {
        assert!(is_valid_env_key("FOO"));
        assert!(is_valid_env_key("_BAR"));
        assert!(is_valid_env_key("A_B_3"));
        assert!(!is_valid_env_key(""));
        assert!(!is_valid_env_key("1ABC"));
        assert!(!is_valid_env_key("A;B"));
        assert!(!is_valid_env_key("A B"));
    }

    #[test]
    fn test_build_slots() {
        let hosts = vec!["a".into(), "b".into()];
        let slots = build_slots(&hosts, 2);
        assert_eq!(slots.len(), 4);
        assert_eq!(slots[0].rank, 0);
        assert_eq!(slots[0].host, "a");
        assert_eq!(slots[0].local_slot, 0);
        assert_eq!(slots[1].rank, 1);
        assert_eq!(slots[1].host, "a");
        assert_eq!(slots[1].local_slot, 1);
        assert_eq!(slots[2].rank, 2);
        assert_eq!(slots[2].host, "b");
        assert_eq!(slots[3].rank, 3);
    }

    #[test]
    fn test_build_remote_command() {
        let slot = Slot {
            rank: 0,
            world_size: 2,
            host: "node1".into(),
            local_slot: 0,
        };
        let extra = BTreeMap::new();
        let cmd = build_remote_command("echo hello", &slot, "rdma", "node1", &extra);
        assert!(cmd.contains("RMLX_RANK=0"));
        assert!(cmd.contains("RMLX_WORLD_SIZE=2"));
        assert!(cmd.contains("RMLX_BACKEND=rdma"));
        assert!(cmd.contains("RMLX_COORDINATOR=node1"));
        assert!(cmd.contains("echo hello"));
    }
}
