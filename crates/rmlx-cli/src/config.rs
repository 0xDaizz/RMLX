use clap::Args;

use crate::hostfile::HostEntry;
use crate::ssh;

#[derive(Args)]
pub struct ConfigArgs {
    /// Comma-separated hostnames or IPs (e.g. node1,node2)
    #[arg(long)]
    hosts: String,

    /// Distributed backend profile for hostfile generation
    #[arg(long, default_value = "rdma", value_parser = ["rdma", "loopback"])]
    backend: String,

    /// Control-plane topology hint
    #[arg(long = "over", default_value = "thunderbolt", value_parser = ["thunderbolt", "ethernet"])]
    over: String,

    /// Interface used to resolve host control IP (default: en0)
    #[arg(long, default_value = "en0")]
    control_iface: String,

    /// Run baseline host setup commands remotely (requires passwordless sudo)
    #[arg(long)]
    auto_setup: bool,

    /// Skip RDMA device probe validation (hostfile still generated)
    #[arg(long)]
    no_verify_rdma: bool,

    /// Output hostfile path (default: rmlx-hosts.json)
    #[arg(long, default_value = "rmlx-hosts.json")]
    output: String,

    /// SSH user override (default: current user)
    #[arg(long)]
    ssh_user: Option<String>,

    /// Per-host command timeout seconds
    #[arg(long, default_value_t = 20)]
    timeout: u32,

    /// Verbose progress logs
    #[arg(long)]
    verbose: bool,
}

struct HostInfo {
    ssh: String,
    ip: String,
    rdma_devices: Vec<String>,
}

fn must_ok(output: std::io::Result<std::process::Output>, context: &str) -> Result<String, String> {
    let output = output.map_err(|e| format!("{context}: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("exit code {}", output.status.code().unwrap_or(-1))
        };
        return Err(format!("{context}: {detail}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn verify_ssh(host: &str, user: Option<&str>, timeout: u32) -> Result<(), String> {
    must_ok(
        ssh::run_remote(host, "echo rmlx-ssh-ok", user, timeout),
        &format!("ssh probe failed on {host}"),
    )?;
    Ok(())
}

fn probe_control_ip(
    host: &str,
    iface: &str,
    user: Option<&str>,
    timeout: u32,
) -> Result<String, String> {
    let cmd = format!("ipconfig getifaddr {}", ssh::shell_quote(iface));
    let out = must_ok(
        ssh::run_remote(host, &cmd, user, timeout),
        &format!("failed to query control IP on {host}"),
    )?;
    let ip = out.lines().next().unwrap_or("").trim().to_string();
    if ip.is_empty() {
        return Err(format!(
            "failed to query control IP on {host}: empty output"
        ));
    }
    Ok(ip)
}

fn probe_rdma_devices(host: &str, user: Option<&str>, timeout: u32) -> Result<Vec<String>, String> {
    let out = must_ok(
        ssh::run_remote(
            host,
            "ibv_devices | awk 'NR>2 && NF>0 {print $1}'",
            user,
            timeout,
        ),
        &format!("failed to probe RDMA devices on {host}"),
    )?;
    Ok(out
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

fn auto_setup_host(
    host: &str,
    user: Option<&str>,
    timeout: u32,
    verbose: bool,
) -> Result<(), String> {
    let cmd = concat!(
        "set -euo pipefail; ",
        "if networksetup -listallnetworkservices | grep -qx 'Thunderbolt Bridge'; then ",
        "sudo networksetup -setnetworkserviceenabled 'Thunderbolt Bridge' off || true; ",
        "fi; ",
        "sudo ifconfig bridge0 down 2>/dev/null || true",
    );
    if verbose {
        println!("[{host}] auto-setup: disable Thunderbolt Bridge");
    }
    must_ok(
        ssh::run_remote(host, cmd, user, timeout),
        &format!("auto-setup failed on {host}"),
    )?;
    Ok(())
}

fn build_rdma_map(devs: &[String], rank: usize, world_size: usize) -> Vec<Option<String>> {
    let mut row = Vec::with_capacity(world_size);
    let mut cursor = 0usize;
    for peer in 0..world_size {
        if peer == rank {
            row.push(None);
        } else {
            row.push(Some(devs[cursor % devs.len()].clone()));
            cursor += 1;
        }
    }
    row
}

pub fn run(args: ConfigArgs) -> i32 {
    let hosts: Vec<String> = args
        .hosts
        .split(',')
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .collect();

    if hosts.is_empty() {
        tracing::error!(target: "rmlx_cli", "at least one host is required");
        return 1;
    }

    let user = args.ssh_user.as_deref();
    let mut infos: Vec<HostInfo> = Vec::new();

    for host in &hosts {
        if args.verbose {
            println!("[{host}] probing ssh");
        }
        if let Err(e) = verify_ssh(host, user, args.timeout) {
            tracing::error!(target: "rmlx_cli", %e, "SSH verification failed");
            return 1;
        }

        if args.auto_setup && args.over == "thunderbolt" {
            if let Err(e) = auto_setup_host(host, user, args.timeout, args.verbose) {
                tracing::error!(target: "rmlx_cli", %e, "auto-setup failed");
                return 1;
            }
        }

        if args.verbose {
            println!("[{host}] probing control IP via {}", args.control_iface);
        }
        let ip = match probe_control_ip(host, &args.control_iface, user, args.timeout) {
            Ok(ip) => ip,
            Err(e) => {
                tracing::error!(target: "rmlx_cli", %e, "failed to probe control IP");
                return 1;
            }
        };

        let mut rdma_devs = Vec::new();
        if args.backend == "rdma" && !args.no_verify_rdma {
            if args.verbose {
                println!("[{host}] probing RDMA devices (ibv_devices)");
            }
            match probe_rdma_devices(host, user, args.timeout) {
                Ok(devs) => {
                    if devs.is_empty() {
                        tracing::error!(
                            target: "rmlx_cli",
                            %host,
                            "no RDMA devices found; use --no-verify-rdma to bypass",
                        );
                        return 1;
                    }
                    rdma_devs = devs;
                }
                Err(e) => {
                    tracing::error!(target: "rmlx_cli", %e, "RDMA probe failed");
                    return 1;
                }
            }
        }

        infos.push(HostInfo {
            ssh: host.clone(),
            ip,
            rdma_devices: rdma_devs,
        });
    }

    let world = infos.len();
    let mut entries: Vec<HostEntry> = Vec::new();

    for (rank, info) in infos.iter().enumerate() {
        let rdma = if args.backend == "rdma" {
            if !args.no_verify_rdma {
                let needed = world - 1;
                if info.rdma_devices.len() < needed {
                    tracing::error!(
                        target: "rmlx_cli",
                        host = %info.ssh,
                        needed,
                        found = info.rdma_devices.len(),
                        devices = ?info.rdma_devices,
                        "insufficient RDMA devices for full mesh",
                    );
                    return 1;
                }
            }
            let devs = if info.rdma_devices.is_empty() {
                vec!["rdma_device_todo".to_string()]
            } else {
                info.rdma_devices.clone()
            };
            Some(build_rdma_map(&devs, rank, world))
        } else {
            None
        };

        entries.push(HostEntry {
            ssh: info.ssh.clone(),
            ips: vec![info.ip.clone()],
            rdma,
        });
    }

    let json = match serde_json::to_string_pretty(&entries) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!(target: "rmlx_cli", %e, "failed to serialize hostfile");
            return 1;
        }
    };

    if let Err(e) = std::fs::write(&args.output, &json) {
        tracing::error!(target: "rmlx_cli", path = %args.output, %e, "failed to write hostfile");
        return 1;
    }

    println!("wrote hostfile: {}", args.output);
    println!("hosts: {}", hosts.join(", "));
    println!("backend: {}  over: {}", args.backend, args.over);
    println!(
        "next: rmlx launch --backend {} --hostfile {} -- <command>",
        args.backend,
        ssh::shell_quote(&args.output),
    );

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_rdma_map_2nodes() {
        let devs = vec!["mlx5_0".into(), "mlx5_1".into()];
        let map = build_rdma_map(&devs, 0, 2);
        assert_eq!(map, vec![None, Some("mlx5_0".into())]);
        let map = build_rdma_map(&devs, 1, 2);
        assert_eq!(map, vec![Some("mlx5_0".into()), None]);
    }

    #[test]
    fn test_build_rdma_map_3nodes() {
        let devs = vec!["d0".into(), "d1".into()];
        let map = build_rdma_map(&devs, 0, 3);
        assert_eq!(map, vec![None, Some("d0".into()), Some("d1".into())]);
        let map = build_rdma_map(&devs, 1, 3);
        assert_eq!(map, vec![Some("d0".into()), None, Some("d1".into())]);
    }

    #[test]
    fn test_build_rdma_map_single_device_wraps() {
        // When there is only one device, it wraps (cursor % 1 == 0 always).
        let devs = vec!["mlx5_0".into()];
        let map = build_rdma_map(&devs, 0, 3);
        assert_eq!(
            map,
            vec![None, Some("mlx5_0".into()), Some("mlx5_0".into())]
        );
    }

    #[test]
    fn test_build_rdma_map_self_is_none() {
        // For any rank configuration, self entry should be None.
        let devs = vec!["d0".into(), "d1".into(), "d2".into()];
        for rank in 0..4 {
            let map = build_rdma_map(&devs, rank, 4);
            assert_eq!(map[rank], None);
            assert_eq!(map.len(), 4);
        }
    }

    #[test]
    fn test_build_rdma_map_4nodes_last_rank() {
        let devs = vec!["d0".into(), "d1".into(), "d2".into()];
        let map = build_rdma_map(&devs, 3, 4);
        // rank=3: peers are 0,1,2. cursor cycles through d0,d1,d2.
        assert_eq!(
            map,
            vec![
                Some("d0".into()),
                Some("d1".into()),
                Some("d2".into()),
                None,
            ]
        );
    }

    #[test]
    fn test_must_ok_success() {
        let _output = std::process::Output {
            status: std::process::ExitStatus::default(),
            stdout: b"hello\n".to_vec(),
            stderr: Vec::new(),
        };
        // On macOS, ExitStatus::default() is success (0).
        // We test the function directly if we can construct a successful Output.
        // Unfortunately ExitStatus cannot be easily constructed, so we use Command.
        let result = must_ok(
            std::process::Command::new("echo").arg("test123").output(),
            "echo test",
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test123");
    }

    #[test]
    fn test_must_ok_failure() {
        let result = must_ok(
            std::process::Command::new("false").output(),
            "should fail",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("should fail"));
    }
}
