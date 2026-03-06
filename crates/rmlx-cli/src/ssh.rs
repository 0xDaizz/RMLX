use std::io::Read;
use std::process::{Child, Command, Output, Stdio};

/// Shell-quote a string (equivalent to Python's `shlex.quote`).
pub fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // If the string only contains safe characters, return as-is.
    if s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"@%+=:,./-_".contains(&b))
    {
        return s.to_string();
    }
    // Wrap in single quotes, escaping any embedded single quotes.
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\"'\"'");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn is_localhost(host: &str) -> bool {
    host == "localhost" || host == "127.0.0.1"
}

/// Validate that a host or user string is safe for use as an SSH argument.
/// Rejects strings starting with `-` (option injection) and strings containing
/// characters outside `[a-zA-Z0-9._@:-]`.
pub fn validate_ssh_target(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if value.starts_with('-') {
        return Err(format!(
            "{field} {value:?} starts with '-', which could inject SSH options"
        ));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"._@:-".contains(&b))
    {
        return Err(format!(
            "{field} {value:?} contains invalid characters (allowed: a-zA-Z0-9._@:-)"
        ));
    }
    Ok(())
}

/// Run a command on a remote host (or locally if localhost) and wait for output.
/// Enforces a per-command timeout matching Python's `subprocess.run(timeout=...)`.
pub fn run_remote(
    host: &str,
    cmd: &str,
    user: Option<&str>,
    timeout_secs: u32,
) -> std::io::Result<Output> {
    validate_ssh_target(host, "host").map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    if let Some(u) = user {
        validate_ssh_target(u, "user").map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    }

    let mut child = if is_localhost(host) {
        Command::new("bash")
            .args(["-lc", cmd])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    } else {
        let target = match user {
            Some(u) => format!("{u}@{host}"),
            None => host.to_string(),
        };
        let wrapped = format!("bash -lc {}", shell_quote(cmd));
        Command::new("ssh")
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                &format!("ConnectTimeout={timeout_secs}"),
                "--",
                &target,
                &wrapped,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    };

    // Read stdout/stderr in background threads to prevent pipe buffer deadlock.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stdout_pipe {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stderr_pipe {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });

    // Poll child with timeout.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs as u64);
    loop {
        match child.try_wait()? {
            Some(status) => {
                let stdout = stdout_thread.join().unwrap_or_default();
                let stderr = stderr_thread.join().unwrap_or_default();
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("command timed out after {timeout_secs}s"),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

/// Spawn a command on a remote host (or locally) with separately piped stdout and stderr.
pub fn spawn_remote(host: &str, cmd: &str, user: Option<&str>) -> std::io::Result<Child> {
    validate_ssh_target(host, "host").map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    if let Some(u) = user {
        validate_ssh_target(u, "user").map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    }

    if is_localhost(host) {
        return Command::new("bash")
            .args(["-lc", cmd])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
    }
    let target = match user {
        Some(u) => format!("{u}@{host}"),
        None => host.to_string(),
    };
    let wrapped = format!("bash -lc {}", shell_quote(cmd));
    Command::new("ssh")
        .args(["-o", "BatchMode=yes", "--", &target, &wrapped])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_quote_empty() {
        assert_eq!(shell_quote(""), "''");
    }

    #[test]
    fn test_shell_quote_safe() {
        assert_eq!(shell_quote("hello"), "hello");
        assert_eq!(shell_quote("/usr/bin/env"), "/usr/bin/env");
        assert_eq!(shell_quote("KEY=value"), "KEY=value");
    }

    #[test]
    fn test_shell_quote_spaces() {
        assert_eq!(shell_quote("hello world"), "'hello world'");
    }

    #[test]
    fn test_shell_quote_single_quotes() {
        assert_eq!(shell_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn test_shell_quote_special() {
        assert_eq!(shell_quote("$(rm -rf /)"), "'$(rm -rf /)'");
        assert_eq!(shell_quote("a;b"), "'a;b'");
    }

    #[test]
    fn test_validate_ssh_target_rejects_option_injection() {
        assert!(validate_ssh_target("-oProxyCommand=evil", "host").is_err());
        assert!(validate_ssh_target("-v", "host").is_err());
        assert!(validate_ssh_target("--version", "user").is_err());
    }

    #[test]
    fn test_validate_ssh_target_rejects_invalid_chars() {
        assert!(validate_ssh_target("host;rm", "host").is_err());
        assert!(validate_ssh_target("user$(id)", "user").is_err());
        assert!(validate_ssh_target("host name", "host").is_err());
        assert!(validate_ssh_target("", "host").is_err());
    }

    #[test]
    fn test_validate_ssh_target_allows_valid() {
        assert!(validate_ssh_target("my-host.example.com", "host").is_ok());
        assert!(validate_ssh_target("192.168.1.1", "host").is_ok());
        assert!(validate_ssh_target("user@host", "host").is_ok());
        assert!(validate_ssh_target("node_01", "host").is_ok());
        assert!(validate_ssh_target("user:name", "user").is_ok());
    }

    #[test]
    fn test_run_remote_rejects_malicious_host() {
        let result = run_remote("-oProxyCommand=evil", "echo hi", None, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_spawn_remote_rejects_malicious_user() {
        let result = spawn_remote("myhost", "echo hi", Some("-oProxyCommand=evil"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
