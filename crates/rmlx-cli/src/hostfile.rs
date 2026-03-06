use crate::ssh::validate_ssh_target;
use serde::{Deserialize, Serialize};

/// A single entry in the RMLX hostfile JSON array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostEntry {
    pub ssh: String,
    #[serde(default)]
    pub ips: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rdma: Option<Vec<Option<String>>>,
}

/// Load full host entries from a JSON hostfile.
pub fn load_host_entries(path: &str) -> Result<Vec<HostEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read hostfile {path}: {e}"))?;
    let entries: Vec<HostEntry> = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse hostfile {path}: {e}"))?;
    if entries.is_empty() {
        return Err("hostfile has no hosts".into());
    }
    for entry in &entries {
        validate_ssh_target(&entry.ssh, "hostfile ssh field")?;
    }
    Ok(entries)
}

/// Load hosts from a JSON hostfile, returning the SSH targets.
#[allow(dead_code)]
pub fn load_hosts_from_file(path: &str) -> Result<Vec<String>, String> {
    Ok(load_host_entries(path)?
        .into_iter()
        .map(|e| e.ssh)
        .collect())
}

/// Parse a comma-separated host list.
pub fn parse_hosts_csv(csv: &str) -> Result<Vec<String>, String> {
    let hosts: Vec<String> = csv
        .split(',')
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .collect();
    if hosts.is_empty() {
        return Err("no hosts parsed from --hosts".into());
    }
    for h in &hosts {
        validate_ssh_target(h, "host")?;
    }
    Ok(hosts)
}

/// Resolve hosts from either --hosts CSV or --hostfile path.
#[allow(dead_code)]
pub fn resolve_hosts(
    hosts_csv: Option<&str>,
    hostfile: Option<&str>,
) -> Result<Vec<String>, String> {
    match (hosts_csv, hostfile) {
        (Some(csv), None) => parse_hosts_csv(csv),
        (None, Some(path)) => load_hosts_from_file(path),
        (Some(_), Some(_)) => Err("provide only one of --hosts or --hostfile".into()),
        (None, None) => Err("provide exactly one of --hosts or --hostfile".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hosts_csv() {
        assert_eq!(
            parse_hosts_csv("node1,node2").unwrap(),
            vec!["node1", "node2"]
        );
        assert_eq!(parse_hosts_csv(" a , b , ").unwrap(), vec!["a", "b"]);
        assert!(parse_hosts_csv(",,").is_err());
    }

    #[test]
    fn test_host_entry_roundtrip() {
        let entry = HostEntry {
            ssh: "node1".into(),
            ips: vec!["192.168.1.1".into()],
            rdma: Some(vec![None, Some("mlx5_0".into())]),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: HostEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ssh, "node1");
        assert_eq!(back.ips, vec!["192.168.1.1"]);
    }

    #[test]
    fn test_resolve_hosts_both() {
        assert!(resolve_hosts(Some("a"), Some("b")).is_err());
    }

    #[test]
    fn test_resolve_hosts_neither() {
        assert!(resolve_hosts(None, None).is_err());
    }

    #[test]
    fn test_parse_hosts_csv_rejects_injection() {
        let result = parse_hosts_csv("-oProxyCommand=evil,node1");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("starts with '-'"));
    }

    #[test]
    fn test_parse_hosts_csv_rejects_invalid_chars() {
        let result = parse_hosts_csv("node1,host;rm -rf /");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_hosts_csv_valid() {
        let result = parse_hosts_csv("my-host.example.com,192.168.1.1");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec!["my-host.example.com", "192.168.1.1"]);
    }
}
