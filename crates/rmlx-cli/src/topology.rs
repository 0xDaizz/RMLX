use std::fmt;

/// Detected interconnect between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum Interconnect {
    /// Loopback (same host)
    Loopback,
    /// Unknown / not probed
    Unknown,
    /// 10GbE / 25GbE Ethernet
    Ethernet,
    /// Thunderbolt 4/3 (40 Gbps)
    Thunderbolt4,
    /// Thunderbolt 5 (80 Gbps bidirectional)
    Thunderbolt5,
    /// InfiniBand RDMA
    Rdma,
}

impl fmt::Display for Interconnect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Interconnect::Thunderbolt5 => write!(f, "tb5"),
            Interconnect::Thunderbolt4 => write!(f, "tb4"),
            Interconnect::Ethernet => write!(f, "tcp"),
            Interconnect::Rdma => write!(f, "rdma"),
            Interconnect::Loopback => write!(f, "loopback"),
            Interconnect::Unknown => write!(f, "unknown"),
        }
    }
}

/// Probe the local machine for Thunderbolt interfaces.
/// On macOS, uses `system_profiler SPThunderboltDataType -json` to detect TB ports.
/// Returns a list of detected TB bus speeds.
#[cfg(target_os = "macos")]
pub fn probe_thunderbolt() -> Vec<Interconnect> {
    let output = match std::process::Command::new("system_profiler")
        .args(["SPThunderboltDataType", "-json"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    if !output.status.success() {
        return Vec::new();
    }

    let json_str = match std::str::from_utf8(&output.stdout) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    parse_thunderbolt_json(json_str)
}

/// Non-macOS fallback: always returns empty.
#[cfg(not(target_os = "macos"))]
pub fn probe_thunderbolt() -> Vec<Interconnect> {
    Vec::new()
}

/// Parse the JSON output from `system_profiler SPThunderboltDataType -json`.
/// Searches recursively for any string values containing speed indicators.
fn parse_thunderbolt_json(json_str: &str) -> Vec<Interconnect> {
    let value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();
    collect_speeds(&value, &mut results);
    results
}

/// Recursively walk the JSON value looking for speed-related fields.
fn collect_speeds(value: &serde_json::Value, out: &mut Vec<Interconnect>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let key_lower = key.to_lowercase();
                if key_lower.contains("speed") || key_lower.contains("link_speed") {
                    if let Some(s) = val.as_str() {
                        if let Some(ic) = classify_speed(s) {
                            out.push(ic);
                        }
                    }
                }
                collect_speeds(val, out);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_speeds(item, out);
            }
        }
        _ => {}
    }
}

/// Classify a speed string into an Interconnect variant.
fn classify_speed(speed: &str) -> Option<Interconnect> {
    // TB5 reports ~80 Gb/s, TB4/3 reports ~40 Gb/s
    if speed.contains("80") {
        Some(Interconnect::Thunderbolt5)
    } else if speed.contains("40") {
        Some(Interconnect::Thunderbolt4)
    } else {
        None
    }
}

/// Given a set of hosts, determine the best interconnect for each pair.
/// Returns a `hosts.len() x hosts.len()` matrix.
///
/// This is a best-effort heuristic:
/// - Diagonal entries are `Loopback`
/// - If local TB probing finds TB5/TB4, assume all non-self pairs use that
/// - Otherwise, mark as `Unknown`
#[allow(dead_code)]
pub fn detect_topology(hosts: &[String]) -> Vec<Vec<Interconnect>> {
    let n = hosts.len();
    let local_tb = probe_thunderbolt();

    // Pick the best TB interconnect detected locally
    let best_tb = local_tb
        .iter()
        .max()
        .cloned()
        .unwrap_or(Interconnect::Unknown);

    let mut matrix = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                row.push(Interconnect::Loopback);
            } else {
                row.push(best_tb.clone());
            }
        }
        matrix.push(row);
    }
    matrix
}

/// Resolve the backend string from `--backend auto` given topology and hostfile info.
///
/// Priority:
/// 1. If any host has RDMA devices in the hostfile -> "rdma"
/// 2. If TB5 detected locally -> "tb5"
/// 3. If TB4 detected locally -> "tb4"
/// 4. Fallback -> "tcp"
pub fn resolve_auto_backend(has_rdma: bool, local_tb: &[Interconnect]) -> &'static str {
    if has_rdma {
        return "rdma";
    }

    // Pick the best TB interconnect
    let best = local_tb.iter().max();
    match best {
        Some(Interconnect::Thunderbolt5) => "tb5",
        Some(Interconnect::Thunderbolt4) => "tb4",
        _ => "tcp",
    }
}

/// Validate that a backend string is a known value.
pub fn validate_backend(backend: &str) -> Result<(), String> {
    match backend {
        "auto" | "rdma" | "tb5" | "tb4" | "tcp" | "loopback" => Ok(()),
        other => Err(format!(
            "unknown backend {other:?}: expected one of auto, rdma, tb5, tb4, tcp, loopback"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interconnect_display() {
        assert_eq!(Interconnect::Thunderbolt5.to_string(), "tb5");
        assert_eq!(Interconnect::Thunderbolt4.to_string(), "tb4");
        assert_eq!(Interconnect::Ethernet.to_string(), "tcp");
        assert_eq!(Interconnect::Rdma.to_string(), "rdma");
        assert_eq!(Interconnect::Loopback.to_string(), "loopback");
        assert_eq!(Interconnect::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_interconnect_ordering() {
        // Ordering: Loopback < Unknown < Ethernet < TB4 < TB5 < Rdma
        assert!(Interconnect::Loopback < Interconnect::Unknown);
        assert!(Interconnect::Unknown < Interconnect::Ethernet);
        assert!(Interconnect::Ethernet < Interconnect::Thunderbolt4);
        assert!(Interconnect::Thunderbolt4 < Interconnect::Thunderbolt5);
        assert!(Interconnect::Thunderbolt5 < Interconnect::Rdma);
    }

    #[test]
    fn test_probe_thunderbolt_no_panic() {
        // Should not panic regardless of environment (CI has no TB hardware)
        let result = probe_thunderbolt();
        // Result can be empty or populated; just verify no panic
        let _ = result;
    }

    #[test]
    fn test_parse_thunderbolt_json_empty() {
        assert_eq!(parse_thunderbolt_json("{}"), Vec::<Interconnect>::new());
        assert_eq!(parse_thunderbolt_json("[]"), Vec::<Interconnect>::new());
        assert_eq!(
            parse_thunderbolt_json("not json"),
            Vec::<Interconnect>::new()
        );
    }

    #[test]
    fn test_parse_thunderbolt_json_tb5() {
        let json = r#"{
            "SPThunderboltDataType": [{
                "_name": "Thunderbolt Bus",
                "device_speed": "80 Gb/s"
            }]
        }"#;
        let result = parse_thunderbolt_json(json);
        assert_eq!(result, vec![Interconnect::Thunderbolt5]);
    }

    #[test]
    fn test_parse_thunderbolt_json_tb4() {
        let json = r#"{
            "SPThunderboltDataType": [{
                "_name": "Thunderbolt Bus",
                "link_speed": "Up to 40 Gb/s"
            }]
        }"#;
        let result = parse_thunderbolt_json(json);
        assert_eq!(result, vec![Interconnect::Thunderbolt4]);
    }

    #[test]
    fn test_parse_thunderbolt_json_multiple() {
        let json = r#"{
            "SPThunderboltDataType": [
                {"device_speed": "80 Gb/s"},
                {"device_speed": "40 Gb/s"}
            ]
        }"#;
        let result = parse_thunderbolt_json(json);
        assert_eq!(
            result,
            vec![Interconnect::Thunderbolt5, Interconnect::Thunderbolt4]
        );
    }

    #[test]
    fn test_parse_thunderbolt_json_no_speed() {
        let json = r#"{
            "SPThunderboltDataType": [{
                "_name": "Thunderbolt Bus",
                "vendor": "Apple"
            }]
        }"#;
        let result = parse_thunderbolt_json(json);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_topology_single_host() {
        let hosts = vec!["host1".into()];
        let topo = detect_topology(&hosts);
        assert_eq!(topo.len(), 1);
        assert_eq!(topo[0].len(), 1);
        assert_eq!(topo[0][0], Interconnect::Loopback);
    }

    #[test]
    fn test_detect_topology_two_hosts() {
        let hosts = vec!["host1".into(), "host2".into()];
        let topo = detect_topology(&hosts);
        assert_eq!(topo.len(), 2);
        assert_eq!(topo[0][0], Interconnect::Loopback);
        assert_eq!(topo[1][1], Interconnect::Loopback);
        // Off-diagonal entries should be the same (whatever local probe found)
        assert_eq!(topo[0][1], topo[1][0]);
    }

    #[test]
    fn test_resolve_auto_backend_rdma() {
        assert_eq!(resolve_auto_backend(true, &[]), "rdma");
        assert_eq!(
            resolve_auto_backend(true, &[Interconnect::Thunderbolt5]),
            "rdma"
        );
    }

    #[test]
    fn test_resolve_auto_backend_tb5() {
        assert_eq!(
            resolve_auto_backend(false, &[Interconnect::Thunderbolt5]),
            "tb5"
        );
        assert_eq!(
            resolve_auto_backend(
                false,
                &[Interconnect::Thunderbolt4, Interconnect::Thunderbolt5]
            ),
            "tb5"
        );
    }

    #[test]
    fn test_resolve_auto_backend_tb4() {
        assert_eq!(
            resolve_auto_backend(false, &[Interconnect::Thunderbolt4]),
            "tb4"
        );
    }

    #[test]
    fn test_resolve_auto_backend_tcp_fallback() {
        assert_eq!(resolve_auto_backend(false, &[]), "tcp");
        assert_eq!(resolve_auto_backend(false, &[Interconnect::Unknown]), "tcp");
        assert_eq!(
            resolve_auto_backend(false, &[Interconnect::Loopback]),
            "tcp"
        );
    }

    #[test]
    fn test_validate_backend() {
        assert!(validate_backend("auto").is_ok());
        assert!(validate_backend("rdma").is_ok());
        assert!(validate_backend("tb5").is_ok());
        assert!(validate_backend("tb4").is_ok());
        assert!(validate_backend("tcp").is_ok());
        assert!(validate_backend("invalid").is_err());
        assert!(validate_backend("").is_err());
        assert!(validate_backend("AUTO").is_err());
    }

    #[test]
    fn test_classify_speed() {
        assert_eq!(classify_speed("80 Gb/s"), Some(Interconnect::Thunderbolt5));
        assert_eq!(
            classify_speed("Up to 40 Gb/s"),
            Some(Interconnect::Thunderbolt4)
        );
        assert_eq!(classify_speed("10 Gb/s"), None);
        assert_eq!(classify_speed(""), None);
    }
}
