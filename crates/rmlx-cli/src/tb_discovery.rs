use std::collections::HashMap;

use crate::ssh;

// ---- Data Structures ----

/// A single Thunderbolt port on a host.
pub struct TbPort {
    /// Network interface name, e.g., "en5"
    pub iface: String,
    /// This port's domain UUID (from `domain_uuid_key` on the bus entry)
    pub uuid: String,
    /// Peer's domain UUID if a device is connected (from `_items[0].domain_uuid_key`)
    pub connected_to: Option<String>,
}

/// All Thunderbolt ports detected on a single host.
pub struct TbHost {
    /// Device name (from `device_name_key`) or hostname
    pub name: String,
    /// Detected TB ports with resolved interface names
    pub ports: Vec<TbPort>,
}

/// A point-to-point TB link between two hosts.
pub struct TbLink {
    pub src_host: usize,
    pub src_iface: String,
    pub src_ip: String,
    pub dst_host: usize,
    pub dst_iface: String,
    pub dst_ip: String,
}

/// Complete Thunderbolt topology across all hosts.
pub struct TbTopology {
    #[allow(dead_code)]
    pub hosts: Vec<TbHost>,
    pub links: Vec<TbLink>,
}

// ---- Helpers ----

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

// ---- Parsing Functions ----

/// Parse `networksetup -listallhardwareports` output.
/// Returns map: port name (e.g. "Thunderbolt 5") → interface name (e.g. "en5").
pub fn parse_hardware_ports(output: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_port: Option<String> = None;

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Hardware Port:") {
            current_port = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("Device:") {
            if let Some(port) = current_port.take() {
                let device = rest.trim().to_string();
                if !device.is_empty() {
                    map.insert(port, device);
                }
            }
        }
    }

    map
}

/// Parse `system_profiler SPThunderboltDataType -json` output for one host.
/// Uses `hardware_ports` map to resolve receptacle tags → interface names.
pub fn parse_tb_profiler(
    profiler_json: &str,
    hardware_ports: &HashMap<String, String>,
) -> Result<TbHost, String> {
    let root: serde_json::Value =
        serde_json::from_str(profiler_json).map_err(|e| format!("invalid JSON: {e}"))?;

    let buses = root
        .get("SPThunderboltDataType")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing SPThunderboltDataType array".to_string())?;

    let mut host_name = String::new();
    let mut ports = Vec::new();

    for bus in buses {
        // Extract host device name from the bus entry
        if host_name.is_empty() {
            if let Some(name) = bus.get("device_name_key").and_then(|v| v.as_str()) {
                host_name = name.to_string();
            }
        }

        // This port's UUID
        let uuid = match bus.get("domain_uuid_key").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => continue,
        };

        // Resolve interface from receptacle tag
        let iface = resolve_iface_from_bus(bus, hardware_ports);
        let iface = match iface {
            Some(i) => i,
            None => continue, // skip ports we cannot map to a network interface
        };

        // Check for connected peer device
        let connected_to = bus
            .get("_items")
            .and_then(|v| v.as_array())
            .and_then(|items| items.first())
            .and_then(|item| item.get("domain_uuid_key"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        ports.push(TbPort {
            iface,
            uuid,
            connected_to,
        });
    }

    if host_name.is_empty() {
        host_name = "unknown".to_string();
    }

    Ok(TbHost {
        name: host_name,
        ports,
    })
}

/// Try to resolve the network interface name for a TB bus entry.
/// Looks at `receptacle_1_tag.receptacle_id_key` and maps "Thunderbolt {tag}" → iface.
fn resolve_iface_from_bus(
    bus: &serde_json::Value,
    hardware_ports: &HashMap<String, String>,
) -> Option<String> {
    // Try receptacle_1_tag, receptacle_2_tag, etc.
    for suffix in &["receptacle_1_tag", "receptacle_2_tag"] {
        if let Some(tag_obj) = bus.get(*suffix) {
            if let Some(tag) = tag_obj.get("receptacle_id_key").and_then(|v| v.as_str()) {
                let port_name = format!("Thunderbolt {tag}");
                if let Some(iface) = hardware_ports.get(&port_name) {
                    return Some(iface.clone());
                }
            }
        }
    }
    None
}

// ---- Discovery ----

/// SSH to each host, collect system_profiler + networksetup, parse, match UUIDs.
pub fn discover(
    hosts: &[String],
    user: Option<&str>,
    timeout: u32,
    verbose: bool,
) -> Result<TbTopology, String> {
    let mut tb_hosts = Vec::with_capacity(hosts.len());

    for host in hosts {
        if verbose {
            println!("[{host}] querying TB topology");
        }

        // Get hardware port map
        let hw_output = must_ok(
            ssh::run_remote(host, "networksetup -listallhardwareports", user, timeout),
            &format!("failed to query hardware ports on {host}"),
        )?;
        let hw_ports = parse_hardware_ports(&hw_output);

        // Get system profiler TB data
        let profiler_output = must_ok(
            ssh::run_remote(
                host,
                "system_profiler SPThunderboltDataType -json",
                user,
                timeout,
            ),
            &format!("failed to query TB profiler on {host}"),
        )?;

        let mut tb_host = parse_tb_profiler(&profiler_output, &hw_ports)?;
        if tb_host.name == "unknown" {
            tb_host.name = host.clone();
        }

        if verbose {
            println!(
                "[{host}] found {} TB port(s): {}",
                tb_host.ports.len(),
                tb_host
                    .ports
                    .iter()
                    .map(|p| {
                        format!(
                            "{}({}→{})",
                            p.iface,
                            &p.uuid[..8.min(p.uuid.len())],
                            p.connected_to
                                .as_ref()
                                .map(|u| &u[..8.min(u.len())])
                                .unwrap_or("none")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        tb_hosts.push(tb_host);
    }

    // Match UUIDs across hosts to discover links
    let links = match_links(&tb_hosts);

    Ok(TbTopology {
        hosts: tb_hosts,
        links,
    })
}

/// Match connected port UUIDs across hosts to identify point-to-point links.
fn match_links(hosts: &[TbHost]) -> Vec<TbLink> {
    // Build index: uuid → (host_idx, port_idx)
    let mut uuid_to_port: HashMap<&str, (usize, usize)> = HashMap::new();
    for (hi, host) in hosts.iter().enumerate() {
        for (pi, port) in host.ports.iter().enumerate() {
            uuid_to_port.insert(&port.uuid, (hi, pi));
        }
    }

    let mut links = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize, usize, usize)> =
        std::collections::HashSet::new();

    for (hi, host) in hosts.iter().enumerate() {
        for (pi, port) in host.ports.iter().enumerate() {
            if let Some(peer_uuid) = &port.connected_to {
                if let Some(&(pj, qj)) = uuid_to_port.get(peer_uuid.as_str()) {
                    // Only create links between different hosts
                    if hi == pj {
                        continue;
                    }
                    // Deduplicate: only add if we haven't seen this pair
                    let key = if hi < pj {
                        (hi, pi, pj, qj)
                    } else {
                        (pj, qj, hi, pi)
                    };
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);

                    links.push(TbLink {
                        src_host: hi,
                        src_iface: port.iface.clone(),
                        src_ip: String::new(),
                        dst_host: pj,
                        dst_iface: hosts[pj].ports[qj].iface.clone(),
                        dst_ip: String::new(),
                    });
                }
            }
        }
    }

    links
}

// ---- IP Assignment ----

impl TbTopology {
    /// Assign /30 point-to-point IPs to each link (MLX scheme).
    /// Uses 192.168.{ip0}.{ip1+1} / 192.168.{ip0}.{ip1+2} with /30 subnets.
    pub fn assign_ips(&mut self) {
        let mut ip0: u32 = 0;
        let mut ip1: u32 = 0;

        for link in &mut self.links {
            link.src_ip = format!("192.168.{ip0}.{}", ip1 + 1);
            link.dst_ip = format!("192.168.{ip0}.{}", ip1 + 2);
            ip1 += 4; // /30 subnet = 4 addresses
            if ip1 > 255 {
                ip0 += 1;
                ip1 = 0;
            }
        }
    }

    /// Generate setup commands for a specific host.
    /// Returns Vec of "sudo ifconfig ..." and "sudo route ..." commands.
    pub fn setup_commands(&self, host_idx: usize) -> Vec<String> {
        let mut cmds = Vec::new();

        // Always bring down bridge0 first
        cmds.push("sudo ifconfig bridge0 down 2>/dev/null || true".to_string());

        for link in &self.links {
            if link.src_host == host_idx {
                // This host is the source side of the link
                cmds.push(format!(
                    "sudo ifconfig {} inet {} netmask 255.255.255.252",
                    link.src_iface, link.src_ip,
                ));
                cmds.push(format!(
                    "sudo route change {} -interface {}",
                    link.dst_ip, link.src_iface,
                ));
            } else if link.dst_host == host_idx {
                // This host is the destination side of the link
                cmds.push(format!(
                    "sudo ifconfig {} inet {} netmask 255.255.255.252",
                    link.dst_iface, link.dst_ip,
                ));
                cmds.push(format!(
                    "sudo route change {} -interface {}",
                    link.src_ip, link.dst_iface,
                ));
            }
        }

        cmds
    }

    /// Apply setup on all hosts via SSH.
    pub fn apply_setup(
        &self,
        hosts: &[String],
        user: Option<&str>,
        timeout: u32,
        verbose: bool,
    ) -> Result<(), String> {
        for (idx, host) in hosts.iter().enumerate() {
            let cmds = self.setup_commands(idx);
            if cmds.is_empty() {
                continue;
            }
            let script = cmds.join("; ");
            if verbose {
                println!("[{host}] applying: {script}");
            }
            must_ok(
                ssh::run_remote(host, &script, user, timeout),
                &format!("setup failed on {host}"),
            )?;
        }
        Ok(())
    }

    /// Get data-plane IPs for a host (for hostfile "ips" field).
    /// Returns the IPs assigned to this host's TB interfaces.
    pub fn data_ips(&self, host_idx: usize) -> Vec<String> {
        let mut ips = Vec::new();
        for link in &self.links {
            if link.src_host == host_idx {
                ips.push(link.src_ip.clone());
            } else if link.dst_host == host_idx {
                ips.push(link.dst_ip.clone());
            }
        }
        ips
    }

    /// Build RDMA device map for a host (for hostfile "rdma" field).
    /// `rdma[peer]` = `Some("rdma_{iface}")` for connected peers, `None` for self.
    pub fn rdma_map(&self, host_idx: usize, num_hosts: usize) -> Vec<Option<String>> {
        // Build a lookup: peer_host_idx → our local iface
        let mut peer_iface: HashMap<usize, String> = HashMap::new();
        for link in &self.links {
            if link.src_host == host_idx {
                peer_iface.insert(link.dst_host, link.src_iface.clone());
            } else if link.dst_host == host_idx {
                peer_iface.insert(link.src_host, link.dst_iface.clone());
            }
        }

        let mut map = Vec::with_capacity(num_hosts);
        for peer in 0..num_hosts {
            if peer == host_idx {
                map.push(None);
            } else if let Some(iface) = peer_iface.get(&peer) {
                map.push(Some(format!("rdma_{iface}")));
            } else {
                // No direct TB link to this peer
                map.push(None);
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_hardware_ports tests ----

    #[test]
    fn test_parse_hardware_ports_basic() {
        let output = "\
Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: aa:bb:cc:dd:ee:01

Hardware Port: Thunderbolt 5
Device: en5
Ethernet Address: aa:bb:cc:dd:ee:05

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:00
";
        let map = parse_hardware_ports(output);
        assert_eq!(map.get("Thunderbolt 1"), Some(&"en2".to_string()));
        assert_eq!(map.get("Thunderbolt 5"), Some(&"en5".to_string()));
        assert_eq!(map.get("Wi-Fi"), Some(&"en0".to_string()));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_parse_hardware_ports_empty() {
        let map = parse_hardware_ports("");
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_hardware_ports_no_device_line() {
        let output = "Hardware Port: Thunderbolt 1\nEthernet Address: aa:bb\n";
        let map = parse_hardware_ports(output);
        assert!(map.is_empty());
    }

    // ---- parse_tb_profiler tests ----

    #[test]
    fn test_parse_tb_profiler_connected() {
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "_name": "thunderboltusb4_bus_0",
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "AAAA-BBBB-CCCC",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "5"
                    },
                    "_items": [
                        {
                            "domain_uuid_key": "DDDD-EEEE-FFFF",
                            "device_name_key": "Mac Studio"
                        }
                    ]
                }
            ]
        }"#;

        let mut hw = HashMap::new();
        hw.insert("Thunderbolt 5".to_string(), "en5".to_string());

        let host = parse_tb_profiler(json, &hw).unwrap();
        assert_eq!(host.name, "Mac Studio");
        assert_eq!(host.ports.len(), 1);
        assert_eq!(host.ports[0].iface, "en5");
        assert_eq!(host.ports[0].uuid, "AAAA-BBBB-CCCC");
        assert_eq!(
            host.ports[0].connected_to,
            Some("DDDD-EEEE-FFFF".to_string())
        );
    }

    #[test]
    fn test_parse_tb_profiler_disconnected() {
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "_name": "thunderboltusb4_bus_0",
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "AAAA-BBBB-CCCC",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "5"
                    }
                }
            ]
        }"#;

        let mut hw = HashMap::new();
        hw.insert("Thunderbolt 5".to_string(), "en5".to_string());

        let host = parse_tb_profiler(json, &hw).unwrap();
        assert_eq!(host.ports.len(), 1);
        assert_eq!(host.ports[0].connected_to, None);
    }

    #[test]
    fn test_parse_tb_profiler_no_buses() {
        let json = r#"{"SPThunderboltDataType": []}"#;
        let hw = HashMap::new();
        let host = parse_tb_profiler(json, &hw).unwrap();
        assert!(host.ports.is_empty());
        assert_eq!(host.name, "unknown");
    }

    #[test]
    fn test_parse_tb_profiler_missing_key() {
        let json = r#"{"other_key": []}"#;
        let hw = HashMap::new();
        let result = parse_tb_profiler(json, &hw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tb_profiler_invalid_json() {
        let hw = HashMap::new();
        let result = parse_tb_profiler("not json {{{", &hw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tb_profiler_no_matching_hw_port() {
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "_name": "thunderboltusb4_bus_0",
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "AAAA-BBBB-CCCC",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "99"
                    }
                }
            ]
        }"#;
        // No "Thunderbolt 99" in hardware ports → port skipped
        let mut hw = HashMap::new();
        hw.insert("Thunderbolt 5".to_string(), "en5".to_string());

        let host = parse_tb_profiler(json, &hw).unwrap();
        assert!(host.ports.is_empty());
    }

    #[test]
    fn test_parse_tb_profiler_multiple_buses() {
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "_name": "bus_0",
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "UUID-A",
                    "receptacle_1_tag": { "receptacle_id_key": "1" },
                    "_items": [{ "domain_uuid_key": "UUID-X" }]
                },
                {
                    "_name": "bus_1",
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "UUID-B",
                    "receptacle_1_tag": { "receptacle_id_key": "5" },
                    "_items": [{ "domain_uuid_key": "UUID-Y" }]
                }
            ]
        }"#;

        let mut hw = HashMap::new();
        hw.insert("Thunderbolt 1".to_string(), "en2".to_string());
        hw.insert("Thunderbolt 5".to_string(), "en5".to_string());

        let host = parse_tb_profiler(json, &hw).unwrap();
        assert_eq!(host.ports.len(), 2);
        assert_eq!(host.ports[0].iface, "en2");
        assert_eq!(host.ports[0].uuid, "UUID-A");
        assert_eq!(host.ports[1].iface, "en5");
        assert_eq!(host.ports[1].uuid, "UUID-B");
    }

    // ---- assign_ips tests ----

    fn make_2node_topology() -> TbTopology {
        TbTopology {
            hosts: vec![
                TbHost {
                    name: "host0".into(),
                    ports: vec![TbPort {
                        iface: "en5".into(),
                        uuid: "UUID-A".into(),
                        connected_to: Some("UUID-B".into()),
                    }],
                },
                TbHost {
                    name: "host1".into(),
                    ports: vec![TbPort {
                        iface: "en5".into(),
                        uuid: "UUID-B".into(),
                        connected_to: Some("UUID-A".into()),
                    }],
                },
            ],
            links: vec![TbLink {
                src_host: 0,
                src_iface: "en5".into(),
                src_ip: String::new(),
                dst_host: 1,
                dst_iface: "en5".into(),
                dst_ip: String::new(),
            }],
        }
    }

    fn make_3node_topology() -> TbTopology {
        // 3 hosts: 0↔1 via en5, 0↔2 via en2, 1↔2 via en2
        TbTopology {
            hosts: vec![
                TbHost {
                    name: "h0".into(),
                    ports: vec![
                        TbPort {
                            iface: "en5".into(),
                            uuid: "A0".into(),
                            connected_to: Some("B0".into()),
                        },
                        TbPort {
                            iface: "en2".into(),
                            uuid: "A1".into(),
                            connected_to: Some("C0".into()),
                        },
                    ],
                },
                TbHost {
                    name: "h1".into(),
                    ports: vec![
                        TbPort {
                            iface: "en5".into(),
                            uuid: "B0".into(),
                            connected_to: Some("A0".into()),
                        },
                        TbPort {
                            iface: "en2".into(),
                            uuid: "B1".into(),
                            connected_to: Some("C1".into()),
                        },
                    ],
                },
                TbHost {
                    name: "h2".into(),
                    ports: vec![
                        TbPort {
                            iface: "en5".into(),
                            uuid: "C0".into(),
                            connected_to: Some("A1".into()),
                        },
                        TbPort {
                            iface: "en2".into(),
                            uuid: "C1".into(),
                            connected_to: Some("B1".into()),
                        },
                    ],
                },
            ],
            links: vec![
                TbLink {
                    src_host: 0,
                    src_iface: "en5".into(),
                    src_ip: String::new(),
                    dst_host: 1,
                    dst_iface: "en5".into(),
                    dst_ip: String::new(),
                },
                TbLink {
                    src_host: 0,
                    src_iface: "en2".into(),
                    src_ip: String::new(),
                    dst_host: 2,
                    dst_iface: "en5".into(),
                    dst_ip: String::new(),
                },
                TbLink {
                    src_host: 1,
                    src_iface: "en2".into(),
                    src_ip: String::new(),
                    dst_host: 2,
                    dst_iface: "en2".into(),
                    dst_ip: String::new(),
                },
            ],
        }
    }

    #[test]
    fn test_assign_ips_2node() {
        let mut topo = make_2node_topology();
        topo.assign_ips();

        assert_eq!(topo.links[0].src_ip, "192.168.0.1");
        assert_eq!(topo.links[0].dst_ip, "192.168.0.2");
    }

    #[test]
    fn test_assign_ips_3node() {
        let mut topo = make_3node_topology();
        topo.assign_ips();

        // Link 0: 192.168.0.1/2
        assert_eq!(topo.links[0].src_ip, "192.168.0.1");
        assert_eq!(topo.links[0].dst_ip, "192.168.0.2");
        // Link 1: 192.168.0.5/6
        assert_eq!(topo.links[1].src_ip, "192.168.0.5");
        assert_eq!(topo.links[1].dst_ip, "192.168.0.6");
        // Link 2: 192.168.0.9/10
        assert_eq!(topo.links[2].src_ip, "192.168.0.9");
        assert_eq!(topo.links[2].dst_ip, "192.168.0.10");
    }

    #[test]
    fn test_assign_ips_wraps_octet() {
        // Create enough links to wrap past ip1=255
        let mut topo = TbTopology {
            hosts: Vec::new(),
            links: Vec::new(),
        };
        // 64 links × 4 = 256, so the 64th link (idx=63) should be 192.168.0.253/254
        // and the 65th (idx=64) should wrap to 192.168.1.1/2
        for i in 0..65 {
            topo.links.push(TbLink {
                src_host: 0,
                src_iface: format!("en{i}"),
                src_ip: String::new(),
                dst_host: 1,
                dst_iface: format!("en{i}"),
                dst_ip: String::new(),
            });
        }
        topo.assign_ips();

        // Link 63: ip1 = 63*4 = 252 → .253/.254
        assert_eq!(topo.links[63].src_ip, "192.168.0.253");
        assert_eq!(topo.links[63].dst_ip, "192.168.0.254");
        // Link 64: ip1 = 64*4 = 256 > 255 → ip0=1, ip1=0 → .1/.2
        assert_eq!(topo.links[64].src_ip, "192.168.1.1");
        assert_eq!(topo.links[64].dst_ip, "192.168.1.2");
    }

    // ---- setup_commands tests ----

    #[test]
    fn test_setup_commands_2node() {
        let mut topo = make_2node_topology();
        topo.assign_ips();

        let cmds0 = topo.setup_commands(0);
        assert_eq!(cmds0.len(), 3); // bridge0 down + ifconfig + route
        assert_eq!(cmds0[0], "sudo ifconfig bridge0 down 2>/dev/null || true");
        assert_eq!(
            cmds0[1],
            "sudo ifconfig en5 inet 192.168.0.1 netmask 255.255.255.252"
        );
        assert_eq!(cmds0[2], "sudo route change 192.168.0.2 -interface en5");

        let cmds1 = topo.setup_commands(1);
        assert_eq!(cmds1.len(), 3);
        assert_eq!(
            cmds1[1],
            "sudo ifconfig en5 inet 192.168.0.2 netmask 255.255.255.252"
        );
        assert_eq!(cmds1[2], "sudo route change 192.168.0.1 -interface en5");
    }

    #[test]
    fn test_setup_commands_no_links() {
        let topo = TbTopology {
            hosts: vec![TbHost {
                name: "lonely".into(),
                ports: Vec::new(),
            }],
            links: Vec::new(),
        };

        let cmds = topo.setup_commands(0);
        // Still includes bridge0 down
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0], "sudo ifconfig bridge0 down 2>/dev/null || true");
    }

    // ---- data_ips tests ----

    #[test]
    fn test_data_ips_2node() {
        let mut topo = make_2node_topology();
        topo.assign_ips();

        assert_eq!(topo.data_ips(0), vec!["192.168.0.1"]);
        assert_eq!(topo.data_ips(1), vec!["192.168.0.2"]);
    }

    #[test]
    fn test_data_ips_3node() {
        let mut topo = make_3node_topology();
        topo.assign_ips();

        // Host 0 participates in links 0 (src) and 1 (src)
        assert_eq!(topo.data_ips(0), vec!["192.168.0.1", "192.168.0.5"]);
        // Host 1 participates in links 0 (dst) and 2 (src)
        assert_eq!(topo.data_ips(1), vec!["192.168.0.2", "192.168.0.9"]);
        // Host 2 participates in links 1 (dst) and 2 (dst)
        assert_eq!(topo.data_ips(2), vec!["192.168.0.6", "192.168.0.10"]);
    }

    // ---- rdma_map tests ----

    #[test]
    fn test_rdma_map_2node() {
        let mut topo = make_2node_topology();
        topo.assign_ips();

        let map0 = topo.rdma_map(0, 2);
        assert_eq!(map0, vec![None, Some("rdma_en5".to_string())]);

        let map1 = topo.rdma_map(1, 2);
        assert_eq!(map1, vec![Some("rdma_en5".to_string()), None]);
    }

    #[test]
    fn test_rdma_map_3node() {
        let mut topo = make_3node_topology();
        topo.assign_ips();

        let map0 = topo.rdma_map(0, 3);
        assert_eq!(
            map0,
            vec![
                None,
                Some("rdma_en5".to_string()),
                Some("rdma_en2".to_string()),
            ]
        );

        let map1 = topo.rdma_map(1, 3);
        assert_eq!(
            map1,
            vec![
                Some("rdma_en5".to_string()),
                None,
                Some("rdma_en2".to_string()),
            ]
        );

        let map2 = topo.rdma_map(2, 3);
        assert_eq!(
            map2,
            vec![
                Some("rdma_en5".to_string()),
                Some("rdma_en2".to_string()),
                None,
            ]
        );
    }

    #[test]
    fn test_rdma_map_no_link_returns_none() {
        // Host 2 has no links to anyone
        let topo = TbTopology {
            hosts: vec![
                TbHost {
                    name: "h0".into(),
                    ports: Vec::new(),
                },
                TbHost {
                    name: "h1".into(),
                    ports: Vec::new(),
                },
                TbHost {
                    name: "h2".into(),
                    ports: Vec::new(),
                },
            ],
            links: vec![TbLink {
                src_host: 0,
                src_iface: "en5".into(),
                src_ip: "192.168.0.1".into(),
                dst_host: 1,
                dst_iface: "en5".into(),
                dst_ip: "192.168.0.2".into(),
            }],
        };

        let map2 = topo.rdma_map(2, 3);
        // Host 2 has no links → all peers are None (except self which is also None)
        assert_eq!(map2, vec![None, None, None]);
    }

    // ---- match_links tests ----

    #[test]
    fn test_match_links_bidirectional() {
        let hosts = vec![
            TbHost {
                name: "h0".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "AAA".into(),
                    connected_to: Some("BBB".into()),
                }],
            },
            TbHost {
                name: "h1".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "BBB".into(),
                    connected_to: Some("AAA".into()),
                }],
            },
        ];

        let links = match_links(&hosts);
        // Both sides see the connection, but we should only get one link
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].src_host, 0);
        assert_eq!(links[0].dst_host, 1);
    }

    #[test]
    fn test_match_links_no_connections() {
        let hosts = vec![
            TbHost {
                name: "h0".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "AAA".into(),
                    connected_to: None,
                }],
            },
            TbHost {
                name: "h1".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "BBB".into(),
                    connected_to: None,
                }],
            },
        ];

        let links = match_links(&hosts);
        assert!(links.is_empty());
    }

    #[test]
    fn test_match_links_ignores_same_host() {
        // A port connected to another port on the same host (shouldn't happen, but handle it)
        let hosts = vec![TbHost {
            name: "h0".into(),
            ports: vec![
                TbPort {
                    iface: "en5".into(),
                    uuid: "AAA".into(),
                    connected_to: Some("BBB".into()),
                },
                TbPort {
                    iface: "en2".into(),
                    uuid: "BBB".into(),
                    connected_to: Some("AAA".into()),
                },
            ],
        }];

        let links = match_links(&hosts);
        assert!(links.is_empty());
    }

    #[test]
    fn test_match_links_multiple_between_hosts() {
        // Two separate TB cables between the same pair of hosts
        let hosts = vec![
            TbHost {
                name: "h0".into(),
                ports: vec![
                    TbPort {
                        iface: "en5".into(),
                        uuid: "A0".into(),
                        connected_to: Some("B0".into()),
                    },
                    TbPort {
                        iface: "en2".into(),
                        uuid: "A1".into(),
                        connected_to: Some("B1".into()),
                    },
                ],
            },
            TbHost {
                name: "h1".into(),
                ports: vec![
                    TbPort {
                        iface: "en5".into(),
                        uuid: "B0".into(),
                        connected_to: Some("A0".into()),
                    },
                    TbPort {
                        iface: "en2".into(),
                        uuid: "B1".into(),
                        connected_to: Some("A1".into()),
                    },
                ],
            },
        ];

        let links = match_links(&hosts);
        // Two distinct cables → two links
        assert_eq!(links.len(), 2);
    }

    #[test]
    fn test_match_links_unresolved_peer_uuid() {
        // Port says connected_to UUID that doesn't exist in any host
        let hosts = vec![
            TbHost {
                name: "h0".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "AAA".into(),
                    connected_to: Some("NONEXISTENT".into()),
                }],
            },
            TbHost {
                name: "h1".into(),
                ports: vec![TbPort {
                    iface: "en5".into(),
                    uuid: "BBB".into(),
                    connected_to: None,
                }],
            },
        ];

        let links = match_links(&hosts);
        assert!(links.is_empty());
    }
}
