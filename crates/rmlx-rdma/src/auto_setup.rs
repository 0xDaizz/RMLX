//! Automatic RDMA network setup via Thunderbolt discovery.
//!
//! Mirrors the approach used by MLX's `config.py`:
//! 1. Discover Thunderbolt connections via `system_profiler`
//! 2. Map TB ports to network interfaces via `networksetup`
//! 3. Configure static IPs and routes on the TB interfaces
//! 4. Generate a [`DeviceMap`] for the discovered topology
//!
//! This module is macOS-only (Thunderbolt + RDMA over TB5).

use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::process::Command;

use crate::device_file::{DeviceFileError, DeviceMap};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from automatic RDMA network setup.
#[derive(Debug, thiserror::Error)]
pub enum AutoSetupError {
    /// No Thunderbolt connections found.
    #[error("no Thunderbolt connections found")]
    NoThunderboltConnections,

    /// Failed to run a system command.
    #[error("command failed: {0}")]
    CommandFailed(String),

    /// Failed to parse system profiler output.
    #[error("parse error: {0}")]
    ParseError(String),

    /// Network setup (ifconfig/route) failed.
    #[error("network setup failed: {0}")]
    NetworkSetup(String),

    /// Ping verification failed after setup.
    #[error("ping verification failed for {peer_ip} on {iface}")]
    PingFailed { iface: String, peer_ip: String },

    /// Invalid coordinator address.
    #[error("invalid coordinator address: {0}")]
    InvalidCoordinator(String),

    /// Device map construction failed.
    #[error("device map error: {0}")]
    DeviceMap(#[from] DeviceFileError),
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A discovered Thunderbolt peer connection.
#[derive(Debug, Clone)]
pub struct TbPeer {
    /// Network interface name (e.g. "en5").
    pub iface: String,
    /// RDMA device name (e.g. "rdma_en5").
    pub rdma_device: String,
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover Thunderbolt peer connections on this machine.
///
/// Runs `system_profiler SPThunderboltDataType -json` and
/// `networksetup -listallhardwareports` to find connected TB ports
/// and map them to network interface names.
pub fn discover_tb_peers() -> Result<Vec<TbPeer>, AutoSetupError> {
    // Step 1: Get hardware port mapping (port name → interface name)
    let hw_output = run_command("networksetup", &["-listallhardwareports"])
        .map_err(|e| AutoSetupError::CommandFailed(format!("networksetup: {e}")))?;
    let hw_ports = parse_hardware_ports(&hw_output);

    tracing::debug!(
        target: "rmlx_rdma::auto_setup",
        num_ports = hw_ports.len(),
        "parsed hardware ports",
    );

    // Step 2: Get Thunderbolt profiler data
    let profiler_output = run_command("system_profiler", &["SPThunderboltDataType", "-json"])
        .map_err(|e| AutoSetupError::CommandFailed(format!("system_profiler: {e}")))?;

    // Step 3: Parse and find connected TB ports
    let peers = parse_tb_profiler_for_peers(&profiler_output, &hw_ports)?;

    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        num_peers = peers.len(),
        peers = ?peers.iter().map(|p| &p.iface).collect::<Vec<_>>(),
        "discovered Thunderbolt peers",
    );

    Ok(peers)
}

/// Parse `networksetup -listallhardwareports` output.
/// Returns map: port name (e.g. "Thunderbolt 5") → interface name (e.g. "en5").
fn parse_hardware_ports(output: &str) -> HashMap<String, String> {
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

/// Parse `system_profiler SPThunderboltDataType -json` to find connected peers.
fn parse_tb_profiler_for_peers(
    json_str: &str,
    hardware_ports: &HashMap<String, String>,
) -> Result<Vec<TbPeer>, AutoSetupError> {
    let root: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        AutoSetupError::ParseError(format!("invalid JSON from system_profiler: {e}"))
    })?;

    let buses = root
        .get("SPThunderboltDataType")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            AutoSetupError::ParseError("missing SPThunderboltDataType array".to_string())
        })?;

    let mut peers = Vec::new();

    for bus in buses {
        // Check if a device is connected on this bus (has _items with at least one entry)
        let has_connected_device = bus
            .get("_items")
            .and_then(|v| v.as_array())
            .is_some_and(|items| !items.is_empty());

        if !has_connected_device {
            continue;
        }

        // Resolve network interface from receptacle tags
        let iface = resolve_iface_from_bus(bus, hardware_ports);
        let iface = match iface {
            Some(i) => i,
            None => continue, // skip ports we cannot map to a network interface
        };

        let rdma_device = format!("rdma_{iface}");
        peers.push(TbPeer { iface, rdma_device });
    }

    Ok(peers)
}

/// Try to resolve the network interface name for a TB bus entry.
/// Looks at receptacle tags and maps "Thunderbolt {tag}" → iface via hardware_ports.
fn resolve_iface_from_bus(
    bus: &serde_json::Value,
    hardware_ports: &HashMap<String, String>,
) -> Option<String> {
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

// ---------------------------------------------------------------------------
// Network setup
// ---------------------------------------------------------------------------

/// Configure network on a Thunderbolt interface with static IPs.
///
/// Performs:
/// 1. Remove interface from any bridge (non-fatal if fails)
/// 2. Bring interface down, then up with static IP
/// 3. Add/change route to peer IP
/// 4. Verify with ping
pub fn setup_network(iface: &str, local_ip: &str, peer_ip: &str) -> Result<(), AutoSetupError> {
    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        iface,
        local_ip,
        peer_ip,
        "configuring network interface",
    );

    // Step 1: Remove interface from bridge (ignore errors — may not be in a bridge)
    let bridge_result = run_sudo_command("ifconfig", &["bridge0", "deletem", iface]);
    match bridge_result {
        Ok(_) => {
            tracing::info!(target: "rmlx_rdma::auto_setup", iface, "removed from bridge0");
        }
        Err(e) => {
            tracing::debug!(
                target: "rmlx_rdma::auto_setup",
                iface,
                %e,
                "bridge removal skipped (not in bridge or bridge not present)",
            );
        }
    }

    // Also try bringing bridge0 down (non-fatal)
    let _ = run_sudo_command("ifconfig", &["bridge0", "down"]);

    // Step 2: Bring interface down then up with static IP
    let down_result = run_sudo_command("ifconfig", &[iface, "down"]);
    if let Err(e) = &down_result {
        tracing::warn!(
            target: "rmlx_rdma::auto_setup",
            iface,
            %e,
            "ifconfig down failed (continuing anyway)",
        );
    }

    run_sudo_command(
        "ifconfig",
        &[iface, "inet", local_ip, "netmask", "255.255.255.252", "up"],
    )
    .map_err(|e| {
        AutoSetupError::NetworkSetup(format!("failed to configure {iface} with {local_ip}: {e}"))
    })?;

    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        iface,
        local_ip,
        "interface configured with static IP",
    );

    // Step 3: Add route to peer IP via this interface
    // Try `route change` first (updates existing route), fall back to `route add`
    let route_result = run_sudo_command("route", &["-n", "change", peer_ip, "-interface", iface]);
    match route_result {
        Ok(_) => {
            tracing::info!(
                target: "rmlx_rdma::auto_setup",
                peer_ip,
                iface,
                "route updated",
            );
        }
        Err(_) => {
            // Fall back to route add
            let add_result =
                run_sudo_command("route", &["-n", "add", peer_ip, "-interface", iface]);
            match add_result {
                Ok(_) => {
                    tracing::info!(
                        target: "rmlx_rdma::auto_setup",
                        peer_ip,
                        iface,
                        "route added",
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "rmlx_rdma::auto_setup",
                        peer_ip,
                        iface,
                        %e,
                        "route add failed (connectivity may still work)",
                    );
                }
            }
        }
    }

    // Step 4: Verify connectivity with ping
    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        peer_ip,
        "verifying connectivity with ping",
    );

    let ping_ok = run_command("ping", &["-c", "1", "-W", "2", peer_ip]).is_ok();
    if !ping_ok {
        return Err(AutoSetupError::PingFailed {
            iface: iface.to_string(),
            peer_ip: peer_ip.to_string(),
        });
    }

    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        peer_ip,
        iface,
        "network setup complete — ping verified",
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// IP computation
// ---------------------------------------------------------------------------

/// Compute local and peer IPs for a 2-node setup.
///
/// Uses the convention: rank 0 gets the base IP from coordinator_addr,
/// rank 1 gets base + 1. For the 10.254.0.x/30 scheme:
/// - Rank 0: 10.254.0.5, Rank 1: 10.254.0.6
///
/// If coordinator_addr is not parseable as an IP (e.g. hostname), falls back
/// to the default 10.254.0.5/10.254.0.6 scheme.
pub fn compute_ips(rank: u32, coordinator_addr: &str) -> Result<(String, String), AutoSetupError> {
    // Strip port if present (e.g. "10.254.0.5:18520" → "10.254.0.5")
    let addr_part = coordinator_addr
        .rsplit_once(':')
        .map(|(host, _port)| host)
        .unwrap_or(coordinator_addr);

    // Try to parse as IPv4
    let base_ip: Ipv4Addr = addr_part.parse().unwrap_or_else(|_| {
        tracing::info!(
            target: "rmlx_rdma::auto_setup",
            coordinator_addr,
            "coordinator not a valid IP, using default 10.254.0.5",
        );
        Ipv4Addr::new(10, 254, 0, 5)
    });

    let base_u32 = u32::from(base_ip);

    // Rank 0 gets base_ip, rank 1 gets base_ip + 1
    let (local_u32, peer_u32) = if rank == 0 {
        (base_u32, base_u32 + 1)
    } else {
        (base_u32 + 1, base_u32)
    };

    let local_ip = Ipv4Addr::from(local_u32).to_string();
    let peer_ip = Ipv4Addr::from(peer_u32).to_string();

    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        rank,
        local_ip = %local_ip,
        peer_ip = %peer_ip,
        "computed IPs",
    );

    Ok((local_ip, peer_ip))
}

// ---------------------------------------------------------------------------
// Device map generation
// ---------------------------------------------------------------------------

/// Generate a [`DeviceMap`] for a 2-node Thunderbolt cluster.
///
/// Produces the matrix:
/// ```text
/// [[null, "rdma_enX"], ["rdma_enX", null]]
/// ```
pub fn auto_generate_device_map(
    _rank: u32,
    world_size: u32,
    rdma_device: &str,
) -> Result<DeviceMap, AutoSetupError> {
    if world_size != 2 {
        return Err(AutoSetupError::ParseError(format!(
            "auto device map generation only supports 2-node clusters, got world_size={world_size}"
        )));
    }

    let entries = vec![
        vec![None, Some(rdma_device.to_string())],
        vec![Some(rdma_device.to_string()), None],
    ];

    DeviceMap::from_matrix(entries).map_err(AutoSetupError::from)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Attempt automatic RDMA setup: discover TB peers, configure network, build device map.
///
/// This is the main entry point called from the distributed init path when no
/// device file is explicitly provided.
pub fn try_auto_setup(
    rank: u32,
    world_size: u32,
    coordinator_addr: &str,
) -> Result<DeviceMap, AutoSetupError> {
    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        rank,
        world_size,
        coordinator_addr,
        "starting automatic RDMA setup",
    );

    // 1. Discover TB peers
    let peers = discover_tb_peers()?;
    if peers.is_empty() {
        return Err(AutoSetupError::NoThunderboltConnections);
    }

    // 2. Pick the first connected TB peer interface
    // For 2-node setups there is typically one TB link
    let peer = &peers[0];
    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        iface = %peer.iface,
        rdma_device = %peer.rdma_device,
        "selected TB peer interface",
    );

    // 3. Determine local/peer IPs
    let (local_ip, peer_ip) = compute_ips(rank, coordinator_addr)?;

    // 4. Setup network
    setup_network(&peer.iface, &local_ip, &peer_ip)?;

    // 5. Generate device map
    let device_map = auto_generate_device_map(rank, world_size, &peer.rdma_device)?;

    tracing::info!(
        target: "rmlx_rdma::auto_setup",
        rank,
        world_size,
        rdma_device = %peer.rdma_device,
        "automatic RDMA setup completed successfully",
    );

    Ok(device_map)
}

// ---------------------------------------------------------------------------
// Command helpers
// ---------------------------------------------------------------------------

/// Run a command and return stdout on success.
fn run_command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("failed to execute {program}: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit code {}", output.status.code().unwrap_or(-1))
        } else {
            stderr
        };
        return Err(format!("{program} failed: {detail}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Run a command via `sudo` and return stdout on success.
fn run_sudo_command(program: &str, args: &[&str]) -> Result<String, String> {
    let mut full_args = vec![program];
    full_args.extend_from_slice(args);

    let output = Command::new("sudo")
        .args(&full_args)
        .output()
        .map_err(|e| format!("failed to execute sudo {program}: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit code {}", output.status.code().unwrap_or(-1))
        } else {
            stderr
        };
        return Err(format!("sudo {program} failed: {detail}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hardware_ports() {
        let input = "\
Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 11:22:33:44:55:66

Hardware Port: Thunderbolt 5
Device: en5
Ethernet Address: 77:88:99:aa:bb:cc
";
        let ports = parse_hardware_ports(input);
        assert_eq!(ports.get("Wi-Fi"), Some(&"en0".to_string()));
        assert_eq!(ports.get("Thunderbolt 1"), Some(&"en1".to_string()));
        assert_eq!(ports.get("Thunderbolt 5"), Some(&"en5".to_string()));
    }

    #[test]
    fn test_compute_ips_rank0() {
        let (local, peer) = compute_ips(0, "10.254.0.5").unwrap();
        assert_eq!(local, "10.254.0.5");
        assert_eq!(peer, "10.254.0.6");
    }

    #[test]
    fn test_compute_ips_rank1() {
        let (local, peer) = compute_ips(1, "10.254.0.5").unwrap();
        assert_eq!(local, "10.254.0.6");
        assert_eq!(peer, "10.254.0.5");
    }

    #[test]
    fn test_compute_ips_with_port() {
        let (local, peer) = compute_ips(0, "10.254.0.5:18520").unwrap();
        assert_eq!(local, "10.254.0.5");
        assert_eq!(peer, "10.254.0.6");
    }

    #[test]
    fn test_compute_ips_hostname_fallback() {
        // Non-IP coordinator should fall back to default
        let (local, peer) = compute_ips(0, "myhost").unwrap();
        assert_eq!(local, "10.254.0.5");
        assert_eq!(peer, "10.254.0.6");
    }

    #[test]
    fn test_auto_generate_device_map_2node() {
        let dm = auto_generate_device_map(0, 2, "rdma_en5").unwrap();
        assert_eq!(dm.world_size(), 2);
        assert_eq!(dm.device_for(0, 1), Some("rdma_en5"));
        assert_eq!(dm.device_for(1, 0), Some("rdma_en5"));
        assert_eq!(dm.device_for(0, 0), None);
    }

    #[test]
    fn test_auto_generate_device_map_rejects_non_2node() {
        assert!(auto_generate_device_map(0, 3, "rdma_en5").is_err());
        assert!(auto_generate_device_map(0, 1, "rdma_en5").is_err());
    }

    #[test]
    fn test_parse_tb_profiler_connected() {
        // Minimal system_profiler JSON with a connected device
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "aaaa-bbbb",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "5"
                    },
                    "_items": [
                        {
                            "device_name_key": "Mac Studio",
                            "domain_uuid_key": "cccc-dddd"
                        }
                    ]
                },
                {
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "eeee-ffff",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "3"
                    }
                }
            ]
        }"#;

        let mut hw_ports = HashMap::new();
        hw_ports.insert("Thunderbolt 5".to_string(), "en5".to_string());
        hw_ports.insert("Thunderbolt 3".to_string(), "en3".to_string());

        let peers = parse_tb_profiler_for_peers(json, &hw_ports).unwrap();
        // Only the first bus has _items (connected device)
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].iface, "en5");
        assert_eq!(peers[0].rdma_device, "rdma_en5");
    }

    #[test]
    fn test_parse_tb_profiler_no_connections() {
        let json = r#"{
            "SPThunderboltDataType": [
                {
                    "device_name_key": "Mac Studio",
                    "domain_uuid_key": "aaaa-bbbb",
                    "receptacle_1_tag": {
                        "receptacle_id_key": "5"
                    }
                }
            ]
        }"#;

        let mut hw_ports = HashMap::new();
        hw_ports.insert("Thunderbolt 5".to_string(), "en5".to_string());

        let peers = parse_tb_profiler_for_peers(json, &hw_ports).unwrap();
        assert!(peers.is_empty());
    }
}
