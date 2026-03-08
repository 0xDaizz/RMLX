# rmlx-cli -- Native CLI Tooling

## Overview

`rmlx-cli` is the command-line interface crate for RMLX, providing native tooling for distributed launch, configuration, topology discovery, and backend selection.

> **Status:** `rmlx launch` with signal forwarding, `rmlx config` with TB5 auto-discovery (system_profiler + IP assignment + RDMA device mapping), `--backend auto` default with topology-aware backend selection.

---

## Module Structure

```
rmlx-cli/src/
├── main.rs          # CLI entry point and argument parsing
├── launch.rs        # `rmlx launch` subcommand + signal forwarding
├── config.rs        # `rmlx config` subcommand (TB discovery + legacy paths)
├── tb_discovery.rs  # TB5 topology detection, UUID matching, IP assignment
├── topology.rs      # Local TB speed classification + auto backend selection
├── hostfile.rs      # Hostfile JSON parsing and serialization
└── ssh.rs           # SSH remote execution utilities
```

---

## launch.rs -- Distributed Launch

Launches distributed inference across multiple nodes with process management and signal forwarding.

**Key features:**
- **Process spawning**: launches worker processes on configured hosts via SSH
- **Signal forwarding**: ctrlc handler forwards SIGINT/SIGTERM to child processes for graceful shutdown
- **`--backend auto`**: default backend selection that uses topology discovery to choose the optimal transport
- **Hostfile support**: reads host configuration from JSON hostfiles

### Backend Selection

The `--backend` flag now defaults to `auto`. When set to `auto`, the CLI uses `resolve_auto_backend()` from `topology.rs` to select the best available transport based on detected hardware:

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | RDMA | RDMA hardware detected (ibverbs available) |
| 2 | TB5 | Thunderbolt 5 interconnect detected |
| 3 | TB4 | Thunderbolt 4 interconnect detected |
| 4 | TCP | Fallback when no high-speed interconnect is found |

---

## config.rs -- Host Configuration

Generates host configuration files for distributed runs. Has two operating modes:

### TB Discovery Path (`--auto-setup --over thunderbolt`)

When both flags are set, config uses `tb_discovery.rs` for fully automatic setup:

```bash
rmlx config --hosts node1,node2 --auto-setup --output rmlx-hosts.json --verbose
```

**Flow:**
1. SSH verify all hosts
2. `tb_discovery::discover()` — SSH to each host, query `system_profiler` + `networksetup`, match TB port UUIDs across hosts
3. `topology.assign_ips()` — assign /30 point-to-point IPs (192.168.x.x)
4. `topology.apply_setup()` — run `sudo ifconfig`/`sudo route` on each host via SSH
5. Build hostfile using `topology.data_ips()` and `topology.rdma_map()`
6. Write hostfile JSON

### Legacy Path (no TB auto-setup)

Without `--auto-setup` or with `--over ethernet`, config uses the traditional flow:

```bash
rmlx config --hosts node1,node2 --backend rdma --over ethernet --control-iface en0 --output rmlx-hosts.json
```

**Flow:**
1. SSH verify + optional bridge0 disable
2. Probe control IP via `ipconfig getifaddr`
3. Probe RDMA devices via `ibv_devices`
4. Build hostfile with round-robin RDMA device mapping

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--hosts` | (required) | Comma-separated list of hostnames/IPs |
| `--backend` | `rdma` | Transport backend (`rdma`, `loopback`) |
| `--over` | `thunderbolt` | Physical interconnect hint (`thunderbolt`, `ethernet`) |
| `--auto-setup` | `false` | Auto-detect TB topology, assign IPs, configure interfaces |
| `--control-iface` | `en0` | Interface for control IP probe (legacy path only) |
| `--no-verify-rdma` | `false` | Skip RDMA device validation |
| `--output` | `rmlx-hosts.json` | Output hostfile path |
| `--ssh-user` | current user | SSH user override |
| `--timeout` | `20` | Per-host SSH command timeout (seconds) |
| `--verbose` | `false` | Enable verbose progress logs |

---

## tb_discovery.rs -- TB5 Auto-Discovery

Implements MLX-style automatic Thunderbolt 5 topology detection and IP configuration. All parsing functions are pure (no SSH) for testability.

### Data Structures

```rust
pub struct TbPort {
    pub iface: String,                  // e.g., "en5"
    pub uuid: String,                   // domain_uuid_key from system_profiler
    pub connected_to: Option<String>,   // peer's domain_uuid_key
}

pub struct TbHost {
    pub name: String,                   // device_name_key or hostname
    pub ports: Vec<TbPort>,
}

pub struct TbLink {
    pub src_host: usize,               // host index
    pub src_iface: String,
    pub src_ip: String,                 // assigned after assign_ips()
    pub dst_host: usize,
    pub dst_iface: String,
    pub dst_ip: String,
}

pub struct TbTopology {
    pub hosts: Vec<TbHost>,
    pub links: Vec<TbLink>,
}
```

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_hardware_ports(output)` | Parse `networksetup -listallhardwareports` → `HashMap<"Thunderbolt 5", "en5">` |
| `parse_tb_profiler(json, hw_ports)` | Parse `system_profiler` JSON → `TbHost` with resolved interface names |
| `discover(hosts, user, timeout, verbose)` | SSH to all hosts, collect + parse topology, match UUIDs → `TbTopology` |

### TbTopology Methods

| Method | Description |
|--------|-------------|
| `assign_ips()` | Assign /30 point-to-point IPs (192.168.X.Y scheme, 4 addresses per link) |
| `setup_commands(host_idx)` | Generate `sudo ifconfig`/`sudo route` commands for a host |
| `apply_setup(hosts, user, timeout, verbose)` | Execute setup on all hosts via SSH |
| `data_ips(host_idx)` | Get assigned data-plane IPs for hostfile `ips` field |
| `rdma_map(host_idx, num_hosts)` | Get RDMA device map (`rdma_{iface}`) for hostfile `rdma` field |

### Detection Method

1. SSH to each host: `system_profiler SPThunderboltDataType -json`
   - Extract `domain_uuid_key` (port UUID), `receptacle_1_tag.receptacle_id_key` (port tag), `_items[0].domain_uuid_key` (connected peer UUID)
2. SSH to each host: `networksetup -listallhardwareports`
   - Map `"Thunderbolt {tag}"` → interface name (e.g., `en5`)
3. Build UUID reverse index across all hosts
4. Match `connected_to` UUIDs to discover point-to-point links
5. Deduplicate bidirectional matches (A→B and B→A = 1 link)

### IP Assignment Scheme (MLX-compatible)

```
Link 0: 192.168.0.1 ↔ 192.168.0.2   (/30, netmask 255.255.255.252)
Link 1: 192.168.0.5 ↔ 192.168.0.6
Link 2: 192.168.0.9 ↔ 192.168.0.10
...
Link 64: 192.168.1.1 ↔ 192.168.1.2  (wraps to next /24)
```

---

## topology.rs -- Local Topology Detection

Discovers the local machine's interconnect topology by querying `system_profiler` for Thunderbolt controller information. Used by `launch.rs` for `--backend auto` resolution.

### Interconnect Enum

```rust
pub enum Interconnect {
    Loopback,
    Unknown,
    Ethernet,
    Thunderbolt4,
    Thunderbolt5,
    Rdma,
}
```

### Key Functions

| Function | Description |
|----------|-------------|
| `probe_thunderbolt()` | Query local `system_profiler` for TB speed classification |
| `detect_topology(hosts)` | Build NxN interconnect matrix (best-effort heuristic) |
| `resolve_auto_backend(has_rdma, local_tb)` | Select optimal backend string for `--backend auto` |
| `validate_backend(backend)` | Validate backend string is a known value |

---

## Hostfile Format

```json
[
  {
    "ssh": "hwstudio1",
    "ips": ["192.168.0.1"],
    "rdma": [null, "rdma_en5"]
  },
  {
    "ssh": "hwstudio2",
    "ips": ["192.168.0.2"],
    "rdma": ["rdma_en5", null]
  }
]
```

| Field | Description |
|-------|-------------|
| `ssh` | SSH hostname or alias |
| `ips` | Data-plane IP addresses (TB interface IPs when using auto-setup) |
| `rdma` | Per-peer RDMA device names; `rdma[i]` = device for communicating with rank `i`, `null` for self |

---

## Dependencies

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
```
