# rmlx-cli -- Native CLI Tooling

## Overview

`rmlx-cli` is the command-line interface crate for RMLX, providing native tooling for distributed launch, configuration, topology discovery, and backend selection.

> **Status:** `rmlx launch` with signal forwarding (Phase 3), `rmlx config` for host/backend configuration. **Phase 5 additions:** TB5/TB4 topology discovery via `system_profiler`, `Interconnect` enum, `detect_topology()`, `resolve_auto_backend()`, and `--backend auto` default with topology-aware backend selection.

---

## Module Structure

```
rmlx-cli/src/
├── main.rs          # CLI entry point and argument parsing
├── launch.rs        # `rmlx launch` subcommand + signal forwarding
├── config.rs        # `rmlx config` subcommand
└── topology.rs      # TB5/TB4 topology discovery + auto backend selection
```

---

## launch.rs -- Distributed Launch

Launches distributed inference across multiple nodes with process management and signal forwarding.

**Key features:**
- **Process spawning**: launches worker processes on configured hosts via SSH
- **Signal forwarding**: ctrlc handler forwards SIGINT/SIGTERM to child processes for graceful shutdown (Phase 3)
- **`--backend auto`**: default backend selection that uses topology discovery to choose the optimal transport (Phase 5)
- **Hostfile support**: reads host configuration from JSON hostfiles

### Backend Selection (Phase 5)

The `--backend` flag now defaults to `auto`. When set to `auto`, the CLI uses `resolve_auto_backend()` from `topology.rs` to select the best available transport based on detected hardware:

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | RDMA | RDMA hardware detected (ibverbs available) |
| 2 | TB5 | Thunderbolt 5 interconnect detected |
| 3 | TB4 | Thunderbolt 4 interconnect detected |
| 4 | TCP | Fallback when no high-speed interconnect is found |

---

## topology.rs -- Topology Discovery (Phase 5)

Discovers the local machine's interconnect topology by querying `system_profiler` for Thunderbolt controller information.

### Interconnect Enum

```rust
pub enum Interconnect {
    Thunderbolt5,
    Thunderbolt4,
    Tcp,
}
```

### Key Functions

| Function | Description |
|----------|-------------|
| `detect_topology()` | Queries `system_profiler SPThunderboltDataType` and parses the output to determine the highest available Thunderbolt version |
| `resolve_auto_backend()` | Combines RDMA availability check with topology detection to select the optimal backend |

**Detection logic:**
1. Run `system_profiler SPThunderboltDataType -json` to enumerate Thunderbolt controllers
2. Parse the JSON output for controller speed/version information
3. Return `Thunderbolt5` if any TB5 controller is found, `Thunderbolt4` if TB4, or `Tcp` as fallback

---

## config.rs -- Host Configuration

Generates host configuration files for distributed runs.

```bash
rmlx config --hosts node1,node2 --backend rdma --over thunderbolt --output rmlx-hosts.json --verbose
```

| Flag | Description |
|------|-------------|
| `--hosts` | Comma-separated list of hostnames |
| `--backend` | Transport backend (auto, rdma, tcp) |
| `--over` | Physical interconnect hint (thunderbolt, ethernet) |
| `--output` | Output JSON hostfile path |
| `--verbose` | Enable verbose logging |

---

## Dependencies

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ctrlc = "3"
```
