# System Requirements

This page describes the hardware and software requirements for building and running rmlx.

---

## Hardware

### Required

| Item | Requirement | Notes |
|------|------------|-------|
| **Processor** | Apple Silicon (M1 or later) | Metal GPU + Unified Memory Architecture (UMA) required |
| **Memory** | 16GB or more recommended | 32GB+ recommended for large model inference |
| **macOS** | macOS 14.0+ (Sonoma) | Utilizes Metal 3 API features |

> **Note**: Metal compute capabilities are limited on Intel Macs, so an Apple Silicon Mac is required.
> Thanks to the UMA architecture, the GPU and CPU share the same physical memory, enabling zero-copy optimizations.

### Distributed Inference (Optional)

The following additional hardware is required for distributed inference (Expert Parallelism):

| Item | Requirement |
|------|------------|
| **Thunderbolt 5 port** | At least 1 per Mac |
| **TB5-compatible cable** | Thunderbolt 5 certified cable |
| **Node count** | 2 or more Apple Silicon Macs |

> **Note**: RDMA communication over Thunderbolt 5 achieves bandwidth of 6.89 GB/s or higher between nodes.
> Using dual TB5 ports can scale to 12 GB/s or higher (Phase 6).

### RDMA Setup (Cluster Validation)

RMLX includes distributed setup/launch helpers modeled after
`mlx.distributed_config` and `mlx.launch`. With `--auto-setup`, you only need to
plug in the TB5 cable and run one command — TB topology detection, IP assignment,
and RDMA device mapping are all handled automatically.

```bash
# 1) Auto-detect TB5 topology + assign IPs + generate hostfile (one-time)
rmlx config \
  --hosts node1,node2 \
  --auto-setup \
  --output rmlx-hosts.json \
  --verbose
```

This performs the following steps automatically:
1. SSH to each host and query `system_profiler SPThunderboltDataType -json`
2. Parse TB port UUIDs and match connections across hosts
3. Map TB port tags to network interfaces via `networksetup -listallhardwareports`
4. Assign /30 point-to-point IPs (192.168.x.x) to each TB link
5. Run `sudo ifconfig` and `sudo route` on each host
6. Generate hostfile with data-plane IPs and RDMA device mappings (`rdma_{iface}`)

```bash
# 2) Validate RDMA visibility on each host
rmlx launch \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- ibv_devices

# 3) Run RDMA crate tests on both hosts
rmlx launch \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- cargo test -p rmlx-rdma -- --nocapture
```

> **Note**: `--auto-setup` requires passwordless `sudo` on each host (for `ifconfig`/`route`).
> If you only want hostfile generation without IP configuration, omit `--auto-setup`.

---

## Software

### Required

| Software | Version | Installation |
|----------|---------|-------------|
| **macOS** | 14.0+ (Sonoma) | System update |
| **Rust toolchain** | stable (1.80+) | [rustup](https://rustup.rs/) |

#### Rust Installation

```bash
# Install rustup (first-time installation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# If already installed, update to the latest stable
rustup update stable
```

> **Note**: The `rust-toolchain.toml` file in the project root automatically manages the exact Rust version.
> If `rustup` is installed, running `cargo build` will automatically install the required toolchain.

### Xcode (Optional)

| Configuration | AOT Compilation | JIT Compilation | Notes |
|--------------|----------------|----------------|-------|
| **Full Xcode installation** | O | O | Supports `.metal` -> `.metallib` AOT compilation |
| **Command Line Tools only** | X | O | AOT unavailable; operates via JIT (`new_library_with_source`) |
| **Not installed** | X | O | Operates via JIT; tests can still run |

- **AOT compilation**: `build.rs` invokes the `xcrun metal` compiler during build to pre-compile `.metal` shaders into `.metallib`. **Full Xcode installation is required** (Command Line Tools alone cannot use the `xcrun -sdk macosx metal` command).
- **JIT compilation**: Even without Xcode, shaders can be compiled at runtime via `MTLDevice::newLibraryWithSource`. Test code uses JIT compilation, so it runs without Xcode.

```bash
# Install Xcode (App Store or xcode-select)
xcode-select --install  # Installs Command Line Tools only
# For full Xcode, install from the App Store
```

---

## Development Tools (Recommended)

Not required, but these tools improve development productivity:

| Tool | Purpose | Installation |
|------|---------|-------------|
| **cargo-watch** | Auto-build/test on file changes | `cargo install cargo-watch` |
| **rust-analyzer** | IDE language server (auto-completion, type inference) | VS Code extension or editor-specific installation |

```bash
# Install cargo-watch
cargo install cargo-watch

# Usage example: auto-test on file changes
cargo watch -x test

# Usage example: watch a specific crate
cargo watch -x 'test -p rmlx-metal'
```

---

## Environment Verification

Run the following commands to verify that your installation is correct:

```bash
# Check Rust toolchain
rustc --version
cargo --version

# Check macOS version
sw_vers

# Check Metal support (verify Apple Silicon)
system_profiler SPDisplaysDataType | grep "Metal"

# Check Xcode Metal compiler (optional)
xcrun -sdk macosx --find metal 2>/dev/null && echo "AOT compilation available" || echo "JIT-only mode"
```

Expected output:

```
rustc 1.80.0 (051478957 2024-07-21)
cargo 1.80.0 (376290515 2024-07-16)
ProductName:    macOS
ProductVersion: 14.x
Metal:          Apple M1 Pro
AOT compilation available
```

---

Next step: [Building and Running](building.md)
