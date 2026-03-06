# rmlx-cli Production Readiness Audit

Date: 2026-03-06
Scope: `crates/rmlx-cli/` (5 files, ~470 LOC)
Auditors: Claude Opus 4.6 + Codex (gpt-5.3-codex)
References: MLX (`local MLX checkout`), torchrun, DeepSpeed, Horovod, MPI, Ray, NCCL

## Crate Overview

Binary crate producing `rmlx` CLI with two subcommands:
- `rmlx config` — SSH probe, control IP discovery, RDMA device enumeration, JSON hostfile generation
- `rmlx launch` — SSH-based distributed job spawner with rank/env injection and output multiplexing

Files: `main.rs`, `config.rs`, `launch.rs`, `hostfile.rs`, `ssh.rs`
Dependencies: clap, serde, serde_json

## Positive Findings

- No `unsafe` blocks (zero UB risk)
- No `unwrap`/`expect` in non-test code (low panic risk)
- All 18 tests pass, clippy clean
- Shell quoting implementation is thorough (`ssh.rs:5-27`)
- Env key validation prevents basic injection (`launch.rs:49-56`)
- Good test coverage for helpers (shell_quote, env parsing, slot building, hostfile roundtrip)

## P0 — Critical

### P0-1: SSH Option Injection via Unvalidated Host/User Input

**Status: ✅ FIXED (Phase 0, PR #38)** — charset validation + -- separator

- **Location:** `ssh.rs:48-64`, `ssh.rs:122-132`, `hostfile.rs:22`
- **Description:** Host and user strings from CLI args or hostfile JSON are passed directly as SSH target arguments. A value starting with `-` (e.g., `-oProxyCommand=...`) is interpreted as an SSH option, enabling local command execution if hostfile input is untrusted.
- **Fix:** Reject host/user strings beginning with `-`, validate allowed hostname/IP charset, and insert `--` before the destination argument in SSH argv.
- **Source:** Codex audit

## P1 — High

### P1-1: `backend=ring` Contract Broken Across Crates
- **Location:** `config.rs:13`, `launch.rs:105`
- **Description:** CLI accepts and exports `"ring"` as `RMLX_BACKEND`, but `rmlx-distributed/src/init.rs` only recognizes `"rdma"` and `"loopback"` — `ring` silently falls back to `Auto`. This is semantic drift between crates.
- **Fix:** Align backend enum across crates. Consider separating topology (`RMLX_TOPOLOGY=ring`) from backend (`RMLX_BACKEND=rdma`).
- **Source:** Codex audit

### P1-2: Coordinator Uses SSH Target Instead of Probed Control IP
- **Location:** `launch.rs:164`, `config.rs:80-98`
- **Description:** `config` probes and stores control IPs, but `launch` ignores them and sets `RMLX_COORDINATOR=hosts[0]` (the SSH target hostname). In multi-network setups (Thunderbolt data plane + Ethernet control), peers may fail RDMA bootstrap because RMLX_COORDINATOR points to the wrong interface.
- **Fix:** `launch` should read the hostfile and use `ips[0]` for coordinator, not the SSH target.
- **Source:** Codex audit

### P1-3: RDMA Mapping Generated but Never Consumed
- **Location:** `config.rs:229-258`, `hostfile.rs:10-11`
- **Description:** `config` writes RDMA device mesh mapping into hostfile entries, but `launch` only reads the `ssh` field. It never exports `RMLX_IBV_DEVICES` or any RDMA mapping env var. The entire RDMA mapping pipeline is orphaned from the launch path.
- **Fix:** `launch` should read hostfile entries fully and export `RMLX_IBV_DEVICES` as JSON for the RDMA runtime.
- **Source:** Codex audit + MLX comparison

### P1-4: No Signal Forwarding / Graceful Shutdown
- **Location:** `launch.rs:169-276`
- **Description:** Launcher does fail-fast on child exit but does not trap SIGINT/SIGTERM. Abrupt CLI termination (Ctrl+C) leaves remote SSH processes running as zombies. MLX uses pidfile-based remote kill; torchrun has signal propagation.
- **Fix:** Install signal handler (ctrlc crate or libc signals) that kills all spawned children and optionally sends remote kill commands.
- **Source:** Codex + MLX + frontier comparison

### P1-5: No SSH ConnectTimeout on Launch Path
- **Location:** `ssh.rs:127-132`
- **Description:** `spawn_remote()` lacks `ConnectTimeout` (unlike `run_remote()` which has it). An unreachable host during launch can stall indefinitely, blocking the entire job.
- **Fix:** Add `-o ConnectTimeout=N` to `spawn_remote()` SSH args.
- **Source:** Codex audit

### P1-6: Backend-Aware Launch Not Implemented
- **Location:** `launch.rs:125-307`
- **Description:** MLX implements distinct launch paths per backend (ring_launch, jaccl_launch, mpi_launch). rmlx-cli uses a single generic SSH launch for all backends. Ring backend requires `MLX_HOSTFILE` with ring structure; JACCL requires `MLX_IBV_DEVICES`. None of this is implemented.
- **Impact:** Multi-backend support is effectively non-functional despite being configurable.
- **Source:** MLX comparison

### P1-7: No Thunderbolt Topology Discovery
- **Location:** `config.rs`
- **Description:** MLX has 1200+ lines of Thunderbolt topology detection (system_profiler parsing, UUID mapping, DFS ring extraction). rmlx-cli's config only generates a round-robin RDMA map without actual connectivity validation.
- **Impact:** Thunderbolt ring topology (critical for Apple Silicon clusters) must be configured manually.
- **Source:** MLX comparison

## P2 — Medium

### P2-1: RDMA Device Requirement Unrealistically Strict
- **Location:** `config.rs:232-242`
- **Description:** Requires `>= world_size-1` RDMA devices per host. Most real clusters have 1-2 NICs, making this check reject valid configurations for N>3 nodes.
- **Fix:** Allow device sharing (already done in `build_rdma_map` via modulo) and remove the strict count check, or make it a warning.

### P2-2: User `--env` Can Override Scheduler-Owned Variables
- **Location:** `launch.rs:102-114`
- **Description:** Extra env vars from `--env` are inserted after `RMLX_*` scheduler vars, allowing users to override `RMLX_RANK`, `RMLX_WORLD_SIZE`, etc. This can cause rank collisions or silent misconfiguration.
- **Fix:** Insert user env first, then overwrite with scheduler-owned vars, or reject `RMLX_*` keys in user env.

### P2-3: Error Paths Silently Dropped
- **Location:** `launch.rs:186`, `launch.rs:202`, `launch.rs:257`
- **Description:** IO errors from pipe reads and process wait calls are silently ignored (`map_while(Result::ok)`, `Err(_) => {}`). This can mask diagnostics and produce false-success exits.
- **Fix:** Log errors to stderr with rank context before dropping.

### P2-4: Busy-Wait Polling Loops
- **Location:** `launch.rs:275` (50ms sleep), `ssh.rs:107` (50ms sleep)
- **Description:** Both the launch output monitor and SSH timeout use busy-wait polling with `thread::sleep(50ms)`. This wastes CPU cycles and adds up to 50ms latency to process completion detection.
- **Fix:** Use `child.wait()` on a separate thread with a timeout, or use `waitpid` with `WNOHANG` at lower frequency.

### P2-5: No Fault Tolerance / Checkpoint-Restart
- **Description:** All frontier tools (torchrun, Ray, DeepSpeed) support fault detection and restart from checkpoints. rmlx-cli has fail-fast only — any single rank failure kills the entire job with no recovery.
- **Impact:** Long-running training jobs (hours/days) on >2 nodes are impractical without this.
- **Source:** Frontier comparison

### P2-6: No Stdin Forwarding to Remote Processes
- **Location:** `launch.rs:169-226`
- **Description:** MLX supports stdin broadcast to all remote processes. rmlx-cli only captures stdout/stderr — no way to send interactive input to running jobs.
- **Source:** MLX comparison

### P2-7: Hostfile `envs` Field Not Supported
- **Description:** MLX hostfiles support a per-hostfile `envs` array for persistent environment configuration. rmlx-cli only supports runtime `--env` flags.
- **Source:** MLX comparison

## P3 — Low

### P3-1: `rdma_device_todo` Placeholder Emitted
- **Location:** `config.rs:245-246`
- **Description:** When `--no-verify-rdma` is used, a literal `"rdma_device_todo"` string is written to the hostfile. This will silently pass through to runtime and cause confusing errors.
- **Fix:** Use a clearly invalid sentinel or omit the field entirely.

### P3-2: No Structured Logging
- **Description:** Only `println!/eprintln!` with an optional `--verbose` flag. No log levels, no structured output, no per-rank machine-readable events. Production debugging requires grep through mixed output.
- **Fix:** Consider `tracing` or `env_logger` crate integration.

### P3-3: No Timeline Profiling
- **Description:** Horovod provides `--timeline-filename` for chrome://tracing visualization of collective operations. No equivalent exists in rmlx-cli. Critical for MoE expert dispatch performance debugging.
- **Source:** Frontier comparison

### P3-4: No Process Affinity/Binding
- **Description:** MPI and DeepSpeed support `--bind-to core/socket` for NUMA pinning. Not implemented. Less critical on Apple Silicon (unified memory) but affects P-core vs E-core scheduling.
- **Source:** Frontier comparison

### P3-5: No MPI/SLURM Backend Support
- **Description:** rmlx-cli only supports SSH launcher. No MPI integration (mpirun detection, hostfile conversion) or SLURM environment detection.
- **Source:** Frontier comparison

### P3-6: Launch `--verbose` Flag Missing
- **Location:** `launch.rs:11-39`
- **Description:** `config` has `--verbose` but `launch` does not. No way to see what commands are being spawned on which hosts.

## Feature Completeness vs MLX Reference

| Feature | MLX | rmlx-cli | Gap |
|---------|-----|----------|-----|
| CLI entry point | ✓ | ✓ | — |
| SSH launch | ✓ | ✓ | — |
| Multi-backend launch | ✓ (ring/mpi/nccl/jaccl) | ✗ (single path) | P1 |
| Hostfile generation | ✓ | ✓ | — |
| Thunderbolt topology | ✓ (1200 LOC) | ✗ | P1 |
| RDMA mesh mapping | ✓ | ✓ (orphaned) | P1 |
| Control IP resolution | ✓ | ✓ | — |
| Ring extraction (DFS) | ✓ | ✗ | P1 |
| Pidfile remote kill | ✓ | ✗ | P1 |
| Stdin broadcast | ✓ | ✗ | P2 |
| Structured logging | ✓ | ✗ | P3 |
| Env var export | ✓ (8+ vars) | ✓ (5 vars) | P1 |
| Script path resolution | ✓ | ✗ | P3 |

## Feature Completeness vs Frontier CUDA Tools

| Feature | Best-in-class | rmlx-cli | Gap |
|---------|--------------|----------|-----|
| Fault tolerance | torchrun/Ray | ✗ | P2 |
| Elastic scaling | torchrun/Ray | ✗ | P3 |
| Process affinity | MPI/DeepSpeed | ✗ | P3 |
| Timeline profiling | Horovod | ✗ | P3 |
| Bandwidth testing | NCCL | ✗ | P3 |
| Multi-launcher | DeepSpeed/Horovod | ✗ | P3 |
| Resource monitoring | Ray | ✗ | P3 |

## Overall Assessment

**rmlx-cli is ~40-50% feature-complete** vs MLX reference and ~25-30% vs frontier CUDA tools.

Suitable for: development/testing on small (2-4 node) Apple Silicon clusters with manual oversight.
Not yet suitable for: production multi-node training, long-running jobs, or unsupervised operation.

**Critical path to production:**
1. Fix P0 SSH injection vulnerability
2. Resolve P1 cross-crate contract mismatches (backend enum, coordinator IP, RDMA export)
3. Add signal forwarding for graceful shutdown
4. Implement backend-aware launch paths matching MLX
5. Add Thunderbolt topology discovery
