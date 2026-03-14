//! MLX-style distributed auto-initialization.
//!
//! Provides `init()` for automatic RDMA bootstrapping from environment variables,
//! with graceful fallback to loopback (single-process) mode.
//!
//! ## Initialization protocol
//!
//! Multi-rank RDMA init follows a **coordinator-mediated all_gather** pattern
//! inspired by Apple's MLX framework:
//!
//! 1. Each rank creates one QP per peer (plus a self-rank placeholder).
//! 2. All ranks pack their per-peer QP infos and exchange them via a single
//!    `all_gather_bytes()` round (rank 0 as hub on a single TCP port).
//! 3. Each rank connects `qp[peer]` to `gathered[peer][my_rank]` — the QP
//!    that peer created specifically for talking to us. This ensures matching
//!    (qpn, psn, gid) on both sides of every connection.
//! 4. The self-rank slot stays unconnected (never used for I/O).

use std::path::Path;
use std::sync::Arc;

use rmlx_rdma::connection::{RdmaConfig, RdmaConnection};
use rmlx_rdma::context::RdmaContext;
use rmlx_rdma::coordinator::CoordinatorConfig;
use rmlx_rdma::device_file::DeviceMap;
use rmlx_rdma::multi_port::Topology;
use rmlx_rdma::qp::{CompletionQueue, QpInfo, QueuePair};

use crate::group::{DistributedError, Group};
use crate::transport::RdmaConnectionTransport;
use crate::warmup::{WarmupConfig, WarmupResult, WarmupState};

// ── Public types ──

/// Hint for which backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHint {
    /// Automatically detect: try RDMA first, fall back to loopback.
    Auto,
    /// Force RDMA backend (fail if unavailable).
    Rdma,
    /// Force loopback (single-process) mode.
    Loopback,
}

/// Configuration for distributed initialization.
#[derive(Debug, Clone)]
pub struct InitConfig {
    /// If true, fail on any error instead of falling back to loopback.
    pub strict: bool,
    /// Backend hint.
    pub backend: BackendHint,
    /// Override rank (otherwise read from env).
    pub rank: Option<u32>,
    /// Override world_size (otherwise read from env).
    pub world_size: Option<u32>,
    /// Coordinator address (otherwise read from RMLX_COORDINATOR env).
    pub coordinator_addr: Option<String>,
    /// Coordinator port (otherwise use default 18520).
    pub coordinator_port: Option<u16>,
    /// Path to device file (otherwise read from RMLX_IBV_DEVICES env).
    pub device_file: Option<String>,
    /// Topology hint.
    pub topology: Option<String>,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            strict: false,
            backend: BackendHint::Auto,
            rank: None,
            world_size: None,
            coordinator_addr: None,
            coordinator_port: None,
            device_file: None,
            topology: None,
        }
    }
}

/// Result of successful distributed initialization.
pub struct DistributedContext {
    /// The communication group.
    pub group: Group,
    /// This process's rank.
    pub rank: u32,
    /// Total number of processes.
    pub world_size: u32,
    /// Which backend was selected.
    pub backend: BackendHint,
    /// Warmup result from RDMA/JIT warmup phase (None for loopback mode).
    pub warmup: Option<WarmupResult>,
}

// ── Helper functions ──

/// Parse an environment variable as u32.
fn parse_env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Try MPI/SLURM compat env vars for rank.
fn from_env_compat_rank() -> Option<u32> {
    parse_env_u32("OMPI_COMM_WORLD_RANK")
        .or_else(|| parse_env_u32("PMI_RANK"))
        .or_else(|| parse_env_u32("SLURM_PROCID"))
}

/// Try MPI/SLURM compat env vars for world_size.
fn from_env_compat_world_size() -> Option<u32> {
    parse_env_u32("OMPI_COMM_WORLD_SIZE")
        .or_else(|| parse_env_u32("PMI_SIZE"))
        .or_else(|| parse_env_u32("SLURM_NTASKS"))
}

/// Resolve rank: config > RMLX_RANK > compat env > 0.
pub(crate) fn resolve_rank(config: &InitConfig) -> u32 {
    config
        .rank
        .or_else(|| parse_env_u32("RMLX_RANK"))
        .or_else(from_env_compat_rank)
        .unwrap_or(0)
}

/// Resolve world_size: config > RMLX_WORLD_SIZE > compat env > 1.
pub(crate) fn resolve_world_size(config: &InitConfig) -> u32 {
    config
        .world_size
        .or_else(|| parse_env_u32("RMLX_WORLD_SIZE"))
        .or_else(from_env_compat_world_size)
        .unwrap_or(1)
}

/// Resolve backend hint: config > RMLX_BACKEND env > Auto.
///
/// Returns an error if the RMLX_BACKEND env var is set to an unrecognized value.
fn resolve_backend(config: &InitConfig) -> Result<BackendHint, DistributedError> {
    if config.backend != BackendHint::Auto {
        return Ok(config.backend);
    }
    match std::env::var("RMLX_BACKEND").ok().as_deref() {
        Some("rdma") | Some("RDMA") => Ok(BackendHint::Rdma),
        Some("loopback") | Some("LOOPBACK") => Ok(BackendHint::Loopback),
        Some("tb5") | Some("TB5") | Some("tb4") | Some("TB4") | Some("tcp") | Some("TCP") => Ok(BackendHint::Auto),
        Some("auto") | Some("AUTO") | None => Ok(BackendHint::Auto),
        Some(other) => Err(DistributedError::Config(format!(
            "unknown RMLX_BACKEND value {other:?}; expected \"rdma\", \"loopback\", \"tb5\", \"tb4\", \"tcp\", or \"auto\""
        ))),
    }
}

/// Resolve topology from config, env, or device file structure.
///
/// Priority:
/// 1. Explicit config/env topology hint.
/// 2. Device file structure (full matrix -> Mesh, ring neighbors only -> Ring).
/// 3. Default: Ring.
fn resolve_topology(config: &InitConfig, device_map: Option<&DeviceMap>) -> Topology {
    // Check explicit hint first (config > env).
    let topo_str: Option<String> = config
        .topology
        .clone()
        .or_else(|| std::env::var("RMLX_TOPOLOGY").ok());

    if let Some(ref t) = topo_str {
        return match t.to_lowercase().as_str() {
            "mesh" => Topology::Mesh,
            "hybrid" => Topology::Hybrid { group_size: 4 },
            _ => Topology::Ring,
        };
    }

    // Infer from device file: if every off-diagonal entry is Some, it's a mesh.
    if let Some(dm) = device_map {
        let ws = dm.world_size();
        let all_connected =
            (0..ws).all(|i| (0..ws).all(|j| i == j || dm.device_for(i, j).is_some()));
        if all_connected {
            return Topology::Mesh;
        }
    }

    Topology::Ring
}

/// Resolve device file path: config > RMLX_IBV_DEVICES env.
fn resolve_device_file(config: &InitConfig) -> Option<String> {
    config
        .device_file
        .clone()
        .or_else(|| std::env::var("RMLX_IBV_DEVICES").ok())
}

/// Load a DeviceMap from the resolved device file path, if any.
fn load_device_map(config: &InitConfig) -> Option<DeviceMap> {
    let path_str = resolve_device_file(config)?;
    match DeviceMap::from_file(Path::new(&path_str)) {
        Ok(dm) => Some(dm),
        Err(e) => {
            tracing::warn!(
                target: "rmlx_distributed",
                path = %path_str,
                %e,
                "failed to load device file",
            );
            None
        }
    }
}

/// Build a loopback (single-process) context.
fn loopback_context() -> Result<DistributedContext, DistributedError> {
    let group = Group::world(1, 0)?;
    Ok(DistributedContext {
        group,
        rank: 0,
        world_size: 1,
        backend: BackendHint::Loopback,
        warmup: None,
    })
}

// ── RDMA initialization (coordinator-mediated all_gather pattern) ──

/// Attempt to initialize RDMA connections and build a transport-backed group.
///
/// Uses a coordinator-mediated all_gather pattern with per-peer QP creation:
/// 1. Each rank creates one QP per peer (plus a self-rank placeholder)
/// 2. All ranks pack their per-peer QP infos and exchange via `all_gather_bytes()`
/// 3. Each rank connects QP[peer] to the remote QP that peer created for us
/// 4. Self-rank slot stays unconnected (placeholder for transport indexing)
fn try_rdma_init(
    rank: u32,
    world_size: u32,
    config: &InitConfig,
) -> Result<DistributedContext, DistributedError> {
    // Check hardware availability
    if !rmlx_rdma::is_available() {
        return Err(DistributedError::BackendUnavailable(
            "RDMA hardware not available on this system".to_string(),
        ));
    }

    // Resolve coordinator address (required for multi-rank)
    let coordinator_addr = config
        .coordinator_addr
        .clone()
        .or_else(|| std::env::var("RMLX_COORDINATOR").ok())
        .ok_or_else(|| {
            DistributedError::Config(
                "coordinator address required for RDMA init; set RMLX_COORDINATOR or \
                 provide coordinator_addr in InitConfig"
                    .to_string(),
            )
        })?;

    // Single coordinator port for all_gather (no per-pair port arithmetic)
    let coordinator_port = config
        .coordinator_port
        .or_else(|| parse_env_u32("RMLX_COORDINATOR_PORT").and_then(|p| u16::try_from(p).ok()))
        .unwrap_or(rmlx_rdma::coordinator::COORDINATOR_PORT);

    // Load device file for per-peer device selection
    let device_map = load_device_map(config);
    if let Some(ref dm) = device_map {
        if dm.world_size() != world_size as usize {
            return Err(DistributedError::Config(format!(
                "device file world_size ({}) does not match configured world_size ({world_size})",
                dm.world_size()
            )));
        }
        tracing::info!(
            target: "rmlx_distributed",
            world_size = dm.world_size(),
            "loaded device file",
        );
    }

    // Resolve topology (may be inferred from device file)
    let topology = resolve_topology(config, device_map.as_ref());
    tracing::info!(target: "rmlx_distributed", ?topology, "resolved topology");

    // --- Phase 1: Create per-peer QPs and gather all QP info ---
    //
    // Correct protocol for UC (Unreliable Connected) with N peers:
    //   1. Each rank creates one QP per peer (N-1 QPs + 1 self placeholder).
    //   2. Each rank packs its per-peer QP infos into a byte buffer indexed by
    //      destination rank: local_infos[dst] = qp_for_dst.local_info().
    //   3. A single all_gather_bytes round exchanges the packed buffers.
    //   4. Each rank connects qp_for_peer to gathered[peer][my_rank]
    //      (the QP that peer created for talking to us).
    //
    // This ensures QP numbers match: rank A's QP for B is connected to
    // rank B's QP for A, with matching (qpn, psn, gid) on both sides.

    let coord_cfg = CoordinatorConfig {
        port: coordinator_port,
        accept_timeout_secs: 120,
        connect_timeout_ms: 5000,
        connect_retries: 30,
        retry_delay_ms: 500,
        io_timeout_secs: 60,
    };

    let peers = topology.peers(rank, world_size);

    // Helper: open RDMA context using device file name if available, else default.
    let open_ctx = |peer: u32| -> Result<RdmaContext, DistributedError> {
        let device_name = device_map
            .as_ref()
            .and_then(|dm| dm.device_for(rank as usize, peer as usize));
        let ctx = match device_name {
            Some(name) => RdmaContext::open_by_name(name),
            None => RdmaContext::open_default(),
        };
        ctx.map_err(|e| DistributedError::Transport(e.to_string()))
    };

    // Phase 1a: Create per-peer QPs and collect their local QP infos.
    // qps[peer] holds (ctx, pd, cq, qp) for each peer slot.
    // For self-rank, we create a dummy QP (no connect needed).
    struct QpSlot {
        ctx: RdmaContext,
        pd: rmlx_rdma::context::ProtectionDomain,
        cq: CompletionQueue,
        qp: QueuePair,
    }

    let mut slots: Vec<Option<QpSlot>> = (0..world_size).map(|_| None).collect();
    let mut local_infos: Vec<QpInfo> = vec![
        QpInfo {
            lid: 0,
            qpn: 0,
            psn: 0,
            gid: [0; 16],
        };
        world_size as usize
    ];

    for peer in 0..world_size {
        let ctx = open_ctx(peer)?;
        let pd = ctx
            .alloc_pd()
            .map_err(|e| DistributedError::Transport(e.to_string()))?;
        let cq =
            CompletionQueue::new(&ctx).map_err(|e| DistributedError::Transport(e.to_string()))?;
        let mut qp = QueuePair::create_uc(&pd, &cq, &ctx)
            .map_err(|e| DistributedError::Transport(e.to_string()))?;

        if peer != rank {
            // Query local info so we have (lid, qpn, psn, gid) to share.
            qp.query_local_info(&ctx, rank)
                .map_err(|e| DistributedError::Transport(e.to_string()))?;
            local_infos[peer as usize] = qp.local_info().clone();
        }
        // Self-slot: local_infos[rank] stays zeroed (never used by peers).

        slots[peer as usize] = Some(QpSlot { ctx, pd, cq, qp });
    }

    // Phase 1b: Pack local QP infos and all_gather via coordinator.
    // Wire format: world_size * 26 bytes (lid:2 + qpn:4 + psn:4 + gid:16).
    const QP_INFO_WIRE_SIZE: usize = 2 + 4 + 4 + 16;
    let mut local_buf = vec![0u8; world_size as usize * QP_INFO_WIRE_SIZE];
    for (i, info) in local_infos.iter().enumerate() {
        let off = i * QP_INFO_WIRE_SIZE;
        local_buf[off..off + 2].copy_from_slice(&info.lid.to_le_bytes());
        local_buf[off + 2..off + 6].copy_from_slice(&info.qpn.to_le_bytes());
        local_buf[off + 6..off + 10].copy_from_slice(&info.psn.to_le_bytes());
        local_buf[off + 10..off + 26].copy_from_slice(&info.gid);
    }

    let hub_host = if rank == 0 { "" } else { &coordinator_addr };
    let gathered_bytes = rmlx_rdma::coordinator::all_gather_bytes(
        rank, world_size, &local_buf, hub_host, &coord_cfg,
    )
    .map_err(|e| DistributedError::Transport(format!("all_gather_bytes failed: {e}")))?;

    if gathered_bytes.len() != world_size as usize {
        return Err(DistributedError::Transport(format!(
            "all_gather returned {} entries, expected {world_size}",
            gathered_bytes.len()
        )));
    }

    // Phase 1c: Deserialize remote QP infos.
    // remote_for_me[peer] = the QpInfo that `peer` created for talking to `rank`.
    let mut remote_for_me: Vec<QpInfo> = Vec::with_capacity(world_size as usize);
    for peer in 0..world_size {
        let peer_buf = &gathered_bytes[peer as usize];
        // peer's buffer has world_size entries; entry at index `rank` is for us.
        let off = rank as usize * QP_INFO_WIRE_SIZE;
        if peer_buf.len() < off + QP_INFO_WIRE_SIZE {
            return Err(DistributedError::Transport(format!(
                "gathered buffer from rank {peer} too short: {} < {}",
                peer_buf.len(),
                off + QP_INFO_WIRE_SIZE
            )));
        }
        let lid = u16::from_le_bytes([peer_buf[off], peer_buf[off + 1]]);
        let qpn = u32::from_le_bytes([
            peer_buf[off + 2],
            peer_buf[off + 3],
            peer_buf[off + 4],
            peer_buf[off + 5],
        ]);
        let psn = u32::from_le_bytes([
            peer_buf[off + 6],
            peer_buf[off + 7],
            peer_buf[off + 8],
            peer_buf[off + 9],
        ]);
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&peer_buf[off + 10..off + 26]);
        remote_for_me.push(QpInfo { lid, qpn, psn, gid });
    }

    // --- Phase 2: Connect per-peer QPs using gathered info ---
    //
    // qp_for_peer[peer].connect(remote_for_me[peer]) ensures matching
    // QP numbers on both sides of the connection.

    let mut connections: Vec<Option<RdmaConnection>> = (0..world_size).map(|_| None).collect();

    for peer in 0..world_size {
        let slot = slots[peer as usize]
            .take()
            .expect("QP slot should be populated");

        if peer != rank {
            // Connect this QP to the peer's QP that was created for us.
            slot.qp
                .connect(&remote_for_me[peer as usize])
                .map_err(|e| {
                    DistributedError::Transport(format!("QP connect to rank {peer} failed: {e}"))
                })?;
        }
        // Self-slot: leave unconnected (placeholder for transport indexing).

        let peer_config = RdmaConfig {
            rank,
            world_size,
            peer_host: coordinator_addr.clone(),
            exchange_port: coordinator_port,
            sync_port: coordinator_port,
            cq_timeout_ms: 5000,
            accept_timeout_secs: 60,
            io_max_retries: 3,
            io_retry_delay_ms: 1000,
            connect_timeout_ms: 5000,
        };

        connections[peer as usize] = Some(RdmaConnection::from_parts(
            slot.ctx,
            slot.pd,
            slot.cq,
            slot.qp,
            peer_config,
        ));
    }

    // Unwrap all Options — every slot should be populated now.
    let connections: Vec<RdmaConnection> = connections
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.ok_or_else(|| DistributedError::Transport(format!("missing connection for rank {i}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let transport = RdmaConnectionTransport::new(connections, rank);
    let all_ranks: Vec<u32> = (0..world_size).collect();
    let group = Group::with_transport(all_ranks, rank, world_size, Arc::new(transport))?;

    tracing::info!(
        target: "rmlx_distributed",
        rank,
        world_size,
        num_peers = peers.len(),
        ?topology,
        ?peers,
        "connected to peers",
    );

    // ── Warmup: RDMA ping-pong + JIT shader pre-compilation + calibration ──
    // Run warmup after connection establishment to ensure RDMA paths are hot
    // and Metal shaders are JIT-compiled before the first real dispatch.
    let warmup_result = {
        let mut state = WarmupState::new();
        let warmup_config = WarmupConfig::default();
        match state.run_warmup(
            &warmup_config,
            // RDMA warmup: no-op for now (connection is already established;
            // a real warmup would send small test messages to verify paths).
            || Ok(()),
            // JIT warmup: no-op (kernel registry is not available at init time;
            // JIT pre-compilation happens when the kernel registry is first used).
            || Ok(()),
        ) {
            Ok(result) => {
                tracing::info!(
                    target: "rmlx_distributed",
                    rdma_warmup = ?result.rdma_warmup,
                    jit_warmup = ?result.jit_warmup,
                    calibration = ?result.calibration,
                    total = ?result.total,
                    "warmup complete",
                );
                Some(result)
            }
            Err(e) => {
                tracing::warn!(target: "rmlx_distributed", %e, "warmup failed (non-fatal)");
                None
            }
        }
    };

    Ok(DistributedContext {
        group,
        rank,
        world_size,
        backend: BackendHint::Rdma,
        warmup: warmup_result,
    })
}

// ── Core init function ──

/// Initialize the distributed context.
///
/// Resolves configuration from `InitConfig` fields, falling back to environment
/// variables, with sensible defaults for single-process operation.
///
/// # Environment Variables
///
/// - `RMLX_RANK` / `RMLX_WORLD_SIZE` -- rank and world size
/// - `RMLX_BACKEND` -- "auto", "rdma", or "loopback"
/// - `RMLX_COORDINATOR` -- coordinator address (rank 0's IP)
/// - `RMLX_COORDINATOR_PORT` -- coordinator port (default 18520)
/// - `RMLX_IBV_DEVICES` -- path to JSON device file
/// - `RMLX_TOPOLOGY` -- "ring", "mesh", or "hybrid"
///
/// Also checks MPI/SLURM compat env vars as fallback for rank/world_size.
pub fn init(config: InitConfig) -> Result<DistributedContext, DistributedError> {
    let rank = resolve_rank(&config);
    let world_size = resolve_world_size(&config);
    let backend = resolve_backend(&config)?;

    // M1: Validate rank < world_size
    if rank >= world_size {
        return Err(DistributedError::Config(format!(
            "rank ({rank}) must be less than world_size ({world_size})"
        )));
    }

    // Single-process or explicit loopback: return immediately.
    if world_size <= 1 || backend == BackendHint::Loopback {
        return loopback_context();
    }

    // Multi-process: attempt RDMA.
    match try_rdma_init(rank, world_size, &config) {
        Ok(ctx) => Ok(ctx),
        Err(e) => {
            if config.strict || backend == BackendHint::Rdma {
                // Strict mode or explicit RDMA request: propagate the error.
                Err(e)
            } else {
                // Auto mode, non-strict: fall back to loopback with a warning.
                tracing::warn!(target: "rmlx_distributed", %e, "RDMA init failed, falling back to loopback");
                loopback_context()
            }
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// Serialize tests that mutate environment variables to avoid races.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Clear all RMLX and compat env vars used by the resolver.
    fn clear_init_env_vars() {
        for var in [
            "RMLX_RANK",
            "RMLX_WORLD_SIZE",
            "RMLX_BACKEND",
            "RMLX_COORDINATOR",
            "RMLX_COORDINATOR_PORT",
            "RMLX_IBV_DEVICES",
            "RMLX_TOPOLOGY",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_RANK",
            "PMI_SIZE",
            "SLURM_PROCID",
            "SLURM_NTASKS",
        ] {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn test_init_single_process() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // No env vars -> defaults to loopback (rank=0, world=1)
        let ctx = init(InitConfig::default()).unwrap();
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
        assert_eq!(ctx.backend, BackendHint::Loopback);
    }

    #[test]
    fn test_init_loopback_explicit() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        let config = InitConfig {
            backend: BackendHint::Loopback,
            rank: Some(0),
            world_size: Some(4),
            ..Default::default()
        };
        let ctx = init(config).unwrap();
        // Loopback always gives world=1, rank=0 regardless of requested world_size
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
        assert_eq!(ctx.backend, BackendHint::Loopback);
    }

    #[test]
    fn test_init_strict_no_rdma() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // strict=true, Rdma backend, no hardware -> should error
        let config = InitConfig {
            strict: true,
            backend: BackendHint::Rdma,
            rank: Some(0),
            world_size: Some(2),
            ..Default::default()
        };
        let result = init(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_init_auto_fallback() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // Auto backend, world_size > 1, no RDMA hardware -> falls back to loopback
        let config = InitConfig {
            strict: false,
            backend: BackendHint::Auto,
            rank: Some(0),
            world_size: Some(2),
            ..Default::default()
        };
        let ctx = init(config).unwrap();
        assert_eq!(ctx.backend, BackendHint::Loopback);
    }

    #[test]
    fn test_init_config_overrides_env() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // Config rank/world_size take precedence
        std::env::set_var("RMLX_RANK", "5");
        std::env::set_var("RMLX_WORLD_SIZE", "10");
        let config = InitConfig {
            rank: Some(0),
            world_size: Some(1),
            ..Default::default()
        };
        let ctx = init(config).unwrap();
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
        clear_init_env_vars();
    }

    #[test]
    fn test_init_compat_env() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // MPI compat env vars should work
        std::env::set_var("OMPI_COMM_WORLD_RANK", "3");
        std::env::set_var("OMPI_COMM_WORLD_SIZE", "8");

        let rank = resolve_rank(&InitConfig::default());
        let ws = resolve_world_size(&InitConfig::default());
        assert_eq!(rank, 3);
        assert_eq!(ws, 8);

        clear_init_env_vars();
    }

    #[test]
    fn test_rank_validation() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // rank >= world_size should fail
        let config = InitConfig {
            rank: Some(4),
            world_size: Some(4),
            ..Default::default()
        };
        let result = init(config);
        assert!(result.is_err());
        match result {
            Err(DistributedError::Config(msg)) => {
                assert!(msg.contains("rank (4) must be less than world_size (4)"));
            }
            Err(other) => panic!("expected Config error, got: {other}"),
            Ok(_) => panic!("expected Config error, got Ok"),
        }
    }

    #[test]
    fn test_rank_validation_overflow() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        // rank > world_size should also fail
        let config = InitConfig {
            rank: Some(10),
            world_size: Some(4),
            ..Default::default()
        };
        let result = init(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_backend_errors() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();
        std::env::set_var("RMLX_BACKEND", "ring");
        let result = resolve_backend(&InitConfig::default());
        assert!(result.is_err());
        match result {
            Err(DistributedError::Config(msg)) => {
                assert!(
                    msg.contains("unknown RMLX_BACKEND"),
                    "expected 'unknown RMLX_BACKEND' in error message, got: {msg}"
                );
            }
            other => panic!("expected Config error, got: {other:?}"),
        }
        clear_init_env_vars();
    }

    #[test]
    fn test_known_backends_resolve() {
        let _guard = ENV_LOCK.lock();
        clear_init_env_vars();

        std::env::set_var("RMLX_BACKEND", "rdma");
        assert_eq!(
            resolve_backend(&InitConfig::default()).unwrap(),
            BackendHint::Rdma
        );

        std::env::set_var("RMLX_BACKEND", "loopback");
        assert_eq!(
            resolve_backend(&InitConfig::default()).unwrap(),
            BackendHint::Loopback
        );

        std::env::set_var("RMLX_BACKEND", "auto");
        assert_eq!(
            resolve_backend(&InitConfig::default()).unwrap(),
            BackendHint::Auto
        );

        std::env::remove_var("RMLX_BACKEND");
        assert_eq!(
            resolve_backend(&InitConfig::default()).unwrap(),
            BackendHint::Auto
        );

        clear_init_env_vars();
    }

    #[test]
    fn test_resolve_topology_explicit() {
        assert_eq!(
            resolve_topology(
                &InitConfig {
                    topology: Some("mesh".into()),
                    ..Default::default()
                },
                None
            ),
            Topology::Mesh
        );
        assert_eq!(
            resolve_topology(
                &InitConfig {
                    topology: Some("ring".into()),
                    ..Default::default()
                },
                None
            ),
            Topology::Ring
        );
    }

    #[test]
    fn test_resolve_topology_from_device_map() {
        // Full mesh device map -> Mesh topology
        let json = r#"[[null, "mlx5_0", "mlx5_1"], ["mlx5_0", null, "mlx5_2"], ["mlx5_1", "mlx5_2", null]]"#;
        let dm = DeviceMap::from_json(json).unwrap();
        let topo = resolve_topology(&InitConfig::default(), Some(&dm));
        assert_eq!(topo, Topology::Mesh);
    }

    #[test]
    fn test_resolve_topology_default_ring() {
        let topo = resolve_topology(&InitConfig::default(), None);
        assert_eq!(topo, Topology::Ring);
    }
}
