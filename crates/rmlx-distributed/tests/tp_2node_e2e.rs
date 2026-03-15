//! 2-node Tensor Parallel matmul E2E test.
//!
//! Gates on `RMLX_TEST_2NODE=1` + `RMLX_TEST_RDMA=1`.
//! Run via `scripts/test_rdma_2node.sh`.
//!
//! Uses the same connection pattern as `rdma_2node_integration.rs`.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_distributed::group::Group;
use rmlx_distributed::transport::RdmaConnectionTransport;
use rmlx_metal::device::GpuDevice;
use rmlx_nn::parallel::{ColumnParallelLinear, RowParallelLinear};
use rmlx_rdma::connection::{RdmaConfig, RdmaConnection};
use std::sync::Arc;

// ─── Connection helpers ───

/// Wrap `RdmaConnection::establish()` with an overall timeout to prevent
/// hangs from stuck RDMA kernel state.
fn establish_with_timeout(config: RdmaConfig, timeout_secs: u64) -> Option<RdmaConnection> {
    let handle = std::thread::spawn(move || RdmaConnection::establish(config));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if handle.is_finished() {
            return match handle.join() {
                Ok(Ok(conn)) => Some(conn),
                Ok(Err(e)) => {
                    eprintln!("establish failed: {e}");
                    None
                }
                Err(_) => {
                    eprintln!("establish thread panicked");
                    None
                }
            };
        }
        if std::time::Instant::now() >= deadline {
            eprintln!(
                "establish timed out after {timeout_secs}s — possible RDMA kernel resource contamination."
            );
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

// ─── Main test ───

#[test]
#[ignore = "2-node TP E2E; run via scripts/test_rdma_2node.sh"]
fn test_tp_2node_full_suite() {
    // Skip unless 2-node env set
    if std::env::var("RMLX_TEST_2NODE").is_err() || std::env::var("RMLX_TEST_RDMA").is_err() {
        eprintln!("skipping: requires RMLX_TEST_2NODE=1 + RMLX_TEST_RDMA=1");
        return;
    }

    // 1. Create Metal device + KernelRegistry + CommandQueue
    let mtl_device = objc2::rc::autoreleasepool(|_| objc2_metal::MTLCreateSystemDefaultDevice())
        .expect("Metal device");
    let gpu = GpuDevice::from_raw_device(mtl_device);
    let queue = gpu.new_command_queue();
    let registry = KernelRegistry::new(gpu);

    // 2. Setup RDMA connection
    let rank: u32 = std::env::var("RMLX_RANK")
        .expect("RMLX_RANK required")
        .parse()
        .expect("RMLX_RANK must be u32");
    let peer_host = std::env::var("RMLX_PEER_HOST").expect("RMLX_PEER_HOST required");
    let port: u16 = std::env::var("RMLX_TEST_PORT")
        .unwrap_or("18515".into())
        .parse()
        .unwrap();

    let config = RdmaConfig {
        rank,
        world_size: 2,
        peer_host,
        exchange_port: port,
        sync_port: port + 1,
        accept_timeout_secs: 5,
        connect_timeout_ms: 1000,
        io_max_retries: 1,
        io_retry_delay_ms: 200,
        ..Default::default()
    };

    let conn = match establish_with_timeout(config, 8) {
        Some(c) => c,
        None => {
            eprintln!("skipping: RDMA connection failed");
            return;
        }
    };

    let peer_rank = 1 - rank;
    let mut connections: Vec<Option<RdmaConnection>> = vec![None, None];
    connections[peer_rank as usize] = Some(conn);
    let transport = Arc::new(RdmaConnectionTransport::new(connections, rank));
    let group = Group::with_transport(vec![0, 1], rank, 2, transport).expect("Group");

    // 3. Subtest: RowParallel
    eprintln!("=== Subtest 1: RowParallelLinear ===");
    subtest_row_parallel(registry.device().raw(), &registry, &queue, &group, rank);

    // 4. Subtest: ColumnParallel
    eprintln!("=== Subtest 2: ColumnParallelLinear ===");
    subtest_column_parallel(registry.device().raw(), &registry, &queue, &group, rank);

    // Exit barrier
    let _ = group.allreduce(&[0u8; 4]);
    eprintln!("=== All subtests passed ===");
}

// ─── Subtest: RowParallelLinear ───

fn subtest_row_parallel(
    device: &ProtocolObject<dyn MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    group: &Group,
    rank: u32,
) {
    // RowParallel: each rank has W_shard [out=16, in/2=8]
    // Input: x_shard [batch=2, in/2=8]
    // After allreduce: result [batch=2, out=16] = sum of partial matmuls

    let out_features = 16usize;
    let in_features = 16usize;
    let shard_in = in_features / 2;
    let batch = 2usize;

    // Create weight shard [out_features, shard_in] with deterministic values.
    // Each rank gets a different column slice of the full weight matrix.
    let w_data: Vec<f32> = (0..out_features * shard_in)
        .map(|i| {
            let row = i / shard_in;
            let col = i % shard_in + rank as usize * shard_in;
            ((row * in_features + col) as f32) * 0.01
        })
        .collect();
    let w_bytes: Vec<u8> = w_data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let weight = Array::from_bytes(
        device,
        &w_bytes,
        vec![out_features, shard_in],
        DType::Float32,
    );

    let row_linear = RowParallelLinear::new(weight, None, out_features, in_features, rank, 2)
        .expect("RowParallelLinear");

    // Create input shard [batch, shard_in].
    // Each rank gets a different column slice of the full input.
    let x_data: Vec<f32> = (0..batch * shard_in)
        .map(|i| {
            let row = i / shard_in;
            let col = i % shard_in + rank as usize * shard_in;
            ((row * in_features + col) as f32) * 0.1
        })
        .collect();
    let x_bytes: Vec<u8> = x_data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let input = Array::from_bytes(device, &x_bytes, vec![batch, shard_in], DType::Float32);

    // Forward (GPU matmul + allreduce)
    let result = row_linear
        .forward_with_group(&input, group, registry, queue)
        .expect("RowParallel forward_with_group");

    assert_eq!(result.shape(), &[batch, out_features]);

    // Read result back
    let result_bytes = result.to_bytes();
    let result_f32: Vec<f32> = result_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Sanity: all values finite and non-zero (matmul of non-zero data)
    for (i, &val) in result_f32.iter().enumerate() {
        assert!(val.is_finite(), "element {i} is not finite: {val}");
    }
    assert!(
        result_f32.iter().any(|&v| v.abs() > 1e-6),
        "all-zero result — matmul likely failed"
    );

    // CPU reference: y = x_full @ W_full^T
    // W_full[r][c] = (r * in_features + c) * 0.01
    // x_full[b][c] = (b * in_features + c) * 0.1
    let mut expected = vec![0.0f32; batch * out_features];
    for b in 0..batch {
        for r in 0..out_features {
            let mut sum = 0.0f32;
            for c in 0..in_features {
                let w_val = (r * in_features + c) as f32 * 0.01;
                let x_val = (b * in_features + c) as f32 * 0.1;
                sum += x_val * w_val;
            }
            expected[b * out_features + r] = sum;
        }
    }

    for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let tol = exp.abs() * 0.01 + 0.01; // 1% relative + small absolute
        assert!(
            (got - exp).abs() < tol,
            "RowParallel element {i}: expected {exp:.4}, got {got:.4} (tol={tol:.4})"
        );
    }

    eprintln!(
        "  RowParallel: rank={rank}, shape={:?}, first_val={:.4} OK",
        result.shape(),
        result_f32[0]
    );
}

// ─── Subtest: ColumnParallelLinear ───

fn subtest_column_parallel(
    device: &ProtocolObject<dyn MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    group: &Group,
    rank: u32,
) {
    // ColumnParallel: each rank has W_shard [out/2=8, in=16]
    // Input: x [batch=2, in=16] (same on all ranks)
    // After allgather: result [batch=2, out=16]

    let out_features = 16usize;
    let in_features = 16usize;
    let shard_out = out_features / 2;
    let batch = 2usize;

    // Create weight shard [shard_out, in_features].
    // Each rank gets a different row slice of the full weight matrix.
    let w_data: Vec<f32> = (0..shard_out * in_features)
        .map(|i| {
            let local_row = i / in_features;
            let col = i % in_features;
            let global_row = local_row + rank as usize * shard_out;
            ((global_row * in_features + col) as f32) * 0.01
        })
        .collect();
    let w_bytes: Vec<u8> = w_data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let weight = Array::from_bytes(
        device,
        &w_bytes,
        vec![shard_out, in_features],
        DType::Float32,
    );

    let col_linear = ColumnParallelLinear::new(weight, None, out_features, in_features, rank, 2)
        .expect("ColumnParallelLinear");

    // Create input [batch, in_features] — same data on both ranks
    let x_data: Vec<f32> = (0..batch * in_features)
        .map(|i| {
            let row = i / in_features;
            let col = i % in_features;
            ((row * in_features + col) as f32) * 0.1
        })
        .collect();
    let x_bytes: Vec<u8> = x_data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let input = Array::from_bytes(device, &x_bytes, vec![batch, in_features], DType::Float32);

    // Forward (GPU matmul + allgather)
    let result = col_linear
        .forward_with_group(&input, group, registry, queue)
        .expect("ColumnParallel forward_with_group");

    assert_eq!(result.shape(), &[batch, out_features]);

    // Read result back
    let result_bytes = result.to_bytes();
    let result_f32: Vec<f32> = result_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Sanity checks
    for (i, &val) in result_f32.iter().enumerate() {
        assert!(val.is_finite(), "element {i} is not finite: {val}");
    }
    assert!(
        result_f32.iter().any(|&v| v.abs() > 1e-6),
        "all-zero result — matmul likely failed"
    );

    // CPU reference: y = x @ W_full^T
    // W_full[r][c] = (r * in_features + c) * 0.01
    // x[b][c] = (b * in_features + c) * 0.1
    let mut expected = vec![0.0f32; batch * out_features];
    for b in 0..batch {
        for r in 0..out_features {
            let mut sum = 0.0f32;
            for c in 0..in_features {
                let w_val = (r * in_features + c) as f32 * 0.01;
                let x_val = (b * in_features + c) as f32 * 0.1;
                sum += x_val * w_val;
            }
            expected[b * out_features + r] = sum;
        }
    }

    for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let tol = exp.abs() * 0.01 + 0.01;
        assert!(
            (got - exp).abs() < tol,
            "ColumnParallel element {i}: expected {exp:.4}, got {got:.4} (tol={tol:.4})"
        );
    }

    eprintln!(
        "  ColumnParallel: rank={rank}, shape={:?}, first_val={:.4} OK",
        result.shape(),
        result_f32[0]
    );
}
