//! Simple RDMA init + sendrecv test binary.
//!
//! Usage:
//!   # Rank 0 (coordinator):
//!   RMLX_RANK=0 RMLX_WORLD_SIZE=2 RMLX_COORDINATOR=<coordinator_ip> \
//!     cargo run --release -p rmlx-distributed --example rdma_test
//!
//!   # Rank 1:
//!   RMLX_RANK=1 RMLX_WORLD_SIZE=2 RMLX_COORDINATOR=<coordinator_ip> \
//!     cargo run --release -p rmlx-distributed --example rdma_test

use rmlx_distributed::{init, InitConfig};

fn main() {
    eprintln!("[test] Starting RDMA init test...");

    let config = InitConfig::default();
    let ctx = match init(config) {
        Ok(ctx) => {
            eprintln!(
                "[test] init OK! rank={}, world_size={}, transport={}",
                ctx.group.local_rank(),
                ctx.group.ranks().len(),
                if ctx.group.has_transport() {
                    "rdma"
                } else {
                    "none"
                }
            );
            ctx
        }
        Err(e) => {
            eprintln!("[test] init FAILED: {e}");
            std::process::exit(1);
        }
    };

    let group = &ctx.group;

    // Test 1: barrier
    eprintln!("[test] Testing barrier...");
    match group.barrier() {
        Ok(()) => eprintln!("[test] barrier OK!"),
        Err(e) => {
            eprintln!("[test] barrier FAILED: {e}");
            std::process::exit(1);
        }
    }

    // Test 2: allgather with small data
    let rank = group.local_rank();
    let data = (rank as u32).to_le_bytes();
    eprintln!("[test] Testing allgather with 4 bytes (rank={rank})...");
    match group.allgather(&data) {
        Ok(result) => {
            eprintln!(
                "[test] allgather OK! got {} bytes: {:?}",
                result.len(),
                &result
            );
        }
        Err(e) => {
            eprintln!("[test] allgather FAILED: {e}");
            std::process::exit(1);
        }
    }

    // Test 3: allreduce with small data
    let val: f32 = (rank + 1) as f32;
    let data = val.to_le_bytes();
    eprintln!("[test] Testing allreduce with f32 value={val}...");
    match group.allreduce(&data) {
        Ok(result) => {
            let sum = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
            let expected = (1..=group.ranks().len() as u32).sum::<u32>() as f32;
            eprintln!("[test] allreduce OK! sum={sum} (expected={expected})");
        }
        Err(e) => {
            eprintln!("[test] allreduce FAILED: {e}");
            std::process::exit(1);
        }
    }

    eprintln!("[test] All tests PASSED!");
}
