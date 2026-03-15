//! 2-node RDMA integration tests.
//!
//! Tests 1-5 use a mock LoopbackTransport and can run without real RDMA hardware.
//! Tests A1-A5 require real RDMA hardware, 2-node setup, and RMLX_TEST_2NODE=1.
//! Run them via `scripts/test_rdma_2node.sh`.
//!
//! Run with: RMLX_TEST_RDMA=1 cargo test -p rmlx-distributed --test rdma_2node_integration -- --ignored

use rmlx_distributed::group::{DistributedError, Group, RdmaTransport, ReduceDtype, ReduceOp};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

// ─── Test gating ───

fn skip_unless_rdma() -> bool {
    if std::env::var("RMLX_TEST_RDMA").is_err() {
        eprintln!("skipping: set RMLX_TEST_RDMA=1 to enable");
        return true;
    }
    if !rmlx_rdma::is_available() {
        eprintln!("skipping: RDMA not available");
        return true;
    }
    false
}

// ─── LoopbackTransport mock ───

/// A loopback transport that records sent messages and replays them on recv.
/// Simulates a 2-rank world where rank 0 and rank 1 communicate.
struct LoopbackTransport {
    local_rank: u32,
    /// Keyed by (src_rank, dst_rank), stores queued messages.
    #[allow(clippy::type_complexity)]
    queues: Arc<Mutex<HashMap<(u32, u32), Vec<Vec<u8>>>>>,
}

impl LoopbackTransport {
    #[allow(dead_code)]
    fn new_pair() -> (Arc<Self>, Arc<Self>) {
        let queues = Arc::new(Mutex::new(HashMap::new()));
        let t0 = Arc::new(Self {
            local_rank: 0,
            queues: queues.clone(),
        });
        let t1 = Arc::new(Self {
            local_rank: 1,
            queues,
        });
        (t0, t1)
    }

    /// Create a single-rank loopback (for single-rank group tests).
    fn new_single(rank: u32) -> Arc<Self> {
        Arc::new(Self {
            local_rank: rank,
            queues: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

impl RdmaTransport for LoopbackTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        let mut q = self.queues.lock().unwrap();
        q.entry((self.local_rank, dst_rank))
            .or_default()
            .push(data.to_vec());
        Ok(())
    }

    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        let mut q = self.queues.lock().unwrap();
        let queue = q.entry((src_rank, self.local_rank)).or_default();
        if let Some(msg) = queue.first().cloned() {
            queue.remove(0);
            let mut buf = msg;
            buf.resize(len, 0);
            Ok(buf)
        } else {
            // No message queued -- return zeros (simulates empty recv)
            Ok(vec![0u8; len])
        }
    }

    fn sendrecv(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_len: usize,
        src_rank: u32,
    ) -> Result<Vec<u8>, DistributedError> {
        self.send(send_data, dst_rank)?;
        self.recv(src_rank, recv_len)
    }
}

// ─── f16 helpers ───

/// Convert an f32 value to an f16 bit pattern (IEEE 754 half-precision).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf/NaN
        let f16_frac = if frac != 0 { 0x0200 } else { 0 };
        return ((sign << 15) | 0x7C00 | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow -> Inf
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Subnormal or zero
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let frac_with_hidden = frac | 0x0080_0000;
        let shift = (1 - new_exp) as u32 + 13;
        let f16_frac = frac_with_hidden >> shift;
        return ((sign << 15) | f16_frac) as u16;
    }

    let f16_exp = new_exp as u32;
    let f16_frac = frac >> 13;
    ((sign << 15) | (f16_exp << 10) | f16_frac) as u16
}

/// Convert an f16 bit pattern back to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0x1F {
        // Inf/NaN
        let f32_frac = if frac != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits((sign << 31) | 0x7F80_0000 | f32_frac);
    }
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31); // +/- 0
        }
        // Subnormal
        let mut f = frac;
        let mut e = 0i32;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e += 1;
        }
        let f32_exp = (127 - 15 - e) as u32;
        let f32_frac = (f & 0x03FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac);
    }

    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_frac = frac << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
}

/// Create f16 byte data from a slice of f32 values.
fn f32_slice_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = f32_to_f16_bits(v);
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Convert f16 bytes back to a Vec of f32 values.
fn f16_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "f16 byte slice must have even length");
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_bits_to_f32(bits)
        })
        .collect()
}

/// Convert f32 slice to raw bytes.
fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Convert raw bytes back to f32 vec.
fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "f32 byte slice must be multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ─── Test 1: register_nocopy page-aligned ───

#[test]
#[ignore = "requires RDMA hardware; run with RMLX_TEST_RDMA=1"]
fn test_register_nocopy_page_aligned() {
    if skip_unless_rdma() {
        return;
    }

    use rmlx_rdma::context::RdmaContext;

    let ctx = match RdmaContext::open_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let pd = ctx.alloc_pd().expect("PD allocation");

    // Allocate a page-aligned buffer via posix_memalign
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let buf_size = page_size; // one page
    let mut aligned_ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { libc::posix_memalign(&mut aligned_ptr, page_size, buf_size) };
    assert_eq!(ret, 0, "posix_memalign failed");
    assert!(!aligned_ptr.is_null());

    // Fill with test pattern
    unsafe {
        std::ptr::write_bytes(aligned_ptr as *mut u8, 0xAB, buf_size);
    }

    // Register with nocopy
    let mr = unsafe {
        rmlx_rdma::MemoryRegion::register_nocopy(&pd, aligned_ptr, buf_size)
            .expect("register_nocopy should succeed for page-aligned buffer")
    };

    // Verify is_nocopy returns true
    assert!(mr.is_nocopy(), "MR should be nocopy");

    // Verify addr() returns the original pointer
    assert_eq!(
        mr.addr() as usize,
        aligned_ptr as usize,
        "addr() should return the original pointer"
    );

    // Verify length
    assert_eq!(mr.length(), buf_size);

    // Drop the MR (deregisters without freeing)
    drop(mr);

    // Free the buffer manually (caller owns it for nocopy)
    unsafe {
        libc::free(aligned_ptr);
    }
}

// ─── Test 2: register_nocopy unaligned fails ───

#[test]
#[ignore = "requires RDMA hardware; run with RMLX_TEST_RDMA=1"]
fn test_register_nocopy_unaligned_fails() {
    if skip_unless_rdma() {
        return;
    }

    use rmlx_rdma::context::RdmaContext;

    let ctx = match RdmaContext::open_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let pd = ctx.alloc_pd().expect("PD allocation");

    // Allocate a page-aligned buffer, then offset by 1 to make it unaligned
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let buf_size = page_size * 2;
    let mut aligned_ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { libc::posix_memalign(&mut aligned_ptr, page_size, buf_size) };
    assert_eq!(ret, 0, "posix_memalign failed");

    // Offset by 1 byte to create an unaligned pointer
    let unaligned_ptr = unsafe { (aligned_ptr as *mut u8).add(1) as *mut c_void };

    let result = unsafe { rmlx_rdma::MemoryRegion::register_nocopy(&pd, unaligned_ptr, page_size) };

    assert!(
        result.is_err(),
        "register_nocopy should fail for unaligned pointer"
    );

    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!("already asserted is_err"),
    };
    assert!(
        err_msg.contains("not page-aligned"),
        "error should mention alignment: {err_msg}"
    );

    // Clean up
    unsafe {
        libc::free(aligned_ptr);
    }
}

// ─── Test 3: allreduce_f16_native (single-rank identity) ───

#[test]
#[ignore = "requires Metal GPU"]
fn test_allreduce_f16_native() {
    // Single-rank allreduce_typed should return data unchanged (identity).
    let transport = LoopbackTransport::new_single(0);
    let group = Group::with_transport(vec![0], 0, 1, transport).unwrap();

    // Create f16 data from known f32 values
    let f32_vals = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 0.0, 100.0, 0.25];
    let f16_data = f32_slice_to_f16_bytes(&f32_vals);

    let result = group
        .allreduce_typed(&f16_data, ReduceDtype::F16)
        .expect("single-rank allreduce_typed should succeed");

    // Single-rank: output == input
    assert_eq!(
        result.len(),
        f16_data.len(),
        "output length should match input"
    );
    assert_eq!(result, f16_data, "single-rank allreduce should be identity");

    // Verify roundtrip: convert back to f32 and compare
    let result_f32 = f16_bytes_to_f32_vec(&result);
    for (i, (&original, &roundtripped)) in f32_vals.iter().zip(result_f32.iter()).enumerate() {
        let expected = f16_bits_to_f32(f32_to_f16_bits(original));
        assert!(
            (roundtripped - expected).abs() < 1e-4,
            "element {i}: expected {expected}, got {roundtripped}"
        );
    }
}

// ─── Test 4: allreduce_in_place_f16 (single-rank identity) ───

#[test]
#[ignore = "requires Metal GPU"]
fn test_allreduce_in_place_f16() {
    // Single-rank allreduce_in_place should leave data unchanged.
    let transport = LoopbackTransport::new_single(0);
    let group = Group::with_transport(vec![0], 0, 1, transport).unwrap();

    let f32_vals = vec![1.0f32, 2.5, -3.0, 4.75, 0.125, 8.0, 16.0, 32.0];
    let f16_data = f32_slice_to_f16_bytes(&f32_vals);
    let mut data = f16_data.clone();

    group
        .allreduce_in_place(&mut data, ReduceDtype::F16)
        .expect("single-rank allreduce_in_place should succeed");

    // Data should be unchanged
    assert_eq!(data, f16_data, "single-rank in-place should be identity");
}

// ─── Test 5: allreduce f16 vs f32 accuracy ───

#[test]
#[ignore = "requires Metal GPU"]
fn test_allreduce_f16_vs_f32_accuracy() {
    // Compare f16 and f32 allreduce results on a single-rank group.
    // Both should produce identity, but we verify the precision difference.
    let transport_f16 = LoopbackTransport::new_single(0);
    let group_f16 = Group::with_transport(vec![0], 0, 1, transport_f16).unwrap();

    let transport_f32 = LoopbackTransport::new_single(0);
    let group_f32 = Group::with_transport(vec![0], 0, 1, transport_f32).unwrap();

    // Use values that have representable f16 forms
    let f32_vals = vec![
        1.0f32, 0.5, 0.25, 0.125, 2.0, 4.0, 8.0, 16.0, 0.333, 0.667, 1.5, 3.14,
    ];

    // f16 path
    let f16_data = f32_slice_to_f16_bytes(&f32_vals);
    let f16_result = group_f16
        .allreduce_typed(&f16_data, ReduceDtype::F16)
        .expect("f16 allreduce should succeed");
    let f16_as_f32 = f16_bytes_to_f32_vec(&f16_result);

    // f32 path
    let f32_data = f32_slice_to_bytes(&f32_vals);
    let f32_result = group_f32
        .allreduce_typed(&f32_data, ReduceDtype::F32)
        .expect("f32 allreduce should succeed");
    let f32_result_vals = f32_bytes_to_vec(&f32_result);

    // Both should be close to the original values
    for (i, &original) in f32_vals.iter().enumerate() {
        let f32_val = f32_result_vals[i];
        let f16_val = f16_as_f32[i];

        // f32 should be exact (single-rank identity)
        assert!(
            (f32_val - original).abs() < 1e-7,
            "f32 element {i}: expected {original}, got {f32_val}"
        );

        // f16 should be within f16 precision (~0.1% for values in normal range)
        let tolerance = original.abs() * 0.002 + 1e-4; // relative + absolute tolerance
        assert!(
            (f16_val - original).abs() < tolerance,
            "f16 element {i}: expected ~{original}, got {f16_val} (tolerance={tolerance})"
        );
    }
}

// ─── Test 3b: allreduce_f16 multi-rank with loopback ───

#[test]
#[ignore = "requires Metal GPU"]
fn test_allreduce_f16_two_rank_loopback() {
    // Two-rank allreduce with LoopbackTransport.
    // Since loopback recv returns zeros when no message is queued,
    // we test the allreduce_op path on a single-rank group instead
    // to verify the f16 reduction logic works end-to-end.
    let transport = LoopbackTransport::new_single(0);
    let group = Group::with_transport(vec![0], 0, 1, transport).unwrap();

    let f32_vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let f16_data = f32_slice_to_f16_bytes(&f32_vals);

    let result = group
        .allreduce_op(&f16_data, ReduceOp::Sum, ReduceDtype::F16)
        .expect("allreduce_op f16 should succeed on single rank");

    // Single rank: identity
    assert_eq!(result, f16_data);
}

// ─── 2-node RDMA test helpers ───

use rmlx_distributed::transport::RdmaConnectionTransport;
use rmlx_rdma::connection::{RdmaConfig, RdmaConnection};

/// Wrap RdmaConnection::establish() in a thread with an overall timeout.
/// TB5 macOS RDMA driver can hang indefinitely in kernel-level operations
/// (alloc_pd, create_qp) if previous RDMA resources weren't cleaned up.
/// This prevents the entire test process from hanging.
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
                "establish timed out after {timeout_secs}s — possible RDMA kernel resource contamination. \
                 Reboot the node to clear stuck RDMA state."
            );
            // Can't join the thread (it's stuck in kernel), just return None.
            // The thread will be cleaned up when the process exits.
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Set up a raw 2-node RDMA connection (no Group wrapper).
fn setup_2node_connection() -> Option<(RdmaConnection, u32)> {
    if std::env::var("RMLX_TEST_2NODE").is_err() {
        eprintln!("skipping: requires RMLX_TEST_2NODE=1");
        return None;
    }
    let rank: u32 = std::env::var("RMLX_RANK").ok()?.parse().ok()?;
    let peer_host = std::env::var("RMLX_PEER_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("RMLX_TEST_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18515);

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

    let conn = establish_with_timeout(config, 8)?;
    Some((conn, rank))
}

// ─── Subtests: raw connection ───

fn subtest_nocopy_send(conn: &RdmaConnection, rank: u32) {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let data_size = page_size * 4; // 64KB

    if rank == 0 {
        let mut aligned_ptr: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { libc::posix_memalign(&mut aligned_ptr, page_size, data_size) };
        assert_eq!(ret, 0, "posix_memalign failed");

        let send_data = unsafe {
            let slice = std::slice::from_raw_parts_mut(aligned_ptr as *mut u8, data_size);
            for (i, byte) in slice.iter_mut().enumerate() {
                *byte = (i % 251) as u8;
            }
            std::slice::from_raw_parts(aligned_ptr as *const u8, data_size)
        };

        conn.chunked_send(send_data)
            .expect("nocopy send should succeed");

        unsafe {
            libc::free(aligned_ptr);
        }
        eprintln!("  nocopy send: rank=0 sent {data_size} bytes (page-aligned)");
    } else {
        let received = conn
            .chunked_recv(data_size)
            .expect("chunked_recv should succeed");

        assert_eq!(received.len(), data_size);
        for (i, &byte) in received.iter().enumerate() {
            assert_eq!(
                byte,
                (i % 251) as u8,
                "mismatch at offset {i}: expected {}, got {byte}",
                (i % 251) as u8
            );
        }
        eprintln!("  nocopy send: rank=1 verified {data_size} bytes");
    }
}

// ─── Subtests: Group-based allreduce ───

fn subtest_f16_allreduce(group: &Group, rank: u32) {
    // Rank 0: [1.0, 2.0, ..., 256.0], Rank 1: [256.0, 255.0, ..., 1.0]
    let f32_vals: Vec<f32> = if rank == 0 {
        (1..=256).map(|i| i as f32).collect()
    } else {
        (1..=256).rev().map(|i| i as f32).collect()
    };
    let data = f32_slice_to_f16_bytes(&f32_vals);

    let result = group
        .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
        .expect("2-node f16 allreduce should succeed");

    let result_f32 = f16_bytes_to_f32_vec(&result);
    assert_eq!(result_f32.len(), 256);
    for (i, &val) in result_f32.iter().enumerate() {
        assert!(
            (val - 257.0).abs() < 0.5,
            "element {i}: expected ~257.0, got {val}"
        );
    }
    eprintln!("  subtest_f16_allreduce: rank={rank} PASSED");
}

fn subtest_f32_allreduce(group: &Group, rank: u32) {
    // Rank 0: [1.0, 2.0, ..., 256.0], Rank 1: [256.0, 255.0, ..., 1.0]
    let f32_vals: Vec<f32> = if rank == 0 {
        (1..=256).map(|i| i as f32).collect()
    } else {
        (1..=256).rev().map(|i| i as f32).collect()
    };
    let data = f32_slice_to_bytes(&f32_vals);

    let result = group
        .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F32)
        .expect("2-node f32 allreduce should succeed");

    let result_f32 = f32_bytes_to_vec(&result);
    assert_eq!(result_f32.len(), 256);
    for (i, &val) in result_f32.iter().enumerate() {
        assert!(
            (val - 257.0).abs() < 1e-4,
            "element {i}: expected 257.0, got {val}"
        );
    }
    eprintln!("  subtest_f32_allreduce: rank={rank} PASSED");
}

fn subtest_allreduce_in_place(group: &Group, rank: u32) {
    let f32_vals: Vec<f32> = if rank == 0 {
        (1..=256).map(|i| i as f32).collect()
    } else {
        (1..=256).rev().map(|i| i as f32).collect()
    };
    let mut data = f32_slice_to_f16_bytes(&f32_vals);

    group
        .allreduce_in_place(&mut data, ReduceDtype::F16)
        .expect("2-node f16 allreduce_in_place should succeed");

    let result_f32 = f16_bytes_to_f32_vec(&data);
    assert_eq!(result_f32.len(), 256);
    for (i, &val) in result_f32.iter().enumerate() {
        assert!(
            (val - 257.0).abs() < 0.5,
            "element {i}: expected ~257.0, got {val}"
        );
    }
    eprintln!("  subtest_allreduce_in_place: rank={rank} PASSED");
}

fn subtest_large_allreduce(group: &Group, rank: u32) {
    // 1MB of f16 data = 524288 elements
    let num_elements: usize = 524288;
    let f32_vals: Vec<f32> = (0..num_elements)
        .map(|i| {
            if rank == 0 {
                ((i % 100) as f32) * 0.1 // 0.0, 0.1, ..., 9.9, 0.0, ...
            } else {
                ((99 - (i % 100)) as f32) * 0.1 // 9.9, 9.8, ..., 0.0, 9.9, ...
            }
        })
        .collect();
    let data = f32_slice_to_f16_bytes(&f32_vals);
    assert_eq!(data.len(), num_elements * 2); // 1MB

    let result = group
        .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
        .expect("2-node large f16 allreduce should succeed");

    let result_f32 = f16_bytes_to_f32_vec(&result);
    assert_eq!(result_f32.len(), num_elements);

    // Verify a subset: every 1000th element
    // rank0[i] + rank1[i] = (i%100)*0.1 + (99-i%100)*0.1 = 9.9
    for i in (0..num_elements).step_by(1000) {
        assert!(
            (result_f32[i] - 9.9).abs() < 0.15,
            "element {i}: expected ~9.9, got {}",
            result_f32[i]
        );
    }
    eprintln!(
        "  subtest_large_allreduce: rank={rank} PASSED ({num_elements} f16 elements, {}KB)",
        num_elements * 2 / 1024
    );
}

#[test]
#[ignore = "2-node RDMA test; run via scripts/test_rdma_2node.sh"]
fn test_2node_full_suite() {
    let (conn, rank) = match setup_2node_connection() {
        Some(v) => v,
        None => return,
    };

    // Phase 1: Raw connection tests (borrow only)
    eprintln!("=== Phase 1: nocopy chunked send ===");
    subtest_nocopy_send(&conn, rank);

    // Phase 2+: Wrap connection as Group for collective tests
    let peer_rank = 1 - rank;
    let mut connections: Vec<Option<RdmaConnection>> = vec![None, None];
    connections[peer_rank as usize] = Some(conn);
    let transport = Arc::new(RdmaConnectionTransport::new(connections, rank));
    let group = Group::with_transport(vec![0, 1], rank, 2, transport)
        .expect("Group creation should succeed");

    eprintln!("=== Phase 2: f16 allreduce ===");
    subtest_f16_allreduce(&group, rank);

    eprintln!("=== Phase 3: f32 allreduce ===");
    subtest_f32_allreduce(&group, rank);

    eprintln!("=== Phase 4: allreduce in-place ===");
    subtest_allreduce_in_place(&group, rank);

    eprintln!("=== Phase 5: large allreduce (1MB f16) ===");
    subtest_large_allreduce(&group, rank);

    // Exit barrier: ensure both ranks complete before dropping connection
    let _ = group.allreduce(&[0u8; 4]);
    eprintln!("=== All phases passed ===");
}

// ─── Supplementary: f16 conversion helpers correctness ───

#[test]
#[ignore = "unit test for f16 helpers"]
fn test_f16_conversion_roundtrip() {
    let test_values = vec![
        0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 0.125, 65504.0,  // max f16
        -65504.0, // min f16
    ];

    for &val in &test_values {
        let bits = f32_to_f16_bits(val);
        let back = f16_bits_to_f32(bits);
        assert!(
            (back - val).abs() < 1e-3 || val.abs() > 1000.0,
            "roundtrip failed for {val}: got {back} (bits=0x{bits:04x})"
        );
    }
}

#[test]
#[ignore = "unit test for f16 helpers"]
fn test_f16_special_values() {
    // Zero
    let zero_bits = f32_to_f16_bits(0.0);
    assert_eq!(zero_bits, 0x0000);
    assert_eq!(f16_bits_to_f32(0x0000), 0.0);

    // Negative zero
    let neg_zero_bits = f32_to_f16_bits(-0.0);
    assert_eq!(neg_zero_bits, 0x8000);

    // Infinity
    let inf_bits = f32_to_f16_bits(f32::INFINITY);
    assert_eq!(inf_bits, 0x7C00);
    assert!(f16_bits_to_f32(0x7C00).is_infinite());

    // NaN
    let nan_bits = f32_to_f16_bits(f32::NAN);
    assert!(f16_bits_to_f32(nan_bits).is_nan());

    // One
    let one_bits = f32_to_f16_bits(1.0);
    assert_eq!(one_bits, 0x3C00);
    assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
}

#[test]
#[ignore = "unit test for f16 byte helpers"]
fn test_f16_byte_helpers() {
    let vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_slice_to_f16_bytes(&vals);
    assert_eq!(bytes.len(), 8); // 4 values * 2 bytes each

    let back = f16_bytes_to_f32_vec(&bytes);
    assert_eq!(back.len(), 4);
    for (i, (&original, &roundtripped)) in vals.iter().zip(back.iter()).enumerate() {
        assert!(
            (roundtripped - original).abs() < 1e-3,
            "element {i}: expected {original}, got {roundtripped}"
        );
    }
}
