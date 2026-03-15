//! QP reuse test — validates that destroying and re-creating QP/CQ/PD/MR
//! on the same device works correctly (tests RMLX wrapper Drop behavior).
//!
//! Usage:
//!   RANK=0 PEER_IP=10.254.0.6 cargo run --release -p rmlx-rdma --example qp_reuse_test
//!   RANK=1 PEER_IP=10.254.0.5 cargo run --release -p rmlx-rdma --example qp_reuse_test

use std::io::{Read, Write};
use std::mem::MaybeUninit;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

use rmlx_rdma::ffi::{
    self, IbvRecvWr, IbvSendWr, IbvSge, IbvWc,
};
use rmlx_rdma::{CompletionQueue, MemoryRegion, QpInfo, QueuePair, RdmaContext};

const BUF_SIZE: usize = 4096;
const TCP_BASE_PORT: u16 = 19999;
const QP_INFO_WIRE_SIZE: usize = 26;
const CQ_POLL_TIMEOUT_SECS: u64 = 5;

fn main() {
    let rank: u32 = std::env::var("RANK")
        .expect("RANK env not set")
        .parse()
        .expect("RANK must be 0 or 1");
    let peer_ip = std::env::var("PEER_IP").expect("PEER_IP env not set");

    eprintln!("[rank {rank}] QP reuse test starting (peer={peer_ip})");

    // Open device once — kept alive across both rounds
    let ctx = RdmaContext::open_by_name("rdma_en5").expect("failed to open rdma_en5");
    eprintln!("[rank {rank}] Device opened: {}", ctx.device_name());

    // ── Round 1: RMLX wrappers with Drop (RESET transition) ──
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("[rank {rank}] === ROUND 1 (RMLX wrappers, Drop will RESET QP) ===");
    eprintln!("{}", "=".repeat(60));
    run_round(&ctx, rank, &peer_ip, 1);

    // ── Round 2: Same device, new resources — tests QP reuse after Drop ──
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("[rank {rank}] === ROUND 2 (same device, new QP/CQ/PD/MR) ===");
    eprintln!("{}", "=".repeat(60));
    run_round(&ctx, rank, &peer_ip, 2);

    eprintln!("\n[rank {rank}] Both rounds completed successfully!");
}

fn run_round(ctx: &RdmaContext, rank: u32, peer_ip: &str, round: u32) {
    // Step 1: Alloc PD, create CQ, create UC QP
    let pd = ctx.alloc_pd().expect("failed to alloc PD");
    eprintln!("[rank {rank}] round {round}: PD allocated");

    let cq = CompletionQueue::new(ctx).expect("failed to create CQ");
    eprintln!("[rank {rank}] round {round}: CQ created");

    let mut qp = QueuePair::create_uc(&pd, &cq, ctx).expect("failed to create QP");
    eprintln!("[rank {rank}] round {round}: UC QP created");

    // Step 2: Query local QP info (gid_index=1, psn=7)
    qp.query_local_info(ctx, rank)
        .expect("failed to query local info");
    let local_info = qp.local_info().clone();
    eprintln!(
        "[rank {rank}] round {round}: local qpn={}, psn={}, lid={}, gid={:02x?}",
        local_info.qpn, local_info.psn, local_info.lid, &local_info.gid
    );

    // Step 3: Exchange QP info via TCP
    let tcp_port = TCP_BASE_PORT + (round - 1) as u16 * 2;
    let remote_info = tcp_exchange_qp_info(rank, peer_ip, &local_info, tcp_port);
    eprintln!(
        "[rank {rank}] round {round}: remote qpn={}, psn={}, lid={}, gid={:02x?}",
        remote_info.qpn, remote_info.psn, remote_info.lid, &remote_info.gid
    );

    // Step 4: Register MR (4096-byte posix_memalign buffer)
    // Allocate a source buffer; MemoryRegion::register copies it to a page-aligned buffer
    let mut src_buf = vec![0u8; BUF_SIZE];
    // Fill with a pattern: rank-specific so we can verify on recv
    let fill_byte = if rank == 0 { 0xAA } else { 0xBB };
    src_buf.iter_mut().for_each(|b| *b = fill_byte);

    let mr = unsafe {
        MemoryRegion::register(&pd, src_buf.as_mut_ptr() as *mut std::ffi::c_void, BUF_SIZE)
    }
    .expect("failed to register MR");
    eprintln!(
        "[rank {rank}] round {round}: MR registered (addr={:?}, len={}, lkey={})",
        mr.addr(),
        mr.length(),
        mr.lkey()
    );

    // Step 5: Connect QP (RESET -> INIT -> RTR -> RTS)
    qp.connect(&remote_info)
        .expect("failed to connect QP");
    eprintln!("[rank {rank}] round {round}: QP connected (RESET->INIT->RTR->RTS)");

    // Step 6: Bidirectional sendrecv
    // 6a: Post recv
    let mut recv_sge = IbvSge {
        addr: mr.addr() as u64,
        length: BUF_SIZE as u32,
        lkey: mr.lkey(),
    };
    let mut recv_wr = IbvRecvWr {
        wr_id: 1000 + round as u64,
        next: std::ptr::null_mut(),
        sg_list: &mut recv_sge,
        num_sge: 1,
    };
    qp.post_recv(&mut recv_wr)
        .expect("failed to post recv");
    eprintln!("[rank {rank}] round {round}: recv posted");

    // 6b: TCP barrier (ensure both sides have posted recv before sending)
    tcp_barrier(rank, peer_ip, tcp_port + 1);
    eprintln!("[rank {rank}] round {round}: barrier passed");

    // 6c: Post send (using MaybeUninit matching JACCL pattern)
    let mut send_sge = IbvSge {
        addr: mr.addr() as u64,
        length: BUF_SIZE as u32,
        lkey: mr.lkey(),
    };

    // MaybeUninit for IbvSendWr (matching JACCL)
    #[allow(invalid_value)]
    let mut send_wr: IbvSendWr = unsafe { MaybeUninit::uninit().assume_init() };
    send_wr.wr_id = 2000 + round as u64;
    send_wr.next = std::ptr::null_mut();
    send_wr.sg_list = &mut send_sge;
    send_wr.num_sge = 1;
    send_wr.opcode = ffi::wr_opcode::SEND;
    send_wr.send_flags = ffi::send_flags::SIGNALED;
    send_wr.imm_data = 0;
    // Zero the wr union fields we don't use
    send_wr.wr.ah = std::ptr::null_mut();
    send_wr.wr.remote_addr = 0;
    send_wr.wr.rkey = 0;

    qp.post_send(&mut send_wr)
        .expect("failed to post send");
    eprintln!("[rank {rank}] round {round}: send posted");

    // 6d: Poll CQ for both send and recv completions (expect 2)
    let mut completions_remaining = 2u32;
    let start = Instant::now();
    let mut wc = [IbvWc::zeroed(); 4];

    while completions_remaining > 0 {
        if start.elapsed().as_secs() >= CQ_POLL_TIMEOUT_SECS {
            eprintln!(
                "[rank {rank}] round {round}: TIMEOUT waiting for completions ({completions_remaining} remaining)"
            );
            std::process::exit(1);
        }

        let n = cq.poll(&mut wc).expect("CQ poll error");
        for i in 0..n {
            let wc_entry = &wc[i];
            if wc_entry.status != ffi::wc_status::SUCCESS {
                eprintln!(
                    "[rank {rank}] round {round}: FAILURE — wr_id={}, status={} ({})",
                    wc_entry.wr_id,
                    wc_entry.status,
                    ffi::wc_status_str(wc_entry.status)
                );
                std::process::exit(1);
            }
            eprintln!(
                "[rank {rank}] round {round}: completion OK — wr_id={}, opcode={}, byte_len={}",
                wc_entry.wr_id, wc_entry.opcode, wc_entry.byte_len
            );
            completions_remaining -= 1;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "[rank {rank}] round {round}: SUCCESS — 2 completions in {elapsed:?}"
    );

    // Step 7: Verify received data
    let expected_byte: u8 = if rank == 0 { 0xBB } else { 0xAA };
    let recv_buf = unsafe { std::slice::from_raw_parts(mr.addr() as *const u8, BUF_SIZE) };
    let correct = recv_buf.iter().all(|&b| b == expected_byte);
    if correct {
        eprintln!(
            "[rank {rank}] round {round}: Data verification PASSED (all bytes = 0x{expected_byte:02X})"
        );
    } else {
        // Show first few bytes for debugging
        eprintln!(
            "[rank {rank}] round {round}: Data verification FAILED — first 16 bytes: {:02x?}",
            &recv_buf[..16]
        );
        std::process::exit(1);
    }

    // Resources (MR, QP, CQ, PD) are dropped here via RMLX wrappers.
    // QueuePair::drop will transition QP to RESET before ibv_destroy_qp.
    // This is the key behavior we're testing — does Round 2 work after this?
    eprintln!("[rank {rank}] round {round}: dropping MR, QP, CQ, PD...");
    drop(mr);
    drop(qp);
    drop(cq);
    drop(pd);
    eprintln!("[rank {rank}] round {round}: all resources dropped");
}

/// Exchange QP info over TCP. Rank 0 listens, rank 1 connects.
fn tcp_exchange_qp_info(rank: u32, peer_ip: &str, local: &QpInfo, port: u16) -> QpInfo {
    let local_wire = local.to_wire();

    if rank == 0 {
        // Listen and accept
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
            .expect("failed to bind TCP listener");
        eprintln!("[rank 0] TCP listening on port {port}...");
        let (mut stream, addr) = listener.accept().expect("failed to accept");
        eprintln!("[rank 0] TCP accepted from {addr}");

        // Send local info
        stream.write_all(&local_wire).expect("TCP send failed");
        stream.flush().expect("TCP flush failed");

        // Recv remote info
        let mut remote_wire = [0u8; QP_INFO_WIRE_SIZE];
        stream
            .read_exact(&mut remote_wire)
            .expect("TCP recv failed");

        QpInfo::from_wire(remote_wire)
    } else {
        // Connect to rank 0
        let addr = format!("{peer_ip}:{port}");
        eprintln!("[rank 1] TCP connecting to {addr}...");
        let mut stream = TcpStream::connect(&addr).expect("failed to connect TCP");
        eprintln!("[rank 1] TCP connected");

        // Recv remote info first (rank 0 sends first)
        let mut remote_wire = [0u8; QP_INFO_WIRE_SIZE];
        stream
            .read_exact(&mut remote_wire)
            .expect("TCP recv failed");

        // Send local info
        stream.write_all(&local_wire).expect("TCP send failed");
        stream.flush().expect("TCP flush failed");

        QpInfo::from_wire(remote_wire)
    }
}

/// Simple 1-byte TCP barrier between both ranks.
fn tcp_barrier(rank: u32, peer_ip: &str, barrier_port: u16) {

    if rank == 0 {
        let listener = TcpListener::bind(format!("0.0.0.0:{barrier_port}"))
            .expect("failed to bind barrier listener");
        let (mut stream, _) = listener.accept().expect("failed to accept barrier");
        let mut buf = [0u8; 1];
        stream.read_exact(&mut buf).expect("barrier recv failed");
        stream.write_all(&[0x42]).expect("barrier send failed");
        stream.flush().expect("barrier flush failed");
    } else {
        let addr = format!("{peer_ip}:{barrier_port}");
        let mut stream = TcpStream::connect(&addr).expect("failed to connect barrier");
        stream.write_all(&[0x42]).expect("barrier send failed");
        stream.flush().expect("barrier flush failed");
        let mut buf = [0u8; 1];
        stream.read_exact(&mut buf).expect("barrier recv failed");
    }
}

trait WcExt {
    fn zeroed() -> Self;
}

impl WcExt for IbvWc {
    fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}
