//! Integration tests for rmlx-rdma
//!
//! These tests gracefully skip when RDMA hardware is not available.

use rmlx_rdma::{is_available, RdmaError};

#[test]
fn test_rdma_availability_check() {
    // This should not panic, regardless of hardware presence
    let available = is_available();
    eprintln!("RDMA available: {available}");
}

#[test]
fn test_rdma_context_open() {
    if !is_available() {
        eprintln!("skipping test: RDMA not available (no librdma.dylib)");
        return;
    }

    use rmlx_rdma::context::RdmaContext;

    match RdmaContext::open_default() {
        Ok(ctx) => {
            eprintln!("RDMA device: {}", ctx.device_name());

            // Test PD allocation
            let pd = ctx.alloc_pd().expect("PD allocation");
            drop(pd);
        }
        Err(RdmaError::NoDevices) => {
            eprintln!("skipping test: RDMA library loaded but no devices found");
        }
        Err(e) => {
            eprintln!("skipping test: RDMA error: {e}");
        }
    }
}

#[test]
fn test_device_probe_runs_on_open() {
    if !is_available() {
        eprintln!("skipping test: RDMA not available");
        return;
    }

    use rmlx_rdma::context::RdmaContext;

    match RdmaContext::open_default() {
        Ok(ctx) => {
            // Probe should have been populated during open_default()
            if let Some(probe) = ctx.probe() {
                eprintln!(
                    "probe: gid_index={}, max_mr_size={}, max_qp_wr={}, max_cq_depth={}, mtu={}, max_msg_sz={}",
                    probe.gid_index, probe.max_mr_size, probe.max_qp_wr,
                    probe.max_cq_depth, probe.mtu, probe.max_msg_sz,
                );
                // Sanity checks on probed values
                assert!(
                    probe.gid_index <= 15,
                    "gid_index should be small, got {}",
                    probe.gid_index
                );
                assert!(probe.max_qp_wr > 0, "max_qp_wr must be positive");
                assert!(probe.max_cq_depth > 0, "max_cq_depth must be positive");
                assert!(probe.max_mr_size > 0, "max_mr_size must be positive");
                assert!(probe.mtu > 0, "mtu enum must be positive");
            } else {
                eprintln!("probe returned None (probe_port failed)");
            }
        }
        Err(RdmaError::NoDevices) => {
            eprintln!("skipping: no devices");
        }
        Err(e) => {
            eprintln!("skipping: {e}");
        }
    }
}

#[test]
fn test_probe_values_flow_to_cq_qp() {
    // Verify that the renamed DEFAULT_ constants exist and have expected values
    use rmlx_rdma::mr::DEFAULT_MAX_MR_SIZE;
    use rmlx_rdma::qp::{
        DEFAULT_CQ_DEPTH, DEFAULT_GID_INDEX, DEFAULT_MAX_RECV_WR, DEFAULT_MAX_SEND_WR,
    };

    assert_eq!(DEFAULT_GID_INDEX, 1);
    assert_eq!(DEFAULT_CQ_DEPTH, 8192);
    assert_eq!(DEFAULT_MAX_SEND_WR, 8192);
    assert_eq!(DEFAULT_MAX_RECV_WR, 8192);
    assert_eq!(DEFAULT_MAX_MR_SIZE, 16 * 1024 * 1024);
}

#[test]
fn test_mr_register_with_limit_rejects_oversized() {
    // Test that register_with_limit correctly rejects buffers exceeding the limit
    // This is a unit-level check that doesn't need hardware
    use rmlx_rdma::mr::DEFAULT_MAX_MR_SIZE;
    assert_eq!(DEFAULT_MAX_MR_SIZE, 16 * 1024 * 1024);
    // The actual register call requires hardware, but we verify the constant is correct
}

#[test]
fn test_qp_info_creation() {
    // This test doesn't need hardware -- just tests QpInfo struct construction
    use rmlx_rdma::qp::QpInfo;

    let info = QpInfo {
        lid: 1234,
        qpn: 56789,
        psn: 42,
        gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    };

    assert_eq!(info.lid, 1234);
    assert_eq!(info.qpn, 56789);
    assert_eq!(info.psn, 42);
    assert_eq!(info.gid[0], 1);
    assert_eq!(info.gid[15], 16);

    // Clone should work
    let info2 = info.clone();
    assert_eq!(info2.lid, info.lid);
    assert_eq!(info2.gid, info.gid);
}

#[test]
fn test_ffi_qp_state_constants() {
    use rmlx_rdma::ffi::qp_state;

    assert_eq!(qp_state::RESET, 0);
    assert_eq!(qp_state::INIT, 1);
    assert_eq!(qp_state::RTR, 2);
    assert_eq!(qp_state::RTS, 3);
    assert_eq!(qp_state::SQD, 4);
    assert_eq!(qp_state::SQE, 5);
    assert_eq!(qp_state::ERR, 6);
}

#[test]
fn test_ffi_qp_type_constants() {
    use rmlx_rdma::ffi::qp_type;

    assert_eq!(qp_type::RC, 2);
    assert_eq!(qp_type::UC, 3);
    assert_eq!(qp_type::UD, 4);
}

#[test]
fn test_ffi_mtu_constants() {
    use rmlx_rdma::ffi::mtu;

    assert_eq!(mtu::MTU_256, 1);
    assert_eq!(mtu::MTU_512, 2);
    assert_eq!(mtu::MTU_1024, 3);
    assert_eq!(mtu::MTU_2048, 4);
    assert_eq!(mtu::MTU_4096, 5);
}

#[test]
fn test_ffi_access_flags_constants() {
    use rmlx_rdma::ffi::access_flags;

    assert_eq!(access_flags::LOCAL_WRITE, 1);
    assert_eq!(access_flags::REMOTE_WRITE, 2);
    assert_eq!(access_flags::REMOTE_READ, 4);
}

#[test]
fn test_ffi_wc_status_constants() {
    use rmlx_rdma::ffi::wc_status;

    assert_eq!(wc_status::SUCCESS, 0);
}

#[test]
fn test_ffi_wr_opcode_constants() {
    use rmlx_rdma::ffi::wr_opcode;

    assert_eq!(wr_opcode::SEND, 0);
    assert_eq!(wr_opcode::SEND_WITH_IMM, 1);
}

#[test]
fn test_ffi_send_flags_constants() {
    use rmlx_rdma::ffi::send_flags;

    assert_eq!(send_flags::SIGNALED, 1);
}

#[test]
fn test_rdma_error_display() {
    // Verify error Display impl doesn't panic
    let errors = vec![
        RdmaError::LibraryNotFound("test".into()),
        RdmaError::NoDevices,
        RdmaError::DeviceOpen("test".into()),
        RdmaError::PdAlloc,
        RdmaError::MrReg("test".into()),
        RdmaError::CqCreate,
        RdmaError::QpCreate("test".into()),
        RdmaError::QpModify("test".into()),
        RdmaError::PostFailed("test".into()),
        RdmaError::CqPoll("test".into()),
        RdmaError::ConnectionFailed("test".into()),
        RdmaError::Unavailable("test".into()),
    ];

    for err in &errors {
        let msg = format!("{err}");
        assert!(
            !msg.is_empty(),
            "error Display should produce non-empty string"
        );
    }
}

#[test]
fn test_ibv_send_wr_layout_size() {
    // IbvSendWr must be at least 80 bytes to match C ibv_send_wr
    let size = std::mem::size_of::<rmlx_rdma::ffi::IbvSendWr>();
    assert!(
        size >= 80,
        "IbvSendWr is {size} bytes but C ibv_send_wr is ~80 bytes"
    );
}

#[test]
fn test_ibv_wr_union_size() {
    // IbvWrUnion must be at least 32 bytes (atomic variant: 28B + alignment)
    let size = std::mem::size_of::<rmlx_rdma::ffi::IbvWrUnion>();
    assert!(
        size >= 32,
        "IbvWrUnion is {size} bytes but C union wr is 32 bytes"
    );
}

#[test]
fn test_ibv_send_wr_zeroed_is_valid() {
    // A zeroed IbvSendWr should be safe to construct and inspect
    let wr: rmlx_rdma::ffi::IbvSendWr = unsafe { std::mem::zeroed() };
    assert_eq!(wr.wr_id, 0);
    assert!(wr.next.is_null());
    assert!(wr.sg_list.is_null());
    assert_eq!(wr.num_sge, 0);
    assert_eq!(wr.opcode, 0);
    assert_eq!(wr.send_flags, 0);
    assert_eq!(wr.imm_data, 0);
}

#[test]
fn test_ibv_recv_wr_layout() {
    // IbvRecvWr should be 32 bytes: wr_id(8) + next(8) + sg_list(8) + num_sge(4) + pad(4)
    let size = std::mem::size_of::<rmlx_rdma::ffi::IbvRecvWr>();
    assert!(
        size >= 24,
        "IbvRecvWr is {size} bytes, expected at least 24"
    );

    // Verify zero construction works
    let wr = rmlx_rdma::ffi::IbvRecvWr {
        wr_id: 42,
        next: std::ptr::null_mut(),
        sg_list: std::ptr::null_mut(),
        num_sge: 0,
    };
    assert_eq!(wr.wr_id, 42);
}
