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
