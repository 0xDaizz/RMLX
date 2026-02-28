//! TCP-based QP info exchange for RDMA connection setup.
//!
//! Protocol (matching vllm-mlx PoC phase3_e2e_transfer.m):
//! - Rank 0 (server): listen → accept → recv remote QPInfo → send local QPInfo
//! - Rank 1+ (client): connect → send local QPInfo → recv remote QPInfo
//!
//! Also provides TCP barrier synchronization for coordinating post operations.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

use crate::qp::QpInfo;
use crate::RdmaError;

/// Default port for QP info exchange (matches PoC)
pub const TCP_EXCHANGE_PORT: u16 = 18515;
/// Default port for barrier sync
pub const TCP_SYNC_PORT: u16 = 18516;

/// Serialized QPInfo wire size: lid(2) + qpn(4) + psn(4) + gid(16) = 26 bytes.
/// Fixed size avoids padding/alignment issues across platforms.
const QP_INFO_WIRE_SIZE: usize = 2 + 4 + 4 + 16;

/// Serialize QpInfo to a fixed 26-byte little-endian wire format.
fn serialize_qp_info(info: &QpInfo) -> [u8; QP_INFO_WIRE_SIZE] {
    let mut buf = [0u8; QP_INFO_WIRE_SIZE];
    buf[0..2].copy_from_slice(&info.lid.to_le_bytes());
    buf[2..6].copy_from_slice(&info.qpn.to_le_bytes());
    buf[6..10].copy_from_slice(&info.psn.to_le_bytes());
    buf[10..26].copy_from_slice(&info.gid);
    buf
}

/// Deserialize QpInfo from the 26-byte wire format.
fn deserialize_qp_info(buf: &[u8; QP_INFO_WIRE_SIZE]) -> QpInfo {
    let lid = u16::from_le_bytes([buf[0], buf[1]]);
    let qpn = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
    let psn = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
    let mut gid = [0u8; 16];
    gid.copy_from_slice(&buf[10..26]);
    QpInfo { lid, qpn, psn, gid }
}

/// Exchange QP info as rank 0 (server/listener).
///
/// Listens on `port`, accepts one connection, exchanges QPInfo.
/// Server protocol: recv remote QPInfo first, then send local QPInfo.
pub fn exchange_server(local: &QpInfo, port: u16) -> Result<QpInfo, RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("bind port {port}: {e}")))?;

    let (mut stream, peer) = listener
        .accept()
        .map_err(|e| RdmaError::ConnectionFailed(format!("accept: {e}")))?;

    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    // Server protocol: recv first, then send (matches PoC)
    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    stream
        .read_exact(&mut recv_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("recv QPInfo from {peer}: {e}")))?;

    let send_buf = serialize_qp_info(local);
    stream
        .write_all(&send_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("send QPInfo to {peer}: {e}")))?;
    stream.flush().ok();

    Ok(deserialize_qp_info(&recv_buf))
}

/// Exchange QP info as rank 1+ (client).
///
/// Connects to `host:port`, exchanges QPInfo.
/// Client protocol: send local QPInfo first, then recv remote QPInfo.
/// Retries connection up to 30 times with 500ms intervals.
pub fn exchange_client(local: &QpInfo, host: &str, port: u16) -> Result<QpInfo, RdmaError> {
    let addr = format!("{host}:{port}");

    // Retry connection up to 30 times, 500ms apart (matches PoC)
    let mut stream = None;
    for attempt in 0..30 {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt == 29 {
                    return Err(RdmaError::ConnectionFailed(format!(
                        "connect to {addr} after 30 attempts: {e}"
                    )));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    // unwrap is safe: loop either sets stream or returns Err on attempt 29
    let mut stream = stream.unwrap();

    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    // Client protocol: send first, then recv (matches PoC)
    let send_buf = serialize_qp_info(local);
    stream
        .write_all(&send_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("send QPInfo to {addr}: {e}")))?;
    stream.flush().ok();

    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    stream
        .read_exact(&mut recv_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("recv QPInfo from {addr}: {e}")))?;

    Ok(deserialize_qp_info(&recv_buf))
}

/// TCP barrier synchronization — server side (rank 0).
///
/// Both ranks wait for each other:
/// Rank 0: listen + accept + recv 1 byte + send 1 byte.
pub fn tcp_barrier_server(port: u16) -> Result<(), RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier bind port {port}: {e}")))?;

    let (mut stream, _) = listener
        .accept()
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier accept: {e}")))?;

    let mut buf = [0u8; 1];
    stream
        .read_exact(&mut buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier recv: {e}")))?;
    stream
        .write_all(&[1u8])
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier send: {e}")))?;
    stream.flush().ok();
    Ok(())
}

/// TCP barrier synchronization — client side (rank 1+).
///
/// Both ranks wait for each other:
/// Rank 1+: connect + send 1 byte + recv 1 byte.
/// Retries connection up to 30 times with 100ms intervals.
pub fn tcp_barrier_client(host: &str, port: u16) -> Result<(), RdmaError> {
    let addr = format!("{host}:{port}");

    let mut stream = None;
    for attempt in 0..30 {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt == 29 {
                    return Err(RdmaError::ConnectionFailed(format!(
                        "barrier connect to {addr} after 30 attempts: {e}"
                    )));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    // unwrap is safe: loop either sets stream or returns Err on attempt 29
    let mut stream = stream.unwrap();

    stream
        .write_all(&[1u8])
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier send: {e}")))?;
    stream.flush().ok();

    let mut buf = [0u8; 1];
    stream
        .read_exact(&mut buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier recv: {e}")))?;
    Ok(())
}
