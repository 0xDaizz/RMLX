//! TCP-based QP info exchange for RDMA connection setup.
//!
//! Protocol (matching vllm-mlx PoC phase3_e2e_transfer.m):
//! - Rank 0 (server): listen → accept → recv remote QPInfo → send local QPInfo
//! - Rank 1+ (client): connect → send local QPInfo → recv remote QPInfo
//!
//! Also provides TCP barrier synchronization for coordinating post operations.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::qp::QpInfo;
use crate::RdmaError;

/// Default port for QP info exchange (matches PoC)
pub const TCP_EXCHANGE_PORT: u16 = 18515;
/// Default port for barrier sync
pub const TCP_SYNC_PORT: u16 = 18516;

/// Timeout in seconds for server accept() calls.
const ACCEPT_TIMEOUT_SECS: u64 = 60;
/// Number of retries for TCP read/write operations.
const IO_MAX_RETRIES: usize = 3;
/// Delay between IO retries in milliseconds.
const IO_RETRY_DELAY_MS: u64 = 1000;

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
/// Listens on `port`, accepts one connection with a timeout, exchanges QPInfo.
/// Server protocol: recv remote QPInfo first, then send local QPInfo.
/// Accept times out after `ACCEPT_TIMEOUT_SECS`. Read/write retry up to
/// `IO_MAX_RETRIES` times with `IO_RETRY_DELAY_MS` between attempts.
pub fn exchange_server(local: &QpInfo, port: u16) -> Result<QpInfo, RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("bind port {port}: {e}")))?;

    // Accept with timeout: set non-blocking and poll
    let mut stream = accept_with_timeout(&listener, ACCEPT_TIMEOUT_SECS)?;

    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Server protocol: recv first, then send (matches PoC)
    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    retry_read_exact(&mut stream, &mut recv_buf).map_err(|e| {
        RdmaError::ConnectionFailed(format!("recv QPInfo from {peer} after retries: {e}"))
    })?;

    let send_buf = serialize_qp_info(local);
    retry_write_all(&mut stream, &send_buf).map_err(|e| {
        RdmaError::ConnectionFailed(format!("send QPInfo to {peer} after retries: {e}"))
    })?;
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
    retry_write_all(&mut stream, &send_buf).map_err(|e| {
        RdmaError::ConnectionFailed(format!("send QPInfo to {addr} after retries: {e}"))
    })?;
    stream.flush().ok();

    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    retry_read_exact(&mut stream, &mut recv_buf).map_err(|e| {
        RdmaError::ConnectionFailed(format!("recv QPInfo from {addr} after retries: {e}"))
    })?;

    Ok(deserialize_qp_info(&recv_buf))
}

/// TCP barrier synchronization — server side (rank 0).
///
/// Both ranks wait for each other:
/// Rank 0: listen + accept + recv 1 byte + send 1 byte.
/// Accept times out after `ACCEPT_TIMEOUT_SECS`.
pub fn tcp_barrier_server(port: u16) -> Result<(), RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier bind port {port}: {e}")))?;

    let mut stream = accept_with_timeout(&listener, ACCEPT_TIMEOUT_SECS)?;

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

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Accept a connection with a deadline. Uses non-blocking mode and polls
/// with short sleeps until a connection arrives or timeout expires.
fn accept_with_timeout(listener: &TcpListener, timeout_secs: u64) -> Result<TcpStream, RdmaError> {
    listener
        .set_nonblocking(true)
        .map_err(|e| RdmaError::ConnectionFailed(format!("set_nonblocking: {e}")))?;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                // Restore blocking mode on the accepted stream
                stream
                    .set_nonblocking(false)
                    .map_err(|e| RdmaError::ConnectionFailed(format!("set_blocking: {e}")))?;
                return Ok(stream);
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(RdmaError::Timeout(format!(
                        "accept timed out after {timeout_secs}s"
                    )));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Err(RdmaError::ConnectionFailed(format!("accept: {e}")));
            }
        }
    }
}

/// Retry `read_exact` up to `IO_MAX_RETRIES` times with `IO_RETRY_DELAY_MS` delay.
fn retry_read_exact(stream: &mut TcpStream, buf: &mut [u8]) -> Result<(), io::Error> {
    let mut last_err = None;
    for attempt in 0..IO_MAX_RETRIES {
        match stream.read_exact(buf) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!(
                    "[rmlx-rdma] WARN: read_exact attempt {}/{} failed: {e}",
                    attempt + 1,
                    IO_MAX_RETRIES,
                );
                last_err = Some(e);
                if attempt + 1 < IO_MAX_RETRIES {
                    std::thread::sleep(Duration::from_millis(IO_RETRY_DELAY_MS));
                }
            }
        }
    }
    Err(last_err.unwrap())
}

/// Retry `write_all` up to `IO_MAX_RETRIES` times with `IO_RETRY_DELAY_MS` delay.
fn retry_write_all(stream: &mut TcpStream, buf: &[u8]) -> Result<(), io::Error> {
    let mut last_err = None;
    for attempt in 0..IO_MAX_RETRIES {
        match stream.write_all(buf) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!(
                    "[rmlx-rdma] WARN: write_all attempt {}/{} failed: {e}",
                    attempt + 1,
                    IO_MAX_RETRIES,
                );
                last_err = Some(e);
                if attempt + 1 < IO_MAX_RETRIES {
                    std::thread::sleep(Duration::from_millis(IO_RETRY_DELAY_MS));
                }
            }
        }
    }
    Err(last_err.unwrap())
}
