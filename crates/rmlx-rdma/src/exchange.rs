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

/// Configuration for TCP exchange timeout/retry behavior.
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    /// Timeout in seconds for server accept() calls (default: 60).
    pub accept_timeout_secs: u64,
    /// Number of retries for TCP reconnect operations (default: 3).
    pub io_max_retries: u32,
    /// Delay between IO retries in milliseconds (default: 1000).
    pub io_retry_delay_ms: u64,
    /// TCP connect timeout in milliseconds (default: 5000).
    pub connect_timeout_ms: u64,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            accept_timeout_secs: 60,
            io_max_retries: 3,
            io_retry_delay_ms: 1000,
            connect_timeout_ms: 5000,
        }
    }
}

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
/// On I/O error, drops the stream and re-accepts a new connection before retrying.
pub fn exchange_server(
    local: &QpInfo,
    port: u16,
    cfg: &ExchangeConfig,
) -> Result<QpInfo, RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("bind port {port}: {e}")))?;

    let max_retries = cfg.io_max_retries as usize;
    let mut last_err = None;

    for attempt in 0..max_retries {
        // Accept with timeout
        let mut stream = match accept_with_timeout(&listener, cfg.accept_timeout_secs) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "rmlx_rdma",
                    attempt = attempt + 1,
                    max_retries,
                    %e,
                    "server accept failed",
                );
                last_err = Some(e);
                if attempt + 1 < max_retries {
                    std::thread::sleep(Duration::from_millis(cfg.io_retry_delay_ms));
                }
                continue;
            }
        };

        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Server protocol: recv first, then send (matches PoC)
        match server_exchange_on_stream(&mut stream, local) {
            Ok(remote) => return Ok(remote),
            Err(e) => {
                tracing::warn!(
                    target: "rmlx_rdma",
                    %peer,
                    attempt = attempt + 1,
                    max_retries,
                    %e,
                    "server exchange failed, dropping stream and re-accepting",
                );
                last_err = Some(RdmaError::ConnectionFailed(format!(
                    "exchange with {peer}: {e}"
                )));
                // stream is dropped here, freeing the corrupted connection
                if attempt + 1 < max_retries {
                    std::thread::sleep(Duration::from_millis(cfg.io_retry_delay_ms));
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        RdmaError::ConnectionFailed("server exchange failed: no attempts made".into())
    }))
}

/// Perform the server-side exchange on an already-connected stream.
fn server_exchange_on_stream(stream: &mut TcpStream, local: &QpInfo) -> Result<QpInfo, io::Error> {
    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    stream.read_exact(&mut recv_buf)?;

    let send_buf = serialize_qp_info(local);
    stream.write_all(&send_buf)?;
    stream.flush()?;

    Ok(deserialize_qp_info(&recv_buf))
}

/// Exchange QP info as rank 1+ (client).
///
/// Connects to `host:port`, exchanges QPInfo.
/// Client protocol: send local QPInfo first, then recv remote QPInfo.
/// On I/O error, drops the stream and reconnects before retrying.
pub fn exchange_client(
    local: &QpInfo,
    host: &str,
    port: u16,
    cfg: &ExchangeConfig,
) -> Result<QpInfo, RdmaError> {
    let addr = format!("{host}:{port}");
    let max_retries = cfg.io_max_retries as usize;
    let mut last_err = None;

    for attempt in 0..max_retries {
        // Connect (with sub-retries for the TCP handshake)
        let mut stream = match connect_with_retries(&addr, cfg.connect_timeout_ms) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "rmlx_rdma",
                    attempt = attempt + 1,
                    max_retries,
                    %addr,
                    %e,
                    "client connect failed",
                );
                last_err = Some(e);
                if attempt + 1 < max_retries {
                    std::thread::sleep(Duration::from_millis(cfg.io_retry_delay_ms));
                }
                continue;
            }
        };

        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

        // Client protocol: send first, then recv (matches PoC)
        match client_exchange_on_stream(&mut stream, local) {
            Ok(remote) => return Ok(remote),
            Err(e) => {
                tracing::warn!(
                    target: "rmlx_rdma",
                    %addr,
                    attempt = attempt + 1,
                    max_retries,
                    %e,
                    "client exchange failed, dropping stream and reconnecting",
                );
                last_err = Some(RdmaError::ConnectionFailed(format!(
                    "exchange with {addr}: {e}"
                )));
                // stream is dropped here, freeing the corrupted connection
                if attempt + 1 < max_retries {
                    std::thread::sleep(Duration::from_millis(cfg.io_retry_delay_ms));
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        RdmaError::ConnectionFailed(format!(
            "client exchange to {addr} failed: no attempts made"
        ))
    }))
}

/// Perform the client-side exchange on an already-connected stream.
fn client_exchange_on_stream(stream: &mut TcpStream, local: &QpInfo) -> Result<QpInfo, io::Error> {
    let send_buf = serialize_qp_info(local);
    stream.write_all(&send_buf)?;
    stream.flush()?;

    let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
    stream.read_exact(&mut recv_buf)?;

    Ok(deserialize_qp_info(&recv_buf))
}

/// Connect to `addr` with up to 30 sub-retries (500ms apart).
///
/// Uses `TcpStream::connect_timeout` with the configured timeout per attempt.
fn connect_with_retries(addr: &str, connect_timeout_ms: u64) -> Result<TcpStream, RdmaError> {
    let sock_addr: SocketAddr = addr
        .parse()
        .map_err(|e| RdmaError::ConnectionFailed(format!("invalid address {addr}: {e}")))?;
    let timeout = Duration::from_millis(connect_timeout_ms);
    let mut last_err = None;
    for sub_attempt in 0..30 {
        match TcpStream::connect_timeout(&sock_addr, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                if sub_attempt + 1 < 30 {
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }
    }
    Err(RdmaError::ConnectionFailed(format!(
        "connect to {addr} after 30 sub-attempts: {}",
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown".into()),
    )))
}

/// TCP barrier synchronization — server side (rank 0).
///
/// Both ranks wait for each other:
/// Rank 0: listen + accept + recv 1 byte + send 1 byte.
/// Accept times out after `accept_timeout_secs`. Read/write operations
/// time out after `io_timeout_secs` to prevent indefinite blocking if the
/// peer hangs after connecting.
pub fn tcp_barrier_server(
    port: u16,
    accept_timeout_secs: u64,
    io_timeout_secs: u64,
) -> Result<(), RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier bind port {port}: {e}")))?;

    let mut stream = accept_with_timeout(&listener, accept_timeout_secs)?;

    stream
        .set_read_timeout(Some(Duration::from_secs(io_timeout_secs)))
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(io_timeout_secs)))
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier set_write_timeout: {e}")))?;

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
/// Retries connection up to `connect_retries` times with `retry_delay_ms`
/// intervals. Read/write operations time out after `io_timeout_secs`.
pub fn tcp_barrier_client(
    host: &str,
    port: u16,
    io_timeout_secs: u64,
    connect_retries: u32,
    retry_delay_ms: u64,
) -> Result<(), RdmaError> {
    let addr = format!("{host}:{port}");

    let mut stream = None;
    for attempt in 0..connect_retries {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt + 1 == connect_retries {
                    return Err(RdmaError::ConnectionFailed(format!(
                        "barrier connect to {addr} after {connect_retries} attempts: {e}"
                    )));
                }
                std::thread::sleep(Duration::from_millis(retry_delay_ms));
            }
        }
    }
    let mut stream = stream.ok_or_else(|| {
        RdmaError::ConnectionFailed(format!(
            "barrier connect to {addr}: no stream after retry loop"
        ))
    })?;

    stream
        .set_read_timeout(Some(Duration::from_secs(io_timeout_secs)))
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(io_timeout_secs)))
        .map_err(|e| RdmaError::ConnectionFailed(format!("barrier set_write_timeout: {e}")))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn test_qp_info_roundtrip() {
        let info = QpInfo {
            lid: 0x1234,
            qpn: 0xABCD,
            psn: 0x5678,
            gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };
        let buf = serialize_qp_info(&info);
        let out = deserialize_qp_info(&buf);
        assert_eq!(out.lid, info.lid);
        assert_eq!(out.qpn, info.qpn);
        assert_eq!(out.psn, info.psn);
        assert_eq!(out.gid, info.gid);
    }

    #[test]
    fn test_exchange_reconnect_on_partial_io() {
        // Simulate a server that accepts connections.
        // First connection: write partial data (< QP_INFO_WIRE_SIZE) then close.
        // Second connection: write full valid QPInfo.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();

        let client_info = QpInfo {
            lid: 100,
            qpn: 200,
            psn: 300,
            gid: [10; 16],
        };
        let server_info = QpInfo {
            lid: 400,
            qpn: 500,
            psn: 600,
            gid: [20; 16],
        };

        let server_info_clone = server_info.clone();
        let server_thread = thread::spawn(move || {
            // First accept: send partial data to simulate corruption, then drop
            let (mut stream1, _) = listener.accept().expect("accept 1");
            // Write only 5 bytes (partial QPInfo), then drop stream
            stream1.write_all(&[0u8; 5]).ok();
            drop(stream1);

            // Second accept: do proper exchange
            let (mut stream2, _) = listener.accept().expect("accept 2");
            // Read client's QPInfo
            let mut recv_buf = [0u8; QP_INFO_WIRE_SIZE];
            stream2.read_exact(&mut recv_buf).expect("server read");
            let received = deserialize_qp_info(&recv_buf);
            // Send server's QPInfo
            let send_buf = serialize_qp_info(&server_info_clone);
            stream2.write_all(&send_buf).expect("server write");
            stream2.flush().ok();

            received
        });

        // Client side: exchange_client should reconnect after first failure
        let cfg = ExchangeConfig::default();
        let result = exchange_client(&client_info, "127.0.0.1", port, &cfg);
        assert!(
            result.is_ok(),
            "exchange_client should succeed: {:?}",
            result.err()
        );
        let remote = result.expect("checked above");
        assert_eq!(remote.lid, server_info.lid);
        assert_eq!(remote.qpn, server_info.qpn);
        assert_eq!(remote.psn, server_info.psn);
        assert_eq!(remote.gid, server_info.gid);

        // Verify server received correct client info
        let received = server_thread.join().expect("server thread");
        assert_eq!(received.lid, client_info.lid);
        assert_eq!(received.qpn, client_info.qpn);
    }

    #[test]
    fn test_exchange_fails_after_all_retries_exhausted() {
        let cfg = ExchangeConfig::default();
        // Server that always sends partial data and closes
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();

        let max_retries = cfg.io_max_retries;
        let server_thread = thread::spawn(move || {
            // Accept io_max_retries connections, all sending partial data
            for _ in 0..max_retries {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.write_all(&[0u8; 3]).ok();
                    drop(stream);
                }
            }
        });

        let client_info = QpInfo {
            lid: 1,
            qpn: 2,
            psn: 3,
            gid: [0; 16],
        };
        let result = exchange_client(&client_info, "127.0.0.1", port, &cfg);
        assert!(result.is_err(), "should fail after all retries exhausted");

        server_thread.join().ok();
    }

    #[test]
    fn test_exchange_config_default_values() {
        let cfg = ExchangeConfig::default();
        assert_eq!(cfg.accept_timeout_secs, 60);
        assert_eq!(cfg.io_max_retries, 3);
        assert_eq!(cfg.io_retry_delay_ms, 1000);
        assert_eq!(cfg.connect_timeout_ms, 5000);
    }

    #[test]
    fn test_exchange_with_custom_config() {
        // Use custom config with 1 retry so the client fails fast
        let cfg = ExchangeConfig {
            accept_timeout_secs: 2,
            io_max_retries: 1,
            io_retry_delay_ms: 100,
            connect_timeout_ms: 1000,
        };

        // Server that sends partial data and closes — with only 1 retry, should fail
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();

        let server_thread = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.write_all(&[0u8; 3]).ok();
                drop(stream);
            }
        });

        let client_info = QpInfo {
            lid: 1,
            qpn: 2,
            psn: 3,
            gid: [0; 16],
        };
        let result = exchange_client(&client_info, "127.0.0.1", port, &cfg);
        assert!(
            result.is_err(),
            "should fail with custom config (1 retry only)"
        );

        server_thread.join().ok();
    }
}
