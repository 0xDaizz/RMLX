//! Coordinator / SideChannel for N-way QP info exchange.
//!
//! Uses rank 0 as a hub to coordinate connection setup among N ranks.
//! The `all_gather` operation collects data from all ranks at rank 0,
//! then redistributes the gathered data to all ranks.
//!
//! Protocol:
//! - Rank 0 (hub): listens for connections from all other ranks,
//!   collects their data, assembles the full gather vector, and
//!   sends it back to each rank.
//! - Rank 1+ (spoke): connects to rank 0, sends local data,
//!   receives the full gather vector.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use tracing::warn;

use crate::qp::{QpInfo, QP_INFO_WIRE_SIZE};
use crate::RdmaError;

/// Default port for the coordinator side-channel.
pub const COORDINATOR_PORT: u16 = 18520;

/// Configuration for the coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// TCP port for the coordinator hub.
    pub port: u16,
    /// Timeout in seconds for accepting connections at rank 0.
    pub accept_timeout_secs: u64,
    /// TCP connect timeout in milliseconds for non-zero ranks.
    pub connect_timeout_ms: u64,
    /// Number of connection retries for non-zero ranks.
    pub connect_retries: u32,
    /// Delay between connection retries in milliseconds.
    pub retry_delay_ms: u64,
    /// I/O timeout in seconds for read/write operations.
    pub io_timeout_secs: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            port: COORDINATOR_PORT,
            accept_timeout_secs: 120,
            connect_timeout_ms: 5000,
            connect_retries: 30,
            retry_delay_ms: 500,
            io_timeout_secs: 60,
        }
    }
}

/// Perform an all-gather of QpInfo across N ranks using rank 0 as hub.
///
/// Each rank provides its own `QpInfo` (one per peer it wants to connect to,
/// or a single representative entry). Rank 0 collects all entries and
/// redistributes the full vector to every rank.
///
/// Returns a `Vec<QpInfo>` of length `world_size`, indexed by rank.
/// The entry at index `rank` is the QpInfo contributed by that rank.
pub fn all_gather_qp_info(
    rank: u32,
    world_size: u32,
    local_info: &QpInfo,
    hub_host: &str,
    cfg: &CoordinatorConfig,
) -> Result<Vec<QpInfo>, RdmaError> {
    if world_size < 2 {
        return Ok(vec![local_info.clone()]);
    }
    if rank == 0 {
        hub_all_gather(world_size, local_info, cfg)
    } else {
        spoke_all_gather(rank, local_info, hub_host, cfg)
    }
}

/// Rank 0 hub: accept connections from all other ranks, gather their QpInfo,
/// assemble the full vector, and send it back to each rank.
fn hub_all_gather(
    world_size: u32,
    local_info: &QpInfo,
    cfg: &CoordinatorConfig,
) -> Result<Vec<QpInfo>, RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], cfg.port).into();
    let listener = TcpListener::bind(addr).map_err(|e| {
        RdmaError::ConnectionFailed(format!("coordinator bind port {}: {e}", cfg.port))
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| RdmaError::ConnectionFailed(format!("coordinator set_nonblocking: {e}")))?;

    let n_peers = (world_size - 1) as usize;
    let mut streams: Vec<(u32, TcpStream)> = Vec::with_capacity(n_peers);
    let mut gathered: Vec<Option<QpInfo>> = vec![None; world_size as usize];
    gathered[0] = Some(local_info.clone());

    let deadline = Instant::now() + Duration::from_secs(cfg.accept_timeout_secs);
    let io_timeout = Duration::from_secs(cfg.io_timeout_secs);

    // Accept connections from all peers
    while streams.len() < n_peers {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nonblocking(false)
                    .map_err(|e| RdmaError::ConnectionFailed(format!("set_blocking: {e}")))?;
                stream.set_read_timeout(Some(io_timeout)).ok();
                stream.set_write_timeout(Some(io_timeout)).ok();
                stream.set_nodelay(true).ok();

                streams.push((0, stream)); // rank will be read below
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(RdmaError::Timeout(format!(
                        "coordinator hub: only accepted {}/{} peers before timeout",
                        streams.len(),
                        n_peers,
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(RdmaError::ConnectionFailed(format!(
                    "coordinator accept: {e}"
                )));
            }
        }
    }

    // Read rank + QpInfo from each peer
    for (ref mut peer_rank, ref mut stream) in &mut streams {
        // Read 4 bytes for rank
        let mut rank_buf = [0u8; 4];
        stream
            .read_exact(&mut rank_buf)
            .map_err(|e| RdmaError::ConnectionFailed(format!("coordinator read rank: {e}")))?;
        let r = u32::from_le_bytes(rank_buf);
        *peer_rank = r;

        // Read QpInfo
        let mut info_buf = [0u8; QP_INFO_WIRE_SIZE];
        stream.read_exact(&mut info_buf).map_err(|e| {
            RdmaError::ConnectionFailed(format!("coordinator read QpInfo from rank {r}: {e}"))
        })?;
        let info = QpInfo::from_wire(info_buf);

        if (r as usize) >= gathered.len() {
            return Err(RdmaError::InvalidArgument(format!(
                "coordinator: received rank {r} >= world_size {world_size}"
            )));
        }
        gathered[r as usize] = Some(info);
    }

    // Verify all ranks are present
    let result: Vec<QpInfo> = gathered
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or_else(|| {
                RdmaError::ConnectionFailed(format!("coordinator: missing QpInfo for rank {i}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Serialize the full gathered vector
    let full_payload = serialize_qp_info_vec(&result);

    // Send the full vector back to each peer
    for (_, ref mut stream) in &mut streams {
        stream
            .write_all(&full_payload)
            .map_err(|e| RdmaError::ConnectionFailed(format!("coordinator send gathered: {e}")))?;
        if let Err(e) = stream.flush() {
            warn!("TCP flush failed: {e}");
        }
    }

    Ok(result)
}

/// Non-zero rank spoke: connect to rank 0, send local QpInfo, receive full vector.
fn spoke_all_gather(
    rank: u32,
    local_info: &QpInfo,
    hub_host: &str,
    cfg: &CoordinatorConfig,
) -> Result<Vec<QpInfo>, RdmaError> {
    let addr = format!("{hub_host}:{}", cfg.port);
    let mut stream = connect_with_retries(
        &addr,
        cfg.connect_timeout_ms,
        cfg.connect_retries,
        cfg.retry_delay_ms,
    )?;

    let io_timeout = Duration::from_secs(cfg.io_timeout_secs);
    stream.set_read_timeout(Some(io_timeout)).ok();
    stream.set_write_timeout(Some(io_timeout)).ok();
    stream.set_nodelay(true).ok();

    // Send rank
    stream
        .write_all(&rank.to_le_bytes())
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} send rank: {e}")))?;

    // Send QpInfo
    let info_buf = local_info.to_wire();
    stream
        .write_all(&info_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} send QpInfo: {e}")))?;
    stream.flush().ok();

    // Read 4 bytes for the count of QpInfo entries
    let mut count_buf = [0u8; 4];
    stream
        .read_exact(&mut count_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} read count: {e}")))?;
    let count = u32::from_le_bytes(count_buf) as usize;

    // Read all QpInfo entries
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut buf = [0u8; QP_INFO_WIRE_SIZE];
        stream.read_exact(&mut buf).map_err(|e| {
            RdmaError::ConnectionFailed(format!("spoke rank {rank} read QpInfo[{i}]: {e}"))
        })?;
        result.push(QpInfo::from_wire(buf));
    }

    Ok(result)
}

/// Serialize a vector of QpInfo: 4-byte LE count + N * QP_INFO_WIRE_SIZE bytes.
fn serialize_qp_info_vec(infos: &[QpInfo]) -> Vec<u8> {
    let count = infos.len() as u32;
    let mut buf = Vec::with_capacity(4 + infos.len() * QP_INFO_WIRE_SIZE);
    buf.extend_from_slice(&count.to_le_bytes());
    for info in infos {
        buf.extend_from_slice(&info.to_wire());
    }
    buf
}

/// Generic all_gather for arbitrary serializable data using rank-0 as hub.
///
/// Each rank provides a `local_data` byte slice. Rank 0 collects all entries
/// and redistributes them. All entries must have the same `item_size`.
///
/// Returns a `Vec<Vec<u8>>` of length `world_size`, indexed by rank.
pub fn all_gather_bytes(
    rank: u32,
    world_size: u32,
    local_data: &[u8],
    hub_host: &str,
    cfg: &CoordinatorConfig,
) -> Result<Vec<Vec<u8>>, RdmaError> {
    if world_size < 2 {
        return Ok(vec![local_data.to_vec()]);
    }
    if rank == 0 {
        hub_all_gather_bytes(world_size, local_data, cfg)
    } else {
        spoke_all_gather_bytes(rank, local_data, hub_host, cfg)
    }
}

/// Rank 0 hub: gather byte slices from all ranks.
fn hub_all_gather_bytes(
    world_size: u32,
    local_data: &[u8],
    cfg: &CoordinatorConfig,
) -> Result<Vec<Vec<u8>>, RdmaError> {
    let addr: SocketAddr = ([0, 0, 0, 0], cfg.port).into();
    let listener = TcpListener::bind(addr).map_err(|e| {
        RdmaError::ConnectionFailed(format!("coordinator bind port {}: {e}", cfg.port))
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| RdmaError::ConnectionFailed(format!("coordinator set_nonblocking: {e}")))?;

    let n_peers = (world_size - 1) as usize;
    let mut streams: Vec<(u32, TcpStream)> = Vec::with_capacity(n_peers);
    let mut gathered: Vec<Option<Vec<u8>>> = vec![None; world_size as usize];
    gathered[0] = Some(local_data.to_vec());

    let deadline = Instant::now() + Duration::from_secs(cfg.accept_timeout_secs);
    let io_timeout = Duration::from_secs(cfg.io_timeout_secs);

    while streams.len() < n_peers {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nonblocking(false)
                    .map_err(|e| RdmaError::ConnectionFailed(format!("set_blocking: {e}")))?;
                stream.set_read_timeout(Some(io_timeout)).ok();
                stream.set_write_timeout(Some(io_timeout)).ok();
                stream.set_nodelay(true).ok();
                streams.push((0, stream));
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(RdmaError::Timeout(format!(
                        "coordinator hub: only accepted {}/{} peers before timeout",
                        streams.len(),
                        n_peers,
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(RdmaError::ConnectionFailed(format!(
                    "coordinator accept: {e}"
                )));
            }
        }
    }

    // Read rank + data from each peer
    for (ref mut peer_rank, ref mut stream) in &mut streams {
        // Read rank (4 bytes)
        let mut rank_buf = [0u8; 4];
        stream
            .read_exact(&mut rank_buf)
            .map_err(|e| RdmaError::ConnectionFailed(format!("coordinator read rank: {e}")))?;
        let r = u32::from_le_bytes(rank_buf);
        *peer_rank = r;

        // Read data length (4 bytes)
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| {
            RdmaError::ConnectionFailed(format!("coordinator read len from rank {r}: {e}"))
        })?;
        let data_len = u32::from_le_bytes(len_buf) as usize;

        // Read data
        let mut data = vec![0u8; data_len];
        stream.read_exact(&mut data).map_err(|e| {
            RdmaError::ConnectionFailed(format!("coordinator read data from rank {r}: {e}"))
        })?;

        if (r as usize) >= gathered.len() {
            return Err(RdmaError::InvalidArgument(format!(
                "coordinator: received rank {r} >= world_size {world_size}"
            )));
        }
        gathered[r as usize] = Some(data);
    }

    let result: Vec<Vec<u8>> = gathered
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or_else(|| {
                RdmaError::ConnectionFailed(format!("coordinator: missing data for rank {i}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Serialize and send full gathered data back to each peer
    let full_payload = serialize_gathered_bytes(&result);
    for (_, ref mut stream) in &mut streams {
        stream.write_all(&full_payload).map_err(|e| {
            RdmaError::ConnectionFailed(format!("coordinator send gathered bytes: {e}"))
        })?;
        if let Err(e) = stream.flush() {
            warn!("TCP flush failed: {e}");
        }
    }

    Ok(result)
}

/// Non-zero rank spoke: send local data, receive full vector.
fn spoke_all_gather_bytes(
    rank: u32,
    local_data: &[u8],
    hub_host: &str,
    cfg: &CoordinatorConfig,
) -> Result<Vec<Vec<u8>>, RdmaError> {
    let addr = format!("{hub_host}:{}", cfg.port);
    let mut stream = connect_with_retries(
        &addr,
        cfg.connect_timeout_ms,
        cfg.connect_retries,
        cfg.retry_delay_ms,
    )?;

    let io_timeout = Duration::from_secs(cfg.io_timeout_secs);
    stream.set_read_timeout(Some(io_timeout)).ok();
    stream.set_write_timeout(Some(io_timeout)).ok();
    stream.set_nodelay(true).ok();

    // Send rank
    stream
        .write_all(&rank.to_le_bytes())
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} send rank: {e}")))?;

    // Send data length + data
    let len = local_data.len() as u32;
    stream
        .write_all(&len.to_le_bytes())
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} send len: {e}")))?;
    stream
        .write_all(local_data)
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} send data: {e}")))?;
    stream.flush().ok();

    // Read count of entries
    let mut count_buf = [0u8; 4];
    stream
        .read_exact(&mut count_buf)
        .map_err(|e| RdmaError::ConnectionFailed(format!("spoke rank {rank} read count: {e}")))?;
    let count = u32::from_le_bytes(count_buf) as usize;

    // Read each entry: 4 bytes len + data
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| {
            RdmaError::ConnectionFailed(format!("spoke rank {rank} read len[{i}]: {e}"))
        })?;
        let entry_len = u32::from_le_bytes(len_buf) as usize;

        let mut data = vec![0u8; entry_len];
        stream.read_exact(&mut data).map_err(|e| {
            RdmaError::ConnectionFailed(format!("spoke rank {rank} read data[{i}]: {e}"))
        })?;
        result.push(data);
    }

    Ok(result)
}

/// Serialize gathered byte vectors: count(4) + [len(4) + data]...
fn serialize_gathered_bytes(entries: &[Vec<u8>]) -> Vec<u8> {
    let total_size: usize = 4 + entries.iter().map(|e| 4 + e.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(total_size);
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        buf.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        buf.extend_from_slice(entry);
    }
    buf
}

/// TCP connect with retries.
fn connect_with_retries(
    addr: &str,
    timeout_ms: u64,
    max_retries: u32,
    retry_delay_ms: u64,
) -> Result<TcpStream, RdmaError> {
    let sock_addr: SocketAddr = addr
        .parse()
        .map_err(|e| RdmaError::ConnectionFailed(format!("invalid address {addr}: {e}")))?;
    let timeout = Duration::from_millis(timeout_ms);

    let mut last_err = None;
    for attempt in 0..max_retries {
        match TcpStream::connect_timeout(&sock_addr, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < max_retries {
                    std::thread::sleep(Duration::from_millis(retry_delay_ms));
                }
            }
        }
    }
    Err(RdmaError::ConnectionFailed(format!(
        "connect to {addr} after {max_retries} attempts: {}",
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown".into()),
    )))
}

/// Perform a TCP-based barrier across all N ranks using rank 0 as hub.
///
/// All ranks block until every rank has entered the barrier.
pub fn barrier(
    rank: u32,
    world_size: u32,
    hub_host: &str,
    cfg: &CoordinatorConfig,
) -> Result<(), RdmaError> {
    let dummy = [0u8; 1];
    let _ = all_gather_bytes(rank, world_size, &dummy, hub_host, cfg)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_serialize_qp_info_roundtrip() {
        let info = QpInfo {
            lid: 0x1234,
            qpn: 0xABCD,
            psn: 0x5678,
            gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };
        let buf = info.to_wire();
        let out = QpInfo::from_wire(buf);
        assert_eq!(out.lid, info.lid);
        assert_eq!(out.qpn, info.qpn);
        assert_eq!(out.psn, info.psn);
        assert_eq!(out.gid, info.gid);
    }

    #[test]
    fn test_serialize_qp_info_vec_roundtrip() {
        let infos = vec![
            QpInfo {
                lid: 1,
                qpn: 10,
                psn: 100,
                gid: [1; 16],
            },
            QpInfo {
                lid: 2,
                qpn: 20,
                psn: 200,
                gid: [2; 16],
            },
            QpInfo {
                lid: 3,
                qpn: 30,
                psn: 300,
                gid: [3; 16],
            },
        ];
        let payload = serialize_qp_info_vec(&infos);
        // Verify layout: 4 bytes count + 3 * 26 bytes
        assert_eq!(payload.len(), 4 + 3 * QP_INFO_WIRE_SIZE);
        let count = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        assert_eq!(count, 3);
    }

    #[test]
    fn test_serialize_gathered_bytes_roundtrip() {
        let entries = vec![vec![1u8, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let payload = serialize_gathered_bytes(&entries);
        // Verify: 4 (count) + (4+3) + (4+2) + (4+4) = 4 + 7 + 6 + 8 = 25
        assert_eq!(payload.len(), 25);
    }

    #[test]
    fn test_all_gather_single_rank() {
        let info = QpInfo {
            lid: 1,
            qpn: 2,
            psn: 3,
            gid: [0; 16],
        };
        let cfg = CoordinatorConfig::default();
        let result = all_gather_qp_info(0, 1, &info, "", &cfg).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lid, 1);
    }

    #[test]
    fn test_all_gather_bytes_single_rank() {
        let data = vec![10, 20, 30];
        let cfg = CoordinatorConfig::default();
        let result = all_gather_bytes(0, 1, &data, "", &cfg).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_all_gather_qp_info_two_ranks() {
        let cfg0 = CoordinatorConfig {
            port: 0, // will be replaced after bind
            accept_timeout_secs: 5,
            connect_timeout_ms: 2000,
            connect_retries: 10,
            retry_delay_ms: 100,
            io_timeout_secs: 5,
        };

        let info0 = QpInfo {
            lid: 10,
            qpn: 100,
            psn: 1000,
            gid: [0xA; 16],
        };
        let info1 = QpInfo {
            lid: 20,
            qpn: 200,
            psn: 2000,
            gid: [0xB; 16],
        };

        // Bind to port 0 to get an ephemeral port, then communicate it
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let cfg_clone = CoordinatorConfig {
            port,
            ..cfg0.clone()
        };
        let cfg_spoke = CoordinatorConfig { port, ..cfg0 };

        let info0_clone = info0.clone();
        let hub_thread =
            thread::spawn(move || all_gather_qp_info(0, 2, &info0_clone, "", &cfg_clone));

        // Small delay to let hub bind
        std::thread::sleep(Duration::from_millis(50));

        let spoke_result = all_gather_qp_info(1, 2, &info1, "127.0.0.1", &cfg_spoke);

        let hub_result = hub_thread.join().unwrap();

        let hub_vec = hub_result.unwrap();
        let spoke_vec = spoke_result.unwrap();

        assert_eq!(hub_vec.len(), 2);
        assert_eq!(spoke_vec.len(), 2);

        // Both should see the same data
        assert_eq!(hub_vec[0].lid, info0.lid);
        assert_eq!(hub_vec[1].lid, info1.lid);
        assert_eq!(spoke_vec[0].lid, info0.lid);
        assert_eq!(spoke_vec[1].lid, info1.lid);
    }

    #[test]
    fn test_all_gather_bytes_three_ranks() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let cfg = CoordinatorConfig {
            port,
            accept_timeout_secs: 5,
            connect_timeout_ms: 2000,
            connect_retries: 10,
            retry_delay_ms: 100,
            io_timeout_secs: 5,
        };

        let data0 = vec![10u8, 11, 12];
        let data1 = vec![20u8, 21];
        let data2 = vec![30u8, 31, 32, 33];

        let cfg0 = cfg.clone();
        let d0 = data0.clone();
        let hub = thread::spawn(move || all_gather_bytes(0, 3, &d0, "", &cfg0));

        std::thread::sleep(Duration::from_millis(50));

        let cfg1 = cfg.clone();
        let d1 = data1.clone();
        let spoke1 = thread::spawn(move || all_gather_bytes(1, 3, &d1, "127.0.0.1", &cfg1));

        let cfg2 = cfg;
        let d2 = data2.clone();
        let spoke2 = thread::spawn(move || all_gather_bytes(2, 3, &d2, "127.0.0.1", &cfg2));

        let r0 = hub.join().unwrap().unwrap();
        let r1 = spoke1.join().unwrap().unwrap();
        let r2 = spoke2.join().unwrap().unwrap();

        // All ranks should see the same gathered data
        for r in [&r0, &r1, &r2] {
            assert_eq!(r.len(), 3);
            assert_eq!(r[0], data0);
            assert_eq!(r[1], data1);
            assert_eq!(r[2], data2);
        }
    }

    #[test]
    fn test_barrier_two_ranks() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let cfg = CoordinatorConfig {
            port,
            accept_timeout_secs: 5,
            connect_timeout_ms: 2000,
            connect_retries: 10,
            retry_delay_ms: 100,
            io_timeout_secs: 5,
        };

        let cfg0 = cfg.clone();
        let hub = thread::spawn(move || barrier(0, 2, "", &cfg0));

        std::thread::sleep(Duration::from_millis(50));

        let spoke_result = barrier(1, 2, "127.0.0.1", &cfg);
        let hub_result = hub.join().unwrap();

        assert!(hub_result.is_ok());
        assert!(spoke_result.is_ok());
    }
}
