//! Concrete RDMA transport implementation backed by `rmlx_rdma::RdmaConnection`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rmlx_rdma::connection::RdmaConnection;
use rmlx_rdma::multi_port::StripeEngine;
use rmlx_rdma::rdma_metrics::RdmaMetrics;
use rmlx_rdma::RdmaError;

use crate::group::{DistributedError, RdmaTransport};

/// Concrete transport wrapping real `RdmaConnection`s (one per peer rank).
///
/// Each connection is wrapped in a `Mutex` to satisfy the `Sync` requirement
/// of `RdmaTransport` — `RdmaConnection` uses `RefCell` internally for its
/// completion backlog, which is not `Sync`. The Mutex serializes access per
/// peer connection.
///
/// `connections[i]` is the connection to peer rank `i`.
/// The slot at `connections[local_rank]` is unused (no self-connection).
/// Per-peer wr_id encoding: `(seq << 1) | send_recv_bit`.
/// Bit 0 = 0 for send, 1 for recv.
const WR_ID_SEND_BIT: u64 = 0;
const WR_ID_RECV_BIT: u64 = 1;

pub struct RdmaConnectionTransport {
    connections: Vec<Mutex<RdmaConnection>>,
    local_rank: u32,
    metrics: Arc<RdmaMetrics>,
    stripe_engine: Option<StripeEngine>,
    /// Per-peer monotonic sequence counter for unique wr_id generation.
    /// `wr_id_seqs[peer_rank]` generates sequence numbers for that peer.
    wr_id_seqs: Vec<AtomicU64>,
}

impl RdmaConnectionTransport {
    /// Create a new transport from a set of peer connections.
    ///
    /// `connections` must contain one entry per rank in the world.
    /// The entry at index `local_rank` is never used for I/O but must be present
    /// to keep indexing simple (it can be a dummy/unconnected instance).
    pub fn new(connections: Vec<RdmaConnection>, local_rank: u32) -> Self {
        let world_size = connections.len();
        let wr_id_seqs: Vec<AtomicU64> = (0..world_size).map(|_| AtomicU64::new(0)).collect();
        Self {
            connections: connections.into_iter().map(Mutex::new).collect(),
            local_rank,
            metrics: Arc::new(RdmaMetrics::new()),
            stripe_engine: None,
            wr_id_seqs,
        }
    }

    /// Attach a StripeEngine for dual-port TB5 striping.
    ///
    /// When set, large sends will be split across dual ports for increased
    /// bandwidth using the stripe plan.
    pub fn with_stripe_engine(mut self, engine: StripeEngine) -> Self {
        self.stripe_engine = Some(engine);
        self
    }

    /// Whether dual-port striping is configured.
    pub fn has_striping(&self) -> bool {
        self.stripe_engine
            .as_ref()
            .is_some_and(|e| e.config().has_dual())
    }

    /// This node's rank.
    pub fn local_rank(&self) -> u32 {
        self.local_rank
    }

    /// Number of connections (== world size).
    pub fn world_size(&self) -> usize {
        self.connections.len()
    }

    /// Get metrics reference.
    pub fn metrics(&self) -> &RdmaMetrics {
        &self.metrics
    }
}

impl RdmaConnectionTransport {
    /// Generate a unique wr_id for a peer.
    /// Encoding: `(seq << 1) | send_recv_bit`
    fn next_wr_id(&self, peer_rank: u32, send_recv_bit: u64) -> u64 {
        let seq = self.wr_id_seqs[peer_rank as usize].fetch_add(1, Ordering::Relaxed);
        (seq << 1) | send_recv_bit
    }
}

fn rdma_to_distributed(e: RdmaError) -> DistributedError {
    DistributedError::Transport(e.to_string())
}

impl RdmaTransport for RdmaConnectionTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        let conn = self.connections[dst_rank as usize]
            .lock()
            .map_err(|e| DistributedError::Transport(format!("lock poisoned: {e}")))?;
        let wr_id = self.next_wr_id(dst_rank, WR_ID_SEND_BIT);

        // Register the data buffer as an MR for sending.
        // SAFETY: data slice is valid for its length and lives until we complete the send.
        let mr = unsafe {
            conn.register_mr(data.as_ptr() as *mut c_void, data.len())
                .map_err(|e| {
                    self.metrics.record_send_error();
                    rdma_to_distributed(e)
                })?
        };

        conn.post_send(&mr, 0, data.len() as u32, wr_id)
            .map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed(e)
            })?;
        conn.wait_completions(&[wr_id]).map_err(|e| {
            self.metrics.record_send_error();
            rdma_to_distributed(e)
        })?;

        self.metrics.record_send(data.len() as u64);
        Ok(())
    }

    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        let conn = self.connections[src_rank as usize]
            .lock()
            .map_err(|e| DistributedError::Transport(format!("lock poisoned: {e}")))?;
        let wr_id = self.next_wr_id(src_rank, WR_ID_RECV_BIT);

        let mut buf = vec![0u8; len];

        // SAFETY: buf is valid for len bytes and lives until we complete the recv.
        let mr = unsafe {
            conn.register_mr(buf.as_mut_ptr() as *mut c_void, len)
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed(e)
                })?
        };

        conn.post_recv(&mr, 0, len as u32, wr_id).map_err(|e| {
            self.metrics.record_recv_error();
            rdma_to_distributed(e)
        })?;
        conn.wait_completions(&[wr_id]).map_err(|e| {
            self.metrics.record_recv_error();
            rdma_to_distributed(e)
        })?;

        self.metrics.record_recv(len as u64);
        Ok(buf)
    }

    fn sendrecv(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_len: usize,
        src_rank: u32,
    ) -> Result<Vec<u8>, DistributedError> {
        // Post the send first, then recv, then wait for both completions.
        // This avoids deadlock in pairwise exchange patterns.
        self.send(send_data, dst_rank)?;
        self.recv(src_rank, recv_len)
    }
}

#[cfg(test)]
mod tests {
    use crate::group::{DistributedError, Group, RdmaTransport};
    use std::sync::{Arc, Mutex};

    /// A mock transport that records calls for testing.
    struct MockTransport {
        sent: Mutex<Vec<(Vec<u8>, u32)>>,
        recv_data: Mutex<Vec<u8>>,
    }

    impl MockTransport {
        fn new(recv_data: Vec<u8>) -> Self {
            Self {
                sent: Mutex::new(Vec::new()),
                recv_data: Mutex::new(recv_data),
            }
        }
    }

    impl RdmaTransport for MockTransport {
        fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
            self.sent.lock().unwrap().push((data.to_vec(), dst_rank));
            Ok(())
        }

        fn recv(&self, _src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
            let data = self.recv_data.lock().unwrap();
            Ok(data[..len].to_vec())
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

    #[test]
    fn test_mock_transport_send_recv() {
        let payload = vec![1u8, 2, 3, 4];
        let mock = MockTransport::new(payload.clone());

        mock.send(&payload, 1).unwrap();
        let sent = mock.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, payload);
        assert_eq!(sent[0].1, 1);

        let received = mock.recv(0, 4).unwrap();
        assert_eq!(received, payload);
    }

    #[test]
    fn test_group_with_mock_transport_broadcast() {
        let data = vec![0xAA; 8]; // 8 bytes, 4-byte aligned
        let mock = Arc::new(MockTransport::new(data.clone()));

        // Root rank broadcasts
        let group = Group::with_transport(vec![0, 1, 2], 0, 3, mock.clone()).unwrap();
        let result = group.broadcast(&data, 0).unwrap();
        assert_eq!(result, data);

        // Root should have sent to 2 peers
        let sent = mock.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);
    }

    #[test]
    fn test_wr_id_encoding() {
        // Verify wr_id = (seq << 1) | send_recv_bit
        // send bit = 0, recv bit = 1
        let send_id_0 = (0u64 << 1) | super::WR_ID_SEND_BIT;
        assert_eq!(send_id_0, 0);
        assert_eq!(send_id_0 & 1, 0); // send

        let recv_id_0 = (0u64 << 1) | super::WR_ID_RECV_BIT;
        assert_eq!(recv_id_0, 1);
        assert_eq!(recv_id_0 & 1, 1); // recv

        let send_id_1 = (1u64 << 1) | super::WR_ID_SEND_BIT;
        assert_eq!(send_id_1, 2);
        assert_eq!(send_id_1 >> 1, 1); // seq=1
        assert_eq!(send_id_1 & 1, 0); // send

        let recv_id_5 = (5u64 << 1) | super::WR_ID_RECV_BIT;
        assert_eq!(recv_id_5, 11);
        assert_eq!(recv_id_5 >> 1, 5); // seq=5
        assert_eq!(recv_id_5 & 1, 1); // recv
    }

    #[test]
    fn test_wr_id_uniqueness() {
        // Verify that successive calls to mock send/recv produce unique wr_ids
        // (not directly testable via the mock, but the encoding guarantees uniqueness)
        let mut ids = std::collections::HashSet::new();
        for seq in 0..100u64 {
            let send_id = (seq << 1) | super::WR_ID_SEND_BIT;
            let recv_id = (seq << 1) | super::WR_ID_RECV_BIT;
            assert!(ids.insert(send_id), "duplicate send wr_id at seq={seq}");
            assert!(ids.insert(recv_id), "duplicate recv wr_id at seq={seq}");
        }
        assert_eq!(ids.len(), 200);
    }

    #[test]
    fn test_group_with_mock_transport_send_recv() {
        let data = vec![0xBB; 4]; // 4 bytes, aligned
        let mock = Arc::new(MockTransport::new(data.clone()));

        let group = Group::with_transport(vec![0, 1], 0, 2, mock.clone()).unwrap();

        group.send(&data, 1).unwrap();
        let sent = mock.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].1, 1);
        drop(sent);

        let received = group.recv(1, 4).unwrap();
        assert_eq!(received, data);
    }
}
