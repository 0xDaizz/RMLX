//! Concrete RDMA transport implementation backed by `rmlx_rdma::RdmaConnection`.

use std::ffi::c_void;
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
pub struct RdmaConnectionTransport {
    connections: Vec<Mutex<RdmaConnection>>,
    local_rank: u32,
    metrics: Arc<RdmaMetrics>,
    stripe_engine: Option<StripeEngine>,
}

impl RdmaConnectionTransport {
    /// Create a new transport from a set of peer connections.
    ///
    /// `connections` must contain one entry per rank in the world.
    /// The entry at index `local_rank` is never used for I/O but must be present
    /// to keep indexing simple (it can be a dummy/unconnected instance).
    pub fn new(connections: Vec<RdmaConnection>, local_rank: u32) -> Self {
        Self {
            connections: connections.into_iter().map(Mutex::new).collect(),
            local_rank,
            metrics: Arc::new(RdmaMetrics::new()),
            stripe_engine: None,
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
            .map_or(false, |e| e.config().has_dual())
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

fn rdma_to_distributed(e: RdmaError) -> DistributedError {
    DistributedError::Transport(e.to_string())
}

impl RdmaTransport for RdmaConnectionTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        let conn = self.connections[dst_rank as usize]
            .lock()
            .map_err(|e| DistributedError::Transport(format!("lock poisoned: {e}")))?;
        let wr_id: u64 = 0x5E_0000_0000 | dst_rank as u64;

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
        let wr_id: u64 = 0xEC_0000_0000 | src_rank as u64;

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
        let group = Group::with_transport(vec![0, 1, 2], 0, 3, mock.clone());
        let result = group.broadcast(&data, 0).unwrap();
        assert_eq!(result, data);

        // Root should have sent to 2 peers
        let sent = mock.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);
    }

    #[test]
    fn test_group_with_mock_transport_send_recv() {
        let data = vec![0xBB; 4]; // 4 bytes, aligned
        let mock = Arc::new(MockTransport::new(data.clone()));

        let group = Group::with_transport(vec![0, 1], 0, 2, mock.clone());

        group.send(&data, 1).unwrap();
        let sent = mock.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].1, 1);
        drop(sent);

        let received = group.recv(1, 4).unwrap();
        assert_eq!(received, data);
    }
}
