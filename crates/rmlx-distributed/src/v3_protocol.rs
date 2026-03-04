//! Variable-length v3 wire protocol for MoE token exchange.
//!
//! The v2 (current) wire format uses fixed-size packets: `4 + token_stride` bytes
//! per token (expert_id prefix + token data), which wastes bandwidth when experts
//! have varying load.
//!
//! The v3 format uses variable-length packets with a 4-byte meta header per token:
//!
//! ```text
//! Dispatch wire data per peer:
//!   [count: u32] [meta0: u32] [token0_data: token_stride bytes] [meta1: u32] [token1_data] ...
//!
//! Total payload = 4 + count * (4 + token_stride) bytes.
//! When count=0, only the 4-byte count header is sent.
//! ```
//!
//! This eliminates wasted bandwidth for peers that receive few or no tokens from
//! a given expert, while maintaining 16B alignment for Metal vectorized access.

use std::fmt;

use crate::group::{DistributedError, Group};

// ─── Error type ───

/// Protocol-level errors for v3 wire format operations.
#[derive(Debug)]
pub enum ProtocolError {
    /// Wire data too short or truncated.
    Truncated(String),
    /// Invalid expert ID or position in packet.
    InvalidField(String),
    /// Transport-level error.
    Transport(String),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(msg) => write!(f, "v3 protocol truncated: {msg}"),
            Self::InvalidField(msg) => write!(f, "v3 protocol invalid field: {msg}"),
            Self::Transport(msg) => write!(f, "v3 protocol transport error: {msg}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<DistributedError> for ProtocolError {
    fn from(e: DistributedError) -> Self {
        ProtocolError::Transport(e.to_string())
    }
}

// ─── Packet meta header ───

/// V3 packet meta header: packed into 4 bytes (u32).
///
/// Bits [31:16] = local_expert_id (u16)
/// Bits [15:0]  = original_position (u16, token position in the batch)
///
/// 16B aligned for Metal vectorized access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketMeta(pub u32);

impl PacketMeta {
    /// Create a new packet meta header.
    pub fn new(local_expert: u16, position: u16) -> Self {
        Self(((local_expert as u32) << 16) | (position as u32))
    }

    /// Extract the local expert ID from the packed header.
    pub fn local_expert(&self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// Extract the original batch position from the packed header.
    pub fn position(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

// ─── Dispatch types ───

/// A packed v3 dispatch packet ready for wire transfer.
#[derive(Debug, Clone)]
pub struct V3DispatchPacket {
    /// Peer rank this packet is destined for.
    pub dst_rank: u32,
    /// Number of tokens in this packet.
    pub token_count: u32,
    /// Wire data: `[count: u32][meta+token_data]*`
    pub wire_data: Vec<u8>,
}

/// Result of unpacking received v3 dispatch packets.
#[derive(Debug, Clone)]
pub struct V3ReceivedTokens {
    /// Unpacked tokens per local expert: expert_id -> Vec<(original_position, token_data)>
    pub tokens_by_expert: Vec<Vec<(u16, Vec<u8>)>>,
    /// Total tokens received.
    pub total_received: usize,
}

// ─── Combine types ───

/// A packed v3 combine request.
///
/// 8 bytes per request: `[token_slot: u32] [expert_and_pos: u32]`
/// where `expert_and_pos = (expert_id << 16) | position`.
#[derive(Debug, Clone)]
pub struct V3CombineRequest {
    /// Peer rank this request is destined for.
    pub dst_rank: u32,
    /// Requests as (token_slot, expert_and_pos) pairs.
    pub requests: Vec<(u32, u32)>,
    /// Wire data: `[count: u32][(token_slot, expert_and_pos): 8 bytes]*`
    pub wire_data: Vec<u8>,
}

/// Result of a v3 combine response.
#[derive(Debug, Clone)]
pub struct V3CombineResponse {
    /// Source rank this response came from.
    pub src_rank: u32,
    /// Response token data, ordered by request.
    pub token_data: Vec<Vec<u8>>,
}

// ─── Dispatch pack/unpack ───

/// Pack tokens for dispatch to a specific peer using v3 variable-length format.
///
/// `tokens`: slice of `(local_expert_id, original_position, token_bytes)`
/// `dst_rank`: destination peer rank
///
/// Returns a [`V3DispatchPacket`] with wire-ready data.
///
/// Wire layout: `[count: u32] [meta0: u32] [token0_data] [meta1: u32] [token1_data] ...`
pub fn pack_dispatch_v3(tokens: &[(u16, u16, &[u8])], dst_rank: u32) -> V3DispatchPacket {
    let count = tokens.len() as u32;
    let token_stride = if tokens.is_empty() {
        0
    } else {
        tokens[0].2.len()
    };

    // Total wire size: 4 (count) + count * (4 (meta) + token_stride)
    let wire_size = 4 + (count as usize) * (4 + token_stride);
    let mut wire_data = Vec::with_capacity(wire_size);

    // Write count header
    wire_data.extend_from_slice(&count.to_le_bytes());

    // Write each token: meta header + token data
    for &(expert_id, position, token_bytes) in tokens {
        let meta = PacketMeta::new(expert_id, position);
        wire_data.extend_from_slice(&meta.0.to_le_bytes());
        wire_data.extend_from_slice(token_bytes);
    }

    V3DispatchPacket {
        dst_rank,
        token_count: count,
        wire_data,
    }
}

/// Unpack received v3 dispatch wire data.
///
/// `wire_data`: raw bytes from the wire
/// `token_stride`: expected byte size of each token's data
/// `num_local_experts`: number of experts on this rank
///
/// Returns unpacked tokens grouped by local expert.
pub fn unpack_dispatch_v3(
    wire_data: &[u8],
    token_stride: usize,
    num_local_experts: usize,
) -> Result<V3ReceivedTokens, ProtocolError> {
    // Need at least 4 bytes for the count header
    if wire_data.len() < 4 {
        return Err(ProtocolError::Truncated(format!(
            "dispatch wire data too short for count header: {} bytes (need >= 4)",
            wire_data.len()
        )));
    }

    let count =
        u32::from_le_bytes([wire_data[0], wire_data[1], wire_data[2], wire_data[3]]) as usize;

    // Verify total wire length
    let record_size = 4 + token_stride; // meta + token data
    let expected_len = 4 + count * record_size;
    if wire_data.len() < expected_len {
        return Err(ProtocolError::Truncated(format!(
            "dispatch wire data truncated: {} bytes, expected {} (count={}, token_stride={})",
            wire_data.len(),
            expected_len,
            count,
            token_stride
        )));
    }

    let mut tokens_by_expert: Vec<Vec<(u16, Vec<u8>)>> = vec![vec![]; num_local_experts];
    let mut offset = 4; // skip count header

    for i in 0..count {
        // Read meta header
        if offset + 4 > wire_data.len() {
            return Err(ProtocolError::Truncated(format!(
                "dispatch wire data truncated at token {i} meta: offset={offset}, len={}",
                wire_data.len()
            )));
        }
        let meta_bits = u32::from_le_bytes([
            wire_data[offset],
            wire_data[offset + 1],
            wire_data[offset + 2],
            wire_data[offset + 3],
        ]);
        let meta = PacketMeta(meta_bits);
        offset += 4;

        // Read token data
        if offset + token_stride > wire_data.len() {
            return Err(ProtocolError::Truncated(format!(
                "dispatch wire data truncated at token {i} data: offset={offset}, \
                 need {token_stride} bytes, len={}",
                wire_data.len()
            )));
        }
        let token_data = wire_data[offset..offset + token_stride].to_vec();
        offset += token_stride;

        let expert_id = meta.local_expert() as usize;
        if expert_id >= num_local_experts {
            return Err(ProtocolError::InvalidField(format!(
                "token {i} has expert_id={expert_id} but num_local_experts={num_local_experts}"
            )));
        }

        tokens_by_expert[expert_id].push((meta.position(), token_data));
    }

    Ok(V3ReceivedTokens {
        tokens_by_expert,
        total_received: count,
    })
}

// ─── Combine pack/unpack ───

/// Pack combine requests for a specific peer.
///
/// `requests`: slice of `(token_slot, expert_id, position)`
/// `dst_rank`: destination peer rank
///
/// Wire layout: `[count: u32] [(token_slot: u32, expert_and_pos: u32)]*`
pub fn pack_combine_request_v3(requests: &[(u32, u16, u16)], dst_rank: u32) -> V3CombineRequest {
    let count = requests.len() as u32;

    // Wire size: 4 (count) + count * 8 (token_slot + expert_and_pos)
    let wire_size = 4 + (count as usize) * 8;
    let mut wire_data = Vec::with_capacity(wire_size);

    // Write count header
    wire_data.extend_from_slice(&count.to_le_bytes());

    let mut packed_requests = Vec::with_capacity(requests.len());
    for &(token_slot, expert_id, position) in requests {
        let expert_and_pos = ((expert_id as u32) << 16) | (position as u32);
        wire_data.extend_from_slice(&token_slot.to_le_bytes());
        wire_data.extend_from_slice(&expert_and_pos.to_le_bytes());
        packed_requests.push((token_slot, expert_and_pos));
    }

    V3CombineRequest {
        dst_rank,
        requests: packed_requests,
        wire_data,
    }
}

/// Unpack combine requests from wire data.
///
/// Returns a vector of `(token_slot, expert_id, position)` tuples.
///
/// Wire layout: `[count: u32] [(token_slot: u32, expert_and_pos: u32)]*`
pub fn unpack_combine_request_v3(wire_data: &[u8]) -> Result<Vec<(u32, u16, u16)>, ProtocolError> {
    if wire_data.len() < 4 {
        return Err(ProtocolError::Truncated(format!(
            "combine request wire data too short for count header: {} bytes (need >= 4)",
            wire_data.len()
        )));
    }

    let count =
        u32::from_le_bytes([wire_data[0], wire_data[1], wire_data[2], wire_data[3]]) as usize;

    let expected_len = 4 + count * 8;
    if wire_data.len() < expected_len {
        return Err(ProtocolError::Truncated(format!(
            "combine request wire data truncated: {} bytes, expected {} (count={})",
            wire_data.len(),
            expected_len,
            count
        )));
    }

    let mut results = Vec::with_capacity(count);
    let mut offset = 4;

    for i in 0..count {
        if offset + 8 > wire_data.len() {
            return Err(ProtocolError::Truncated(format!(
                "combine request truncated at entry {i}: offset={offset}, len={}",
                wire_data.len()
            )));
        }

        let token_slot = u32::from_le_bytes([
            wire_data[offset],
            wire_data[offset + 1],
            wire_data[offset + 2],
            wire_data[offset + 3],
        ]);
        offset += 4;

        let expert_and_pos = u32::from_le_bytes([
            wire_data[offset],
            wire_data[offset + 1],
            wire_data[offset + 2],
            wire_data[offset + 3],
        ]);
        offset += 4;

        let expert_id = (expert_and_pos >> 16) as u16;
        let position = (expert_and_pos & 0xFFFF) as u16;
        results.push((token_slot, expert_id, position));
    }

    Ok(results)
}

/// Pack combine response (token data for requested slots).
///
/// Wire layout: `[count: u32] [token0_data] [token1_data] ...`
pub fn pack_combine_response_v3(token_data: &[&[u8]]) -> Vec<u8> {
    let count = token_data.len() as u32;
    let token_stride = if token_data.is_empty() {
        0
    } else {
        token_data[0].len()
    };

    let wire_size = 4 + (count as usize) * token_stride;
    let mut wire = Vec::with_capacity(wire_size);

    wire.extend_from_slice(&count.to_le_bytes());
    for data in token_data {
        wire.extend_from_slice(data);
    }

    wire
}

/// Unpack combine response.
///
/// `wire_data`: raw bytes from the wire
/// `token_stride`: expected byte size of each token's data
/// `expected_count`: expected number of tokens in the response
///
/// Returns a vector of token data buffers, ordered by request.
pub fn unpack_combine_response_v3(
    wire_data: &[u8],
    token_stride: usize,
    expected_count: usize,
) -> Result<Vec<Vec<u8>>, ProtocolError> {
    if wire_data.len() < 4 {
        return Err(ProtocolError::Truncated(format!(
            "combine response wire data too short for count header: {} bytes (need >= 4)",
            wire_data.len()
        )));
    }

    let count =
        u32::from_le_bytes([wire_data[0], wire_data[1], wire_data[2], wire_data[3]]) as usize;

    if count != expected_count {
        return Err(ProtocolError::InvalidField(format!(
            "combine response count mismatch: wire says {count}, expected {expected_count}"
        )));
    }

    let expected_len = 4 + count * token_stride;
    if wire_data.len() < expected_len {
        return Err(ProtocolError::Truncated(format!(
            "combine response wire data truncated: {} bytes, expected {} \
             (count={}, token_stride={})",
            wire_data.len(),
            expected_len,
            count,
            token_stride
        )));
    }

    let mut results = Vec::with_capacity(count);
    let mut offset = 4;
    for _ in 0..count {
        results.push(wire_data[offset..offset + token_stride].to_vec());
        offset += token_stride;
    }

    Ok(results)
}

// ─── Two-phase variable-length exchange ───

/// Perform a two-phase variable-length exchange between peers.
///
/// Phase 1: Exchange counts (4 bytes each direction) via sendrecv
/// Phase 2: Exchange payloads (variable-length) via sendrecv
///
/// This is the building block for dispatch_v3 and combine_v3. For each peer
/// in the group, we send our packet and receive theirs. Peers not present in
/// `send_packets` receive a zero-count header (4 bytes).
///
/// Returns a vector of `(src_rank, wire_data)` for each peer that sent data.
pub fn blocking_exchange_v3(
    group: &Group,
    send_packets: &[V3DispatchPacket],
) -> Result<Vec<(u32, Vec<u8>)>, ProtocolError> {
    let peers = group.peers();
    if peers.is_empty() {
        return Ok(vec![]);
    }

    // Build a lookup from dst_rank -> packet index for O(1) access
    let mut packet_by_rank = std::collections::HashMap::with_capacity(send_packets.len());
    for (i, pkt) in send_packets.iter().enumerate() {
        packet_by_rank.insert(pkt.dst_rank, i);
    }

    // Zero-count header for peers we have nothing to send to
    let empty_count: [u8; 4] = 0u32.to_le_bytes();

    let mut received = Vec::with_capacity(peers.len());

    for &peer_rank in &peers {
        // ── Phase 1: exchange counts ──
        let send_count_bytes = if let Some(&idx) = packet_by_rank.get(&peer_rank) {
            send_packets[idx].token_count.to_le_bytes()
        } else {
            empty_count
        };

        let recv_count_bytes = group.sendrecv(&send_count_bytes, peer_rank, 4, peer_rank)?;

        let recv_count = u32::from_le_bytes([
            recv_count_bytes[0],
            recv_count_bytes[1],
            recv_count_bytes[2],
            recv_count_bytes[3],
        ]);

        // ── Phase 2: exchange payloads ──
        // Determine what to send: full wire_data (which includes the count header)
        // or just the empty count header.
        let send_payload = if let Some(&idx) = packet_by_rank.get(&peer_rank) {
            &send_packets[idx].wire_data[..]
        } else {
            &empty_count[..]
        };

        if recv_count == 0 {
            // Peer has nothing to send. We still need to send our payload.
            // Use send for our side, peer will recv.
            if send_payload.len() > 4 {
                // We have real data to send but peer has nothing to recv payload-wise.
                // Send the full wire data; peer receives only the count header.
                group.send(send_payload, peer_rank)?;
            }
            // Store empty result for this peer
            let mut empty_wire = Vec::with_capacity(4);
            empty_wire.extend_from_slice(&0u32.to_le_bytes());
            received.push((peer_rank, empty_wire));
        } else {
            // Both sides may have data. Use sendrecv.
            // The receiver expects: 4 (count) + recv_count * record_size bytes.
            // But we don't know token_stride here, so we compute recv payload
            // size from the send_payload structure.
            //
            // Actually, for the exchange we already know recv_count from phase 1.
            // We need to receive the full wire_data from the peer.
            // The peer's wire_data length is their full packet.
            // We need to know token_stride to compute the expected recv length.
            //
            // Since we can't know token_stride generically, we use a simpler
            // approach: exchange the full payload size first, then exchange payloads.

            // Exchange payload sizes
            let send_size = send_payload.len() as u32;
            let recv_size_bytes =
                group.sendrecv(&send_size.to_le_bytes(), peer_rank, 4, peer_rank)?;

            let recv_size = u32::from_le_bytes([
                recv_size_bytes[0],
                recv_size_bytes[1],
                recv_size_bytes[2],
                recv_size_bytes[3],
            ]) as usize;

            // Exchange actual payloads
            if recv_size > 0 {
                let recv_payload = group.sendrecv(send_payload, peer_rank, recv_size, peer_rank)?;
                received.push((peer_rank, recv_payload));
            } else {
                // Peer promised tokens but payload size is 0 — shouldn't happen
                // but handle gracefully.
                group.send(send_payload, peer_rank)?;
                let mut empty_wire = Vec::with_capacity(4);
                empty_wire.extend_from_slice(&0u32.to_le_bytes());
                received.push((peer_rank, empty_wire));
            }
        }
    }

    Ok(received)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PacketMeta tests ───

    #[test]
    fn test_packet_meta_roundtrip() {
        let meta = PacketMeta::new(42, 1023);
        assert_eq!(meta.local_expert(), 42);
        assert_eq!(meta.position(), 1023);
    }

    #[test]
    fn test_packet_meta_zero() {
        let meta = PacketMeta::new(0, 0);
        assert_eq!(meta.local_expert(), 0);
        assert_eq!(meta.position(), 0);
        assert_eq!(meta.0, 0);
    }

    #[test]
    fn test_packet_meta_max_values() {
        let meta = PacketMeta::new(u16::MAX, u16::MAX);
        assert_eq!(meta.local_expert(), u16::MAX);
        assert_eq!(meta.position(), u16::MAX);
        assert_eq!(meta.0, u32::MAX);
    }

    #[test]
    fn test_packet_meta_bit_layout() {
        // expert_id=1 should be in bits [31:16], position=2 in bits [15:0]
        let meta = PacketMeta::new(1, 2);
        assert_eq!(meta.0, (1u32 << 16) | 2);
    }

    // ─── Dispatch pack/unpack tests ───

    #[test]
    fn test_dispatch_pack_unpack_roundtrip() {
        let token0 = vec![0xAA; 16];
        let token1 = vec![0xBB; 16];
        let token2 = vec![0xCC; 16];

        let tokens: Vec<(u16, u16, &[u8])> =
            vec![(0, 10, &token0), (1, 20, &token1), (0, 30, &token2)];

        let packet = pack_dispatch_v3(&tokens, 1);
        assert_eq!(packet.dst_rank, 1);
        assert_eq!(packet.token_count, 3);
        // wire_data = 4 (count) + 3 * (4 + 16) = 64 bytes
        assert_eq!(packet.wire_data.len(), 4 + 3 * (4 + 16));

        let result = unpack_dispatch_v3(&packet.wire_data, 16, 2).unwrap();
        assert_eq!(result.total_received, 3);

        // Expert 0 should have tokens at positions 10 and 30
        assert_eq!(result.tokens_by_expert[0].len(), 2);
        assert_eq!(result.tokens_by_expert[0][0].0, 10);
        assert_eq!(result.tokens_by_expert[0][0].1, token0);
        assert_eq!(result.tokens_by_expert[0][1].0, 30);
        assert_eq!(result.tokens_by_expert[0][1].1, token2);

        // Expert 1 should have token at position 20
        assert_eq!(result.tokens_by_expert[1].len(), 1);
        assert_eq!(result.tokens_by_expert[1][0].0, 20);
        assert_eq!(result.tokens_by_expert[1][0].1, token1);
    }

    #[test]
    fn test_dispatch_empty_packet() {
        let tokens: Vec<(u16, u16, &[u8])> = vec![];
        let packet = pack_dispatch_v3(&tokens, 3);
        assert_eq!(packet.dst_rank, 3);
        assert_eq!(packet.token_count, 0);
        assert_eq!(packet.wire_data.len(), 4); // just the count header

        let result = unpack_dispatch_v3(&packet.wire_data, 32, 4).unwrap();
        assert_eq!(result.total_received, 0);
        assert_eq!(result.tokens_by_expert.len(), 4);
        for expert_tokens in &result.tokens_by_expert {
            assert!(expert_tokens.is_empty());
        }
    }

    #[test]
    fn test_dispatch_single_token() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens: Vec<(u16, u16, &[u8])> = vec![(0, 42, &data)];
        let packet = pack_dispatch_v3(&tokens, 0);

        assert_eq!(packet.token_count, 1);
        assert_eq!(packet.wire_data.len(), 4 + (4 + 8));

        let result = unpack_dispatch_v3(&packet.wire_data, 8, 1).unwrap();
        assert_eq!(result.total_received, 1);
        assert_eq!(result.tokens_by_expert[0].len(), 1);
        assert_eq!(result.tokens_by_expert[0][0].0, 42);
        assert_eq!(result.tokens_by_expert[0][0].1, data);
    }

    #[test]
    fn test_dispatch_unpack_truncated_count() {
        // Wire data too short for count header
        let wire = vec![0u8; 3];
        let err = unpack_dispatch_v3(&wire, 16, 2).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("too short for count header"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    #[test]
    fn test_dispatch_unpack_truncated_data() {
        // Pack 2 tokens with stride 8, then truncate the wire data
        let data = vec![0xFF; 8];
        let tokens: Vec<(u16, u16, &[u8])> = vec![(0, 0, &data), (0, 1, &data)];
        let packet = pack_dispatch_v3(&tokens, 0);

        // Truncate: remove last 4 bytes so second token is incomplete
        let truncated = &packet.wire_data[..packet.wire_data.len() - 4];
        let err = unpack_dispatch_v3(truncated, 8, 1).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("truncated"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    #[test]
    fn test_dispatch_unpack_invalid_expert_id() {
        // Pack a token with expert_id=5, but unpack with num_local_experts=2
        let data = vec![0x11; 8];
        let tokens: Vec<(u16, u16, &[u8])> = vec![(5, 0, &data)];
        let packet = pack_dispatch_v3(&tokens, 0);

        let err = unpack_dispatch_v3(&packet.wire_data, 8, 2).unwrap_err();
        match err {
            ProtocolError::InvalidField(msg) => {
                assert!(msg.contains("expert_id=5"), "got: {msg}");
                assert!(msg.contains("num_local_experts=2"), "got: {msg}");
            }
            other => panic!("expected InvalidField, got: {other:?}"),
        }
    }

    #[test]
    fn test_dispatch_wire_format_layout() {
        // Verify exact byte layout of the wire format
        let data = [0x01, 0x02, 0x03, 0x04];
        let tokens: Vec<(u16, u16, &[u8])> = vec![(3, 7, &data)];
        let packet = pack_dispatch_v3(&tokens, 0);

        // Count: 1 (little-endian u32)
        assert_eq!(&packet.wire_data[0..4], &1u32.to_le_bytes());

        // Meta: expert=3, position=7 -> (3 << 16) | 7
        let expected_meta = PacketMeta::new(3, 7);
        assert_eq!(&packet.wire_data[4..8], &expected_meta.0.to_le_bytes());

        // Token data
        assert_eq!(&packet.wire_data[8..12], &data);
    }

    // ─── Combine request pack/unpack tests ───

    #[test]
    fn test_combine_request_roundtrip() {
        let requests: Vec<(u32, u16, u16)> = vec![(100, 2, 50), (200, 3, 60), (300, 0, 70)];

        let packed = pack_combine_request_v3(&requests, 1);
        assert_eq!(packed.dst_rank, 1);
        assert_eq!(packed.requests.len(), 3);
        // wire_data = 4 (count) + 3 * 8 = 28 bytes
        assert_eq!(packed.wire_data.len(), 4 + 3 * 8);

        let unpacked = unpack_combine_request_v3(&packed.wire_data).unwrap();
        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0], (100, 2, 50));
        assert_eq!(unpacked[1], (200, 3, 60));
        assert_eq!(unpacked[2], (300, 0, 70));
    }

    #[test]
    fn test_combine_request_empty() {
        let requests: Vec<(u32, u16, u16)> = vec![];
        let packed = pack_combine_request_v3(&requests, 5);
        assert_eq!(packed.wire_data.len(), 4);

        let unpacked = unpack_combine_request_v3(&packed.wire_data).unwrap();
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_combine_request_truncated() {
        let wire = vec![0u8; 2]; // too short for count header
        let err = unpack_combine_request_v3(&wire).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("too short"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    #[test]
    fn test_combine_request_truncated_payload() {
        // Count says 2 entries but only enough bytes for 1
        let requests: Vec<(u32, u16, u16)> = vec![(10, 1, 2), (20, 3, 4)];
        let packed = pack_combine_request_v3(&requests, 0);

        // Truncate: remove last entry (8 bytes)
        let truncated = &packed.wire_data[..packed.wire_data.len() - 4];
        let err = unpack_combine_request_v3(truncated).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("truncated"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    #[test]
    fn test_combine_request_expert_and_pos_encoding() {
        // Verify the packed (expert_id << 16 | position) encoding
        let requests: Vec<(u32, u16, u16)> = vec![(0, 0xFF, 0xAB)];
        let packed = pack_combine_request_v3(&requests, 0);

        let expected_expert_and_pos = (0xFFu32 << 16) | 0xABu32;
        assert_eq!(packed.requests[0], (0, expected_expert_and_pos));
    }

    // ─── Combine response pack/unpack tests ───

    #[test]
    fn test_combine_response_roundtrip() {
        let t0 = vec![0xAA; 32];
        let t1 = vec![0xBB; 32];
        let t2 = vec![0xCC; 32];
        let data: Vec<&[u8]> = vec![&t0, &t1, &t2];

        let wire = pack_combine_response_v3(&data);
        assert_eq!(wire.len(), 4 + 3 * 32);

        let unpacked = unpack_combine_response_v3(&wire, 32, 3).unwrap();
        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0], t0);
        assert_eq!(unpacked[1], t1);
        assert_eq!(unpacked[2], t2);
    }

    #[test]
    fn test_combine_response_empty() {
        let data: Vec<&[u8]> = vec![];
        let wire = pack_combine_response_v3(&data);
        assert_eq!(wire.len(), 4);

        let unpacked = unpack_combine_response_v3(&wire, 16, 0).unwrap();
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_combine_response_count_mismatch() {
        let t0 = vec![0x11; 8];
        let data: Vec<&[u8]> = vec![&t0];
        let wire = pack_combine_response_v3(&data);

        // Expect 5 tokens but wire contains 1
        let err = unpack_combine_response_v3(&wire, 8, 5).unwrap_err();
        match err {
            ProtocolError::InvalidField(msg) => {
                assert!(msg.contains("count mismatch"), "got: {msg}");
            }
            other => panic!("expected InvalidField, got: {other:?}"),
        }
    }

    #[test]
    fn test_combine_response_truncated() {
        let wire = vec![0u8; 1]; // too short
        let err = unpack_combine_response_v3(&wire, 8, 1).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("too short"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    #[test]
    fn test_combine_response_truncated_payload() {
        // Pack 2 tokens then truncate
        let t0 = vec![0xAA; 16];
        let t1 = vec![0xBB; 16];
        let data: Vec<&[u8]> = vec![&t0, &t1];
        let wire = pack_combine_response_v3(&data);

        let truncated = &wire[..wire.len() - 8];
        let err = unpack_combine_response_v3(truncated, 16, 2).unwrap_err();
        match err {
            ProtocolError::Truncated(msg) => {
                assert!(msg.contains("truncated"), "got: {msg}");
            }
            other => panic!("expected Truncated, got: {other:?}"),
        }
    }

    // ─── Error conversion tests ───

    #[test]
    fn test_distributed_error_to_protocol_error() {
        let dist_err = DistributedError::Transport("connection refused".to_string());
        let proto_err: ProtocolError = dist_err.into();
        match proto_err {
            ProtocolError::Transport(msg) => {
                assert!(msg.contains("connection refused"), "got: {msg}");
            }
            other => panic!("expected Transport, got: {other:?}"),
        }
    }

    #[test]
    fn test_protocol_error_display() {
        let err = ProtocolError::Truncated("missing 4 bytes".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("truncated"), "got: {msg}");
        assert!(msg.contains("missing 4 bytes"), "got: {msg}");
    }

    // ─── Large token stride tests ───

    #[test]
    fn test_dispatch_large_tokens() {
        // Simulate hidden_dim=4096, f32 -> 16384 bytes per token
        let token_stride = 4096 * 4;
        let token_data: Vec<u8> = (0..token_stride).map(|i| (i % 256) as u8).collect();

        let tokens: Vec<(u16, u16, &[u8])> = vec![(0, 0, &token_data), (1, 1, &token_data)];

        let packet = pack_dispatch_v3(&tokens, 2);
        assert_eq!(packet.token_count, 2);
        assert_eq!(packet.wire_data.len(), 4 + 2 * (4 + token_stride));

        let result = unpack_dispatch_v3(&packet.wire_data, token_stride, 2).unwrap();
        assert_eq!(result.total_received, 2);
        assert_eq!(result.tokens_by_expert[0][0].1, token_data);
        assert_eq!(result.tokens_by_expert[1][0].1, token_data);
    }

    // ─── Multi-token same expert tests ───

    #[test]
    fn test_dispatch_many_tokens_same_expert() {
        let token_stride = 8;
        let num_tokens = 100;
        let mut tokens = Vec::with_capacity(num_tokens);
        let mut token_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_tokens);

        for i in 0..num_tokens {
            let data: Vec<u8> = vec![(i & 0xFF) as u8; token_stride];
            token_data_vec.push(data);
        }

        for (i, data) in token_data_vec.iter().enumerate() {
            tokens.push((0u16, i as u16, data.as_slice()));
        }

        let packet = pack_dispatch_v3(&tokens, 0);
        assert_eq!(packet.token_count, num_tokens as u32);

        let result = unpack_dispatch_v3(&packet.wire_data, token_stride, 1).unwrap();
        assert_eq!(result.total_received, num_tokens);
        assert_eq!(result.tokens_by_expert[0].len(), num_tokens);

        // Verify ordering is preserved
        for (i, (pos, _data)) in result.tokens_by_expert[0].iter().enumerate() {
            assert_eq!(*pos, i as u16);
        }
    }

    // ─── Wire size computation tests ───

    #[test]
    fn test_dispatch_wire_size() {
        // Verify the wire size formula: 4 + count * (4 + token_stride)
        let token_stride = 64;
        let data = vec![0u8; token_stride];
        let counts = [0, 1, 5, 100];

        for &count in &counts {
            let tokens: Vec<(u16, u16, &[u8])> = (0..count)
                .map(|i| (0u16, i as u16, data.as_slice()))
                .collect();

            let packet = pack_dispatch_v3(&tokens, 0);
            let expected = 4 + count * (4 + token_stride);
            assert_eq!(
                packet.wire_data.len(),
                expected,
                "wire size mismatch for count={count}"
            );
        }
    }

    #[test]
    fn test_combine_request_wire_size() {
        // Verify the wire size formula: 4 + count * 8
        let counts = [0, 1, 5, 100];

        for &count in &counts {
            let requests: Vec<(u32, u16, u16)> =
                (0..count).map(|i| (i as u32, 0u16, i as u16)).collect();

            let packed = pack_combine_request_v3(&requests, 0);
            let expected = 4 + count * 8;
            assert_eq!(
                packed.wire_data.len(),
                expected,
                "wire size mismatch for count={count}"
            );
        }
    }
}
