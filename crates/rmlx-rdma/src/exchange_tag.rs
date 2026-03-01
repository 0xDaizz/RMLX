//! ExchangeTag and wr_id encoding/decoding for RDMA work requests.
//!
//! wr_id layout (64-bit):
//! - [63..24] seq      (40 bits, ~1 trillion unique ops)
//! - [23..16] tag      (8 bits, ExchangeTag)
//! - [15..8]  buf_slot (8 bits, PIPELINE slot)
//! - [7..0]   peer_id  (8 bits, max 256 nodes)

/// Identifies the purpose of an RDMA exchange operation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeTag {
    Data = 0,
    Warmup = 1,
    Barrier = 2,
    MoeDispatchCount = 100,
    MoeDispatchPayload = 101,
    MoeCombineOutput = 102,
    MoeCombineWeights = 103,
}

impl ExchangeTag {
    /// Convert a raw u8 value to an ExchangeTag, if valid.
    pub fn from_u8(v: u8) -> Option<ExchangeTag> {
        match v {
            0 => Some(ExchangeTag::Data),
            1 => Some(ExchangeTag::Warmup),
            2 => Some(ExchangeTag::Barrier),
            100 => Some(ExchangeTag::MoeDispatchCount),
            101 => Some(ExchangeTag::MoeDispatchPayload),
            102 => Some(ExchangeTag::MoeCombineOutput),
            103 => Some(ExchangeTag::MoeCombineWeights),
            _ => None,
        }
    }
}

/// Decoded fields from a wr_id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrIdFields {
    pub seq: u64,
    pub tag: ExchangeTag,
    pub buf_slot: u8,
    pub peer_id: u8,
}

const SEQ_SHIFT: u32 = 24;
const TAG_SHIFT: u32 = 16;
const SLOT_SHIFT: u32 = 8;
const BYTE_MASK: u64 = 0xFF;

/// Encode fields into a 64-bit wr_id.
///
/// Only the lower 40 bits of `seq` are used.
pub fn encode_wr_id(seq: u64, tag: ExchangeTag, buf_slot: u8, peer_id: u8) -> u64 {
    let seq_masked = seq & 0xFF_FFFF_FFFF; // 40 bits
    (seq_masked << SEQ_SHIFT)
        | ((tag as u64) << TAG_SHIFT)
        | ((buf_slot as u64) << SLOT_SHIFT)
        | (peer_id as u64)
}

/// Decode a 64-bit wr_id into its constituent fields.
///
/// # Panics
/// Panics if the tag byte does not correspond to a valid `ExchangeTag`.
/// Use `try_decode_wr_id` for a non-panicking variant.
pub fn decode_wr_id(wr_id: u64) -> WrIdFields {
    let seq = (wr_id >> SEQ_SHIFT) & 0xFF_FFFF_FFFF;
    let tag_raw = ((wr_id >> TAG_SHIFT) & BYTE_MASK) as u8;
    let buf_slot = ((wr_id >> SLOT_SHIFT) & BYTE_MASK) as u8;
    let peer_id = (wr_id & BYTE_MASK) as u8;
    let tag = ExchangeTag::from_u8(tag_raw)
        .unwrap_or_else(|| panic!("invalid ExchangeTag value: {tag_raw}"));
    WrIdFields {
        seq,
        tag,
        buf_slot,
        peer_id,
    }
}

/// Try to decode a 64-bit wr_id. Returns `None` if the tag byte is invalid.
pub fn try_decode_wr_id(wr_id: u64) -> Option<WrIdFields> {
    let seq = (wr_id >> SEQ_SHIFT) & 0xFF_FFFF_FFFF;
    let tag_raw = ((wr_id >> TAG_SHIFT) & BYTE_MASK) as u8;
    let buf_slot = ((wr_id >> SLOT_SHIFT) & BYTE_MASK) as u8;
    let peer_id = (wr_id & BYTE_MASK) as u8;
    let tag = ExchangeTag::from_u8(tag_raw)?;
    Some(WrIdFields {
        seq,
        tag,
        buf_slot,
        peer_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let wr_id = encode_wr_id(42, ExchangeTag::Data, 1, 7);
        let fields = decode_wr_id(wr_id);
        assert_eq!(fields.seq, 42);
        assert_eq!(fields.tag, ExchangeTag::Data);
        assert_eq!(fields.buf_slot, 1);
        assert_eq!(fields.peer_id, 7);
    }

    #[test]
    fn roundtrip_all_tags() {
        let tags = [
            ExchangeTag::Data,
            ExchangeTag::Warmup,
            ExchangeTag::Barrier,
            ExchangeTag::MoeDispatchCount,
            ExchangeTag::MoeDispatchPayload,
            ExchangeTag::MoeCombineOutput,
            ExchangeTag::MoeCombineWeights,
        ];
        for (i, &tag) in tags.iter().enumerate() {
            let seq = (i as u64) * 1_000_000;
            let slot = (i as u8) % 4;
            let peer = 255 - i as u8;
            let wr_id = encode_wr_id(seq, tag, slot, peer);
            let fields = decode_wr_id(wr_id);
            assert_eq!(fields.seq, seq, "seq mismatch for tag {:?}", tag);
            assert_eq!(fields.tag, tag, "tag mismatch");
            assert_eq!(fields.buf_slot, slot, "buf_slot mismatch for tag {:?}", tag);
            assert_eq!(fields.peer_id, peer, "peer_id mismatch for tag {:?}", tag);
        }
    }

    #[test]
    fn roundtrip_max_seq() {
        let max_seq: u64 = 0xFF_FFFF_FFFF; // 40-bit max
        let wr_id = encode_wr_id(max_seq, ExchangeTag::MoeCombineWeights, 255, 255);
        let fields = decode_wr_id(wr_id);
        assert_eq!(fields.seq, max_seq);
        assert_eq!(fields.tag, ExchangeTag::MoeCombineWeights);
        assert_eq!(fields.buf_slot, 255);
        assert_eq!(fields.peer_id, 255);
    }

    #[test]
    fn seq_truncated_to_40_bits() {
        // Bits above 40 should be masked off
        let over_seq: u64 = 0x1_00_0000_0001; // bit 40 set + bit 0
        let wr_id = encode_wr_id(over_seq, ExchangeTag::Data, 0, 0);
        let fields = decode_wr_id(wr_id);
        assert_eq!(fields.seq, 1); // only lower 40 bits kept
    }

    #[test]
    fn from_u8_valid() {
        assert_eq!(ExchangeTag::from_u8(0), Some(ExchangeTag::Data));
        assert_eq!(ExchangeTag::from_u8(1), Some(ExchangeTag::Warmup));
        assert_eq!(ExchangeTag::from_u8(2), Some(ExchangeTag::Barrier));
        assert_eq!(ExchangeTag::from_u8(100), Some(ExchangeTag::MoeDispatchCount));
        assert_eq!(ExchangeTag::from_u8(101), Some(ExchangeTag::MoeDispatchPayload));
        assert_eq!(ExchangeTag::from_u8(102), Some(ExchangeTag::MoeCombineOutput));
        assert_eq!(ExchangeTag::from_u8(103), Some(ExchangeTag::MoeCombineWeights));
    }

    #[test]
    fn from_u8_invalid() {
        assert_eq!(ExchangeTag::from_u8(3), None);
        assert_eq!(ExchangeTag::from_u8(99), None);
        assert_eq!(ExchangeTag::from_u8(104), None);
        assert_eq!(ExchangeTag::from_u8(255), None);
    }

    #[test]
    fn try_decode_invalid_tag() {
        // Manually construct a wr_id with an invalid tag (e.g., 50)
        let bad_wr_id: u64 = (1u64 << SEQ_SHIFT) | (50u64 << TAG_SHIFT);
        assert!(try_decode_wr_id(bad_wr_id).is_none());
    }
}
