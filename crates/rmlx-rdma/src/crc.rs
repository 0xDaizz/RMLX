//! Application-level CRC32 integrity checks for UC (Unreliable Connection) RDMA transport.
//!
//! UC transport lacks the built-in reliability guarantees of RC (Reliable Connection).
//! This module provides CRC32 checksum append/verify utilities so that data corruption
//! on UC links is detected at the application layer.
//!
//! # Wire format
//!
//! ```text
//! ┌──────────────────────────┬──────────────┐
//! │       payload bytes      │  CRC32 (4B)  │
//! └──────────────────────────┴──────────────┘
//! ```
//!
//! The CRC32 is computed over the payload bytes and appended as the last 4 bytes
//! of the message in **little-endian** byte order.

use crate::RdmaError;

/// Size of the CRC32 checksum trailer in bytes.
pub const CRC32_SIZE: usize = 4;

/// Compute the CRC32 checksum of `payload` and append it (little-endian) to `buf`.
///
/// `buf` must have at least `payload.len() + CRC32_SIZE` capacity.
/// Returns the total frame length (payload + CRC32).
pub fn append_crc32(payload: &[u8], buf: &mut Vec<u8>) -> usize {
    let crc = crc32fast::hash(payload);
    buf.extend_from_slice(payload);
    buf.extend_from_slice(&crc.to_le_bytes());
    payload.len() + CRC32_SIZE
}

/// Compute the CRC32 checksum of `payload` and write it into the 4 bytes
/// immediately following the payload in the provided buffer slice.
///
/// `buf` must be exactly `payload_len + CRC32_SIZE` bytes long.
/// The first `payload_len` bytes are the payload; this function writes
/// the CRC32 into `buf[payload_len..payload_len + 4]`.
pub fn stamp_crc32(buf: &mut [u8], payload_len: usize) {
    debug_assert!(buf.len() >= payload_len + CRC32_SIZE);
    let crc = crc32fast::hash(&buf[..payload_len]);
    buf[payload_len..payload_len + CRC32_SIZE].copy_from_slice(&crc.to_le_bytes());
}

/// Verify the CRC32 checksum on a received frame.
///
/// `frame` must contain at least `CRC32_SIZE + 1` bytes (at least 1 byte of payload).
/// The last 4 bytes are interpreted as a little-endian CRC32 of the preceding bytes.
///
/// Returns `Ok(payload_slice)` on success, or `Err(RdmaError::DataCorruption)` on mismatch.
pub fn verify_crc32(frame: &[u8]) -> Result<&[u8], RdmaError> {
    if frame.len() < CRC32_SIZE {
        return Err(RdmaError::DataCorruption(format!(
            "frame too short for CRC32: {} bytes (minimum {})",
            frame.len(),
            CRC32_SIZE
        )));
    }

    let payload_len = frame.len() - CRC32_SIZE;
    let payload = &frame[..payload_len];
    let expected_bytes: [u8; 4] = frame[payload_len..].try_into().unwrap();
    let expected_crc = u32::from_le_bytes(expected_bytes);

    let actual_crc = crc32fast::hash(payload);

    if actual_crc != expected_crc {
        return Err(RdmaError::DataCorruption(format!(
            "CRC32 mismatch: expected 0x{:08x}, got 0x{:08x} (payload {} bytes)",
            expected_crc, actual_crc, payload_len
        )));
    }

    Ok(payload)
}

/// Compute the total transfer size including CRC32 overhead for a given payload size.
#[inline]
pub fn transfer_size(payload_len: usize) -> usize {
    payload_len + CRC32_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_roundtrip_clean_data() {
        let payload = b"Hello, RDMA world! This is a test payload.";
        let mut buf = Vec::new();
        let total = append_crc32(payload, &mut buf);

        assert_eq!(total, payload.len() + CRC32_SIZE);
        assert_eq!(buf.len(), total);

        let verified = verify_crc32(&buf).expect("clean data should pass CRC32 verification");
        assert_eq!(verified, payload.as_slice());
    }

    #[test]
    fn test_crc32_detects_corruption() {
        let payload = b"Important ML weights that must not be corrupted";
        let mut buf = Vec::new();
        append_crc32(payload, &mut buf);

        // Flip a byte in the payload region
        buf[10] ^= 0xFF;

        let result = verify_crc32(&buf);
        assert!(result.is_err(), "corrupted data should fail CRC32 check");
        match result {
            Err(RdmaError::DataCorruption(msg)) => {
                assert!(
                    msg.contains("CRC32 mismatch"),
                    "error should mention CRC32 mismatch"
                );
            }
            other => panic!("expected DataCorruption, got {:?}", other),
        }
    }

    #[test]
    fn test_crc32_detects_trailing_corruption() {
        let payload = b"test data";
        let mut buf = Vec::new();
        append_crc32(payload, &mut buf);

        // Flip a byte in the CRC32 trailer
        let last = buf.len() - 1;
        buf[last] ^= 0x01;

        let result = verify_crc32(&buf);
        assert!(
            result.is_err(),
            "corrupted CRC trailer should fail verification"
        );
    }

    #[test]
    fn test_crc32_frame_too_short() {
        let short_frame = [0u8; 3]; // Less than CRC32_SIZE
        let result = verify_crc32(&short_frame);
        assert!(result.is_err());
        match result {
            Err(RdmaError::DataCorruption(msg)) => {
                assert!(msg.contains("too short"));
            }
            other => panic!("expected DataCorruption, got {:?}", other),
        }
    }

    #[test]
    fn test_crc32_overhead_accounting() {
        assert_eq!(transfer_size(0), CRC32_SIZE);
        assert_eq!(transfer_size(100), 104);
        assert_eq!(transfer_size(4096), 4100);
    }

    #[test]
    fn test_stamp_crc32_in_place() {
        let payload = b"stamp test payload";
        let payload_len = payload.len();
        let mut buf = vec![0u8; payload_len + CRC32_SIZE];
        buf[..payload_len].copy_from_slice(payload);

        stamp_crc32(&mut buf, payload_len);

        let verified = verify_crc32(&buf).expect("stamped CRC should verify");
        assert_eq!(verified, payload.as_slice());
    }

    #[test]
    fn test_empty_payload_roundtrip() {
        let payload: &[u8] = b"";
        let mut buf = Vec::new();
        let total = append_crc32(payload, &mut buf);

        assert_eq!(total, CRC32_SIZE);
        let verified = verify_crc32(&buf).expect("empty payload should verify");
        assert_eq!(verified.len(), 0);
    }

    #[test]
    fn test_large_payload_roundtrip() {
        // 1 MB payload
        let payload: Vec<u8> = (0..1_048_576).map(|i| (i % 256) as u8).collect();
        let mut buf = Vec::new();
        append_crc32(&payload, &mut buf);

        let verified = verify_crc32(&buf).expect("large payload should verify");
        assert_eq!(verified, payload.as_slice());
    }
}
