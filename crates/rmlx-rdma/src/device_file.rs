//! MLX-compatible JSON device file parser.
//!
//! MLX device files are JSON 2D arrays where `entries[i][j]` is the IB device
//! name that rank *i* should use to communicate with rank *j*, or `null` when
//! `i == j` (same-rank).
//!
//! Example (2-node cluster):
//! ```json
//! [[null, "mlx5_0"], ["mlx5_0", null]]
//! ```

use std::path::Path;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur while loading or parsing a device file.
#[derive(Debug)]
pub enum DeviceFileError {
    /// Underlying I/O error (e.g. file not found).
    Io(std::io::Error),
    /// JSON syntax / structure error.
    Parse(String),
    /// The matrix is not square or is empty.
    InvalidDimensions(String),
}

impl std::fmt::Display for DeviceFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "device file I/O error: {e}"),
            Self::Parse(msg) => write!(f, "device file parse error: {msg}"),
            Self::InvalidDimensions(msg) => write!(f, "device file dimension error: {msg}"),
        }
    }
}

impl std::error::Error for DeviceFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DeviceFileError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

// ---------------------------------------------------------------------------
// DeviceMap
// ---------------------------------------------------------------------------

/// Parsed device map from an MLX-compatible JSON device file.
#[derive(Debug, Clone)]
pub struct DeviceMap {
    /// 2D matrix: `entries[i][j]` = device name for rank i -> rank j,
    /// `None` for same-rank.
    entries: Vec<Vec<Option<String>>>,
    world_size: usize,
}

impl DeviceMap {
    /// Parse an MLX-format JSON string into a `DeviceMap`.
    ///
    /// The expected format is a 2D JSON array of strings and `null` values:
    /// ```text
    /// [[null, "mlx5_0"], ["mlx5_0", null]]
    /// ```
    pub fn from_json(json: &str) -> Result<Self, DeviceFileError> {
        let entries = parse_2d_array(json)?;

        if entries.is_empty() {
            return Err(DeviceFileError::InvalidDimensions(
                "device map must not be empty".into(),
            ));
        }

        let world_size = entries.len();
        for (i, row) in entries.iter().enumerate() {
            if row.len() != world_size {
                return Err(DeviceFileError::InvalidDimensions(format!(
                    "row {i} has {} columns, expected {world_size} (matrix must be square)",
                    row.len()
                )));
            }
        }

        Ok(Self {
            entries,
            world_size,
        })
    }

    /// Create a `DeviceMap` directly from a pre-built matrix.
    ///
    /// Each entry `entries[i][j]` is the device name for rank i → rank j,
    /// or `None` for same-rank / no connection.
    pub fn from_matrix(entries: Vec<Vec<Option<String>>>) -> Result<Self, DeviceFileError> {
        if entries.is_empty() {
            return Err(DeviceFileError::InvalidDimensions(
                "device map must not be empty".into(),
            ));
        }

        let world_size = entries.len();
        for (i, row) in entries.iter().enumerate() {
            if row.len() != world_size {
                return Err(DeviceFileError::InvalidDimensions(format!(
                    "row {i} has {} columns, expected {world_size} (matrix must be square)",
                    row.len()
                )));
            }
        }

        Ok(Self {
            entries,
            world_size,
        })
    }

    /// Read a device file from disk and parse it.
    pub fn from_file(path: &Path) -> Result<Self, DeviceFileError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Look up the IB device name that `src_rank` should use to talk to
    /// `dst_rank`.  Returns `None` when `src_rank == dst_rank` (or the entry
    /// is `null`).
    pub fn device_for(&self, src_rank: usize, dst_rank: usize) -> Option<&str> {
        if src_rank == dst_rank {
            return None; // No self-connection — you never RDMA to yourself.
        }
        self.entries
            .get(src_rank)
            .and_then(|row| row.get(dst_rank))
            .and_then(|v| v.as_deref())
    }

    /// Number of ranks described by the device map.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

// ---------------------------------------------------------------------------
// Hand-rolled minimal JSON parser for `[[...], ...]`
// ---------------------------------------------------------------------------

/// A tiny cursor over a byte slice, used by the parser.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a str) -> Self {
        Self {
            data: data.as_bytes(),
            pos: 0,
        }
    }

    /// Skip ASCII whitespace.
    fn skip_ws(&mut self) {
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Peek at the current byte (after skipping whitespace).
    fn peek(&mut self) -> Result<u8, DeviceFileError> {
        self.skip_ws();
        if self.pos >= self.data.len() {
            return Err(DeviceFileError::Parse("unexpected end of input".into()));
        }
        Ok(self.data[self.pos])
    }

    /// Consume the current byte (after whitespace) if it matches `expected`.
    fn expect(&mut self, expected: u8) -> Result<(), DeviceFileError> {
        let b = self.peek()?;
        if b != expected {
            return Err(DeviceFileError::Parse(format!(
                "expected '{}' at position {}, found '{}'",
                expected as char, self.pos, b as char
            )));
        }
        self.pos += 1;
        Ok(())
    }

    /// Return `true` if we've consumed all non-whitespace input.
    fn is_done(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.data.len()
    }
}

/// Parse the outer 2D array.
fn parse_2d_array(json: &str) -> Result<Vec<Vec<Option<String>>>, DeviceFileError> {
    let mut cur = Cursor::new(json);
    cur.expect(b'[')?;

    let mut rows: Vec<Vec<Option<String>>> = Vec::new();

    loop {
        let b = cur.peek()?;
        if b == b']' {
            cur.pos += 1;
            break;
        }

        if !rows.is_empty() {
            cur.expect(b',')?;
        }

        rows.push(parse_inner_array(&mut cur)?);
    }

    if !cur.is_done() {
        return Err(DeviceFileError::Parse(format!(
            "trailing characters at position {}",
            cur.pos
        )));
    }

    Ok(rows)
}

/// Parse a single inner array: `[elem, elem, ...]`
fn parse_inner_array(cur: &mut Cursor<'_>) -> Result<Vec<Option<String>>, DeviceFileError> {
    cur.expect(b'[')?;

    let mut elems: Vec<Option<String>> = Vec::new();

    loop {
        let b = cur.peek()?;
        if b == b']' {
            cur.pos += 1;
            return Ok(elems);
        }

        if !elems.is_empty() {
            cur.expect(b',')?;
        }

        elems.push(parse_element(cur)?);
    }
}

/// Parse a single element: either `null` or a JSON string `"..."`.
fn parse_element(cur: &mut Cursor<'_>) -> Result<Option<String>, DeviceFileError> {
    let b = cur.peek()?;
    match b {
        b'n' => parse_null(cur),
        b'"' => parse_string(cur).map(Some),
        other => Err(DeviceFileError::Parse(format!(
            "unexpected character '{}' at position {} (expected null or string)",
            other as char, cur.pos
        ))),
    }
}

/// Parse the literal `null`.
fn parse_null(cur: &mut Cursor<'_>) -> Result<Option<String>, DeviceFileError> {
    let remaining = &cur.data[cur.pos..];
    if remaining.len() >= 4 && &remaining[..4] == b"null" {
        cur.pos += 4;
        Ok(None)
    } else {
        Err(DeviceFileError::Parse(format!(
            "expected 'null' at position {}",
            cur.pos
        )))
    }
}

/// Parse a JSON string with full support for standard JSON escapes including
/// `\b`, `\f`, and `\uXXXX`.
fn parse_string(cur: &mut Cursor<'_>) -> Result<String, DeviceFileError> {
    cur.expect(b'"')?;
    let mut s = String::new();
    loop {
        if cur.pos >= cur.data.len() {
            return Err(DeviceFileError::Parse("unterminated string".into()));
        }
        let b = cur.data[cur.pos];
        cur.pos += 1;
        match b {
            b'"' => return Ok(s),
            b'\\' => {
                if cur.pos >= cur.data.len() {
                    return Err(DeviceFileError::Parse(
                        "unterminated escape in string".into(),
                    ));
                }
                let escaped = cur.data[cur.pos];
                cur.pos += 1;
                match escaped {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b't' => s.push('\t'),
                    b'r' => s.push('\r'),
                    b'b' => s.push('\u{0008}'),
                    b'f' => s.push('\u{000C}'),
                    b'u' => {
                        // \uXXXX — 4 hex digits
                        if cur.pos + 4 > cur.data.len() {
                            return Err(DeviceFileError::Parse(format!(
                                "incomplete \\u escape at position {}",
                                cur.pos - 2
                            )));
                        }
                        let hex_str = std::str::from_utf8(&cur.data[cur.pos..cur.pos + 4])
                            .map_err(|_| {
                                DeviceFileError::Parse(format!(
                                    "invalid \\u hex digits at position {}",
                                    cur.pos
                                ))
                            })?;
                        let code_point = u32::from_str_radix(hex_str, 16).map_err(|_| {
                            DeviceFileError::Parse(format!(
                                "invalid \\u hex digits '{}' at position {}",
                                hex_str, cur.pos
                            ))
                        })?;
                        let ch = char::from_u32(code_point).ok_or_else(|| {
                            DeviceFileError::Parse(format!(
                                "invalid unicode code point U+{:04X} at position {}",
                                code_point, cur.pos
                            ))
                        })?;
                        s.push(ch);
                        cur.pos += 4;
                    }
                    other => {
                        return Err(DeviceFileError::Parse(format!(
                            "unknown escape '\\{}' at position {}",
                            other as char,
                            cur.pos - 1
                        )));
                    }
                }
            }
            // Reject raw control characters (U+0000..U+001F) per JSON spec.
            b if b < 0x20 => {
                return Err(DeviceFileError::Parse(format!(
                    "raw control character 0x{:02X} in string at position {}",
                    b,
                    cur.pos - 1
                )));
            }
            _ => s.push(b as char),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_map_parse() {
        // 3x3 matrix
        let json = r#"[
            [null, "mlx5_0", "mlx5_1"],
            ["mlx5_0", null, "mlx5_2"],
            ["mlx5_1", "mlx5_2", null]
        ]"#;
        let map = DeviceMap::from_json(json).unwrap();
        assert_eq!(map.world_size(), 3);
        assert_eq!(map.device_for(0, 0), None);
        assert_eq!(map.device_for(0, 1), Some("mlx5_0"));
        assert_eq!(map.device_for(1, 2), Some("mlx5_2"));
    }

    #[test]
    fn test_device_map_validation() {
        // Non-square matrix
        let json = r#"[["mlx5_0"], ["mlx5_0", "mlx5_1"]]"#;
        assert!(DeviceMap::from_json(json).is_err());

        // Invalid JSON
        assert!(DeviceMap::from_json("not json").is_err());

        // Empty
        assert!(DeviceMap::from_json("[]").is_err());
    }

    #[test]
    fn test_device_map_2x2() {
        let json = r#"[[null, "mlx5_0"], ["mlx5_0", null]]"#;
        let map = DeviceMap::from_json(json).unwrap();
        assert_eq!(map.world_size(), 2);
        assert_eq!(map.device_for(0, 1), Some("mlx5_0"));
        assert_eq!(map.device_for(1, 0), Some("mlx5_0"));
    }

    #[test]
    fn test_device_map_self_rank_always_none() {
        // Even when diagonal entries are non-null, device_for must return None
        // for self-connections — you never RDMA to yourself.
        let json = r#"[["mlx5_0", "mlx5_1"], ["mlx5_1", "mlx5_0"]]"#;
        let map = DeviceMap::from_json(json).unwrap();
        assert_eq!(map.device_for(0, 0), None);
        assert_eq!(map.device_for(1, 1), None);
        assert_eq!(map.device_for(0, 1), Some("mlx5_1"));
    }

    #[test]
    fn test_json_escape_backspace_and_formfeed() {
        // \b and \f are valid JSON escapes
        let json = r#"[[null, "a\b\fc"], ["x", null]]"#;
        let map = DeviceMap::from_json(json).unwrap();
        assert_eq!(map.device_for(0, 1), Some("a\u{0008}\u{000C}c"));
    }

    #[test]
    fn test_json_escape_unicode() {
        // \u0041 is 'A', \u004F is 'O'
        let json = r#"[[null, "\u0041\u004F"], ["mlx5_0", null]]"#;
        let map = DeviceMap::from_json(json).unwrap();
        assert_eq!(map.device_for(0, 1), Some("AO"));
    }

    #[test]
    fn test_json_reject_raw_control_character() {
        // A raw tab (0x09) inside a JSON string is invalid
        let json = "[[null, \"mlx5\t0\"], [\"x\", null]]";
        assert!(DeviceMap::from_json(json).is_err());
    }

    #[test]
    fn test_device_map_from_matrix() {
        let matrix = vec![
            vec![None, Some("rdma_en5".to_string())],
            vec![Some("rdma_en5".to_string()), None],
        ];
        let map = DeviceMap::from_matrix(matrix).unwrap();
        assert_eq!(map.world_size(), 2);
        assert_eq!(map.device_for(0, 1), Some("rdma_en5"));
        assert_eq!(map.device_for(1, 0), Some("rdma_en5"));
        assert_eq!(map.device_for(0, 0), None);
    }

    #[test]
    fn test_device_map_from_matrix_validation() {
        // Empty
        assert!(DeviceMap::from_matrix(vec![]).is_err());
        // Non-square
        let matrix = vec![vec![None], vec![None, None]];
        assert!(DeviceMap::from_matrix(matrix).is_err());
    }
}
