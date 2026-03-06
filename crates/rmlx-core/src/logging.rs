//! Structured logging for RMLX runtime.

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log severity levels, ordered from most to least severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Error,
            1 => Self::Warn,
            2 => Self::Info,
            3 => Self::Debug,
            4 => Self::Trace,
            _ => Self::Trace,
        }
    }
}

static GLOBAL_LEVEL: AtomicU8 = AtomicU8::new(2); // Info default

/// Set the global log level filter.
pub fn set_level(level: LogLevel) {
    GLOBAL_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Get the current global log level.
pub fn current_level() -> LogLevel {
    LogLevel::from_u8(GLOBAL_LEVEL.load(Ordering::Relaxed))
}

/// Check whether a given log level is enabled.
pub fn is_enabled(level: LogLevel) -> bool {
    (level as u8) <= GLOBAL_LEVEL.load(Ordering::Relaxed)
}

/// A structured log entry with optional key-value fields.
pub struct LogEntry {
    pub timestamp_ms: u64,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub fields: Vec<(String, String)>,
}

impl LogEntry {
    /// Create a new log entry with the current timestamp.
    pub fn new(level: LogLevel, target: &str, message: &str) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            timestamp_ms,
            level,
            target: target.to_string(),
            message: message.to_string(),
            fields: Vec::new(),
        }
    }

    /// Add a key-value field to this entry.
    pub fn field(mut self, key: &str, value: &str) -> Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Format as a JSON string.
    pub fn format_json(&self) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("{\"ts\":");
        s.push_str(&self.timestamp_ms.to_string());
        s.push_str(",\"level\":\"");
        s.push_str(self.level.as_str());
        s.push_str("\",\"target\":\"");
        json_escape_into(&mut s, &self.target);
        s.push_str("\",\"msg\":\"");
        json_escape_into(&mut s, &self.message);
        s.push('"');
        for (k, v) in &self.fields {
            s.push_str(",\"");
            json_escape_into(&mut s, k);
            s.push_str("\":\"");
            json_escape_into(&mut s, v);
            s.push('"');
        }
        s.push('}');
        s
    }

    /// Format as a human-readable text line.
    pub fn format_text(&self) -> String {
        let mut s = String::with_capacity(128);
        s.push('[');
        s.push_str(&self.timestamp_ms.to_string());
        s.push_str("] ");
        s.push_str(self.level.as_str());
        s.push(' ');
        s.push_str(&self.target);
        s.push_str(": ");
        s.push_str(&self.message);
        for (k, v) in &self.fields {
            s.push(' ');
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s
    }
}

/// Escape a string for JSON output (minimal: backslash, quote, control chars).
fn json_escape_into(buf: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if c.is_control() => {
                buf.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => buf.push(c),
        }
    }
}

/// Log a message at the given level (emits via `tracing` if enabled).
pub fn log(level: LogLevel, target: &str, message: &str) {
    if is_enabled(level) {
        let entry = LogEntry::new(level, target, message);
        match level {
            LogLevel::Error => tracing::error!("{}", entry.format_text()),
            LogLevel::Warn => tracing::warn!("{}", entry.format_text()),
            LogLevel::Info => tracing::info!("{}", entry.format_text()),
            LogLevel::Debug => tracing::debug!("{}", entry.format_text()),
            LogLevel::Trace => tracing::trace!("{}", entry.format_text()),
        }
    }
}

/// Log a message with key-value fields.
pub fn log_with_fields(level: LogLevel, target: &str, message: &str, fields: &[(&str, &str)]) {
    if is_enabled(level) {
        let mut entry = LogEntry::new(level, target, message);
        for (k, v) in fields {
            entry.fields.push(((*k).to_string(), (*v).to_string()));
        }
        match level {
            LogLevel::Error => tracing::error!("{}", entry.format_text()),
            LogLevel::Warn => tracing::warn!("{}", entry.format_text()),
            LogLevel::Info => tracing::info!("{}", entry.format_text()),
            LogLevel::Debug => tracing::debug!("{}", entry.format_text()),
            LogLevel::Trace => tracing::trace!("{}", entry.format_text()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
    }

    #[test]
    fn test_log_level_from_u8_roundtrip() {
        for v in 0..=4u8 {
            let level = LogLevel::from_u8(v);
            assert_eq!(level as u8, v);
        }
    }

    #[test]
    fn test_log_level_from_u8_out_of_range() {
        // Values > 4 should map to Trace.
        assert_eq!(LogLevel::from_u8(5), LogLevel::Trace);
        assert_eq!(LogLevel::from_u8(255), LogLevel::Trace);
    }

    #[test]
    fn test_log_entry_format_json_basic() {
        let entry = LogEntry {
            timestamp_ms: 1234567890,
            level: LogLevel::Info,
            target: "rmlx::test".to_string(),
            message: "hello world".to_string(),
            fields: Vec::new(),
        };
        let json = entry.format_json();
        assert!(json.contains("\"ts\":1234567890"));
        assert!(json.contains("\"level\":\"INFO\""));
        assert!(json.contains("\"target\":\"rmlx::test\""));
        assert!(json.contains("\"msg\":\"hello world\""));
    }

    #[test]
    fn test_log_entry_format_json_with_fields() {
        let entry = LogEntry {
            timestamp_ms: 100,
            level: LogLevel::Error,
            target: "test".to_string(),
            message: "fail".to_string(),
            fields: vec![
                ("key1".to_string(), "val1".to_string()),
                ("key2".to_string(), "val2".to_string()),
            ],
        };
        let json = entry.format_json();
        assert!(json.contains("\"key1\":\"val1\""));
        assert!(json.contains("\"key2\":\"val2\""));
    }

    #[test]
    fn test_log_entry_format_json_escaping() {
        let entry = LogEntry {
            timestamp_ms: 0,
            level: LogLevel::Warn,
            target: "test".to_string(),
            message: "line1\nline2\ttab\"quote\\back".to_string(),
            fields: Vec::new(),
        };
        let json = entry.format_json();
        assert!(json.contains("\\n"));
        assert!(json.contains("\\t"));
        assert!(json.contains("\\\""));
        assert!(json.contains("\\\\"));
    }

    #[test]
    fn test_log_entry_format_text_basic() {
        let entry = LogEntry {
            timestamp_ms: 999,
            level: LogLevel::Debug,
            target: "mymod".to_string(),
            message: "something happened".to_string(),
            fields: Vec::new(),
        };
        let text = entry.format_text();
        assert!(text.starts_with("[999] DEBUG mymod: something happened"));
    }

    #[test]
    fn test_log_entry_format_text_with_fields() {
        let entry = LogEntry {
            timestamp_ms: 0,
            level: LogLevel::Info,
            target: "t".to_string(),
            message: "m".to_string(),
            fields: vec![("k".to_string(), "v".to_string())],
        };
        let text = entry.format_text();
        assert!(text.contains("k=v"));
    }

    #[test]
    fn test_log_entry_new_has_nonzero_timestamp() {
        let entry = LogEntry::new(LogLevel::Info, "test", "msg");
        // Timestamp should be non-zero (current time).
        assert!(entry.timestamp_ms > 0);
    }

    #[test]
    fn test_log_entry_field_builder() {
        let entry = LogEntry::new(LogLevel::Info, "test", "msg")
            .field("a", "1")
            .field("b", "2");
        assert_eq!(entry.fields.len(), 2);
        assert_eq!(entry.fields[0], ("a".to_string(), "1".to_string()));
        assert_eq!(entry.fields[1], ("b".to_string(), "2".to_string()));
    }

    #[test]
    fn test_is_enabled_respects_level() {
        // Note: this test modifies global state. In a real test suite we'd
        // want to serialize tests that touch GLOBAL_LEVEL.
        let original = current_level();

        set_level(LogLevel::Warn);
        assert!(is_enabled(LogLevel::Error));
        assert!(is_enabled(LogLevel::Warn));
        assert!(!is_enabled(LogLevel::Info));
        assert!(!is_enabled(LogLevel::Debug));
        assert!(!is_enabled(LogLevel::Trace));

        // Restore.
        set_level(original);
    }

    #[test]
    fn test_set_and_get_level() {
        let original = current_level();

        set_level(LogLevel::Trace);
        assert_eq!(current_level(), LogLevel::Trace);

        set_level(LogLevel::Error);
        assert_eq!(current_level(), LogLevel::Error);

        set_level(original);
    }
}
