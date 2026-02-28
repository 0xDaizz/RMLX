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

/// Log a message at the given level (prints to stderr if enabled).
pub fn log(level: LogLevel, target: &str, message: &str) {
    if is_enabled(level) {
        let entry = LogEntry::new(level, target, message);
        eprintln!("{}", entry.format_text());
    }
}

/// Log a message with key-value fields.
pub fn log_with_fields(level: LogLevel, target: &str, message: &str, fields: &[(&str, &str)]) {
    if is_enabled(level) {
        let mut entry = LogEntry::new(level, target, message);
        for (k, v) in fields {
            entry.fields.push(((*k).to_string(), (*v).to_string()));
        }
        eprintln!("{}", entry.format_text());
    }
}
