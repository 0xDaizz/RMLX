//! Model architecture definitions.
//!
//! Each sub-module provides:
//! - Config presets (e.g. `qwen2_7b()`, `deepseek_v3_full()`)
//! - Model structs with `from_config()` constructors and `forward()` methods

pub mod deepseek;
pub mod mixtral;
pub mod qwen;

// Re-export model structs for convenience.
pub use deepseek::{DeepSeekV3Block, DeepSeekV3Config, DeepSeekV3Model};
pub use mixtral::{MixtralBlock, MixtralConfig, MixtralModel};
pub use qwen::Qwen2Model;
