//! Continuous batching scheduler for LLM serving.
//!
//! Manages a waiting queue of generation requests and an active batch,
//! admitting new sequences when capacity allows (max batch size and
//! available KV blocks), evicting completed sequences, and producing
//! a [`SchedulerOutput`] each iteration that describes which sequences
//! to prefill or decode.

use std::collections::{HashMap, VecDeque};

use crate::paged_kv_cache::{BlockId, BlockManager};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the continuous batching scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences in the active batch at once.
    pub max_batch_size: usize,
    /// EOS token ID — sequences producing this token are considered done.
    pub eos_token_id: u32,
    /// Number of tokens per KV block (must match BlockManager's block_size).
    pub block_size: usize,
    /// Maximum number of tokens to prefill in a single chunk.
    /// Prompts longer than this are split across multiple scheduling iterations,
    /// interleaving decode batches between chunks to maintain decode latency.
    /// Default: 512.
    pub max_prefill_chunk: usize,
}

// ---------------------------------------------------------------------------
// Request / sequence metadata
// ---------------------------------------------------------------------------

/// An incoming generation request waiting to be scheduled.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique sequence identifier.
    pub seq_id: u64,
    /// Prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Maximum number of *output* tokens to generate (excluding prompt).
    pub max_output_len: usize,
}

/// Metadata for a sequence that the engine should process this step.
#[derive(Debug, Clone)]
pub struct SeqMeta {
    /// Unique sequence identifier.
    pub seq_id: u64,
    /// Length of the original prompt (in tokens).
    pub prompt_len: usize,
    /// How many tokens have been generated so far (excluding prompt).
    pub current_len: usize,
    /// Block table mapping logical blocks to physical BlockIds.
    pub block_table: Vec<BlockId>,
    /// How many prompt tokens have been prefilled so far (chunked prefill progress).
    /// 0 means not started; equals `prompt_len` when prefill is complete.
    pub prefill_progress: usize,
}

impl SeqMeta {
    /// Returns `true` if this sequence is mid-prefill (chunked prefill in progress
    /// but not yet complete).
    pub fn is_chunked_prefill(&self) -> bool {
        self.prefill_progress > 0 && self.prefill_progress < self.prompt_len
    }
}

/// Output of one scheduling iteration.
#[derive(Debug, Clone)]
pub struct SchedulerOutput {
    /// Sequences that need a prefill pass (may be a chunk of the full prompt).
    pub prefill_seqs: Vec<SeqMeta>,
    /// Existing sequences that need one decode step.
    pub decode_seqs: Vec<SeqMeta>,
    /// Sequence IDs that completed and were evicted this step.
    pub evicted_seq_ids: Vec<u64>,
    /// If chunked prefill is active, the `(start, end)` token range within
    /// the prompt that this batch covers. `None` for non-chunked prefills.
    pub prefill_chunk_range: Option<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// Internal active sequence state
// ---------------------------------------------------------------------------

/// Tracks the state of a sequence currently in the active batch.
#[derive(Debug)]
struct ActiveSequence {
    /// Original prompt length.
    prompt_len: usize,
    /// Number of output tokens generated so far.
    generated_len: usize,
    /// Maximum output tokens allowed.
    max_output_len: usize,
    /// Whether this sequence has already been fully prefilled.
    prefilled: bool,
    /// Number of prompt tokens prefilled so far (for chunked prefill).
    prefill_progress: usize,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Continuous batching scheduler.
///
/// Call [`Scheduler::add_request`] to enqueue new generation requests, then
/// call [`Scheduler::schedule`] each iteration to get a [`SchedulerOutput`]
/// describing which sequences to run. After the engine produces new tokens,
/// call [`Scheduler::update_sequence`] (or [`Scheduler::mark_finished`]) to
/// inform the scheduler of progress.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Waiting queue of requests not yet admitted.
    waiting: VecDeque<GenerationRequest>,
    /// Currently active sequences keyed by seq_id.
    active: HashMap<u64, ActiveSequence>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            active: HashMap::new(),
        }
    }

    /// Enqueue a generation request. It will be admitted when capacity allows.
    pub fn add_request(&mut self, request: GenerationRequest) {
        self.waiting.push_back(request);
    }

    /// Number of requests waiting in the queue (not yet active).
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of sequences currently in the active batch.
    pub fn num_active(&self) -> usize {
        self.active.len()
    }

    /// Run one scheduling iteration.
    ///
    /// 1. Identify and evict completed sequences (freeing their KV blocks).
    /// 2. Continue any in-progress chunked prefills, or admit new requests.
    /// 3. Partition active sequences into prefill vs. decode.
    ///
    /// When a prompt is longer than `config.max_prefill_chunk`, the prefill is
    /// split into chunks. Between chunks, decode batches are interleaved so
    /// that decode latency is not starved by long prefills.
    ///
    /// `block_manager` is used to allocate blocks for newly admitted sequences
    /// and to free blocks for evicted sequences.
    ///
    /// `finished_tokens` maps seq_id to the *last produced token* so the
    /// scheduler can detect EOS. Pass an empty map on the first call.
    pub fn schedule(
        &mut self,
        block_manager: &mut BlockManager,
        finished_tokens: &HashMap<u64, u32>,
    ) -> SchedulerOutput {
        let mut evicted_seq_ids = Vec::new();

        // --- Step 1: Identify and evict completed sequences ---
        let active_ids: Vec<u64> = self.active.keys().copied().collect();
        for seq_id in active_ids {
            let seq = &self.active[&seq_id];
            let hit_max_len = seq.generated_len >= seq.max_output_len;
            let hit_eos = finished_tokens
                .get(&seq_id)
                .is_some_and(|&tok| tok == self.config.eos_token_id);

            if hit_max_len || hit_eos {
                self.active.remove(&seq_id);
                block_manager.free_sequence(seq_id);
                evicted_seq_ids.push(seq_id);
            }
        }

        // --- Step 2a: Continue in-progress chunked prefills ---
        let mut prefill_seqs = Vec::new();
        let mut prefill_chunk_range: Option<(usize, usize)> = None;

        // Find any sequence that is mid-chunked-prefill.
        let chunked_seq_id = self.active.iter().find_map(|(&id, seq)| {
            if !seq.prefilled && seq.prefill_progress > 0 && seq.prefill_progress < seq.prompt_len {
                Some(id)
            } else {
                None
            }
        });

        if let Some(seq_id) = chunked_seq_id {
            let seq = self.active.get_mut(&seq_id).unwrap();
            let start = seq.prefill_progress;
            let remaining = seq.prompt_len - start;
            let chunk_size = remaining.min(self.config.max_prefill_chunk);
            let end = start + chunk_size;

            seq.prefill_progress = end;
            let is_final_chunk = end >= seq.prompt_len;
            if is_final_chunk {
                seq.prefilled = true;
            }

            let block_table = block_manager.block_table(seq_id).to_vec();
            prefill_seqs.push(SeqMeta {
                seq_id,
                prompt_len: seq.prompt_len,
                current_len: 0,
                block_table,
                prefill_progress: end,
            });
            prefill_chunk_range = Some((start, end));
        }

        // --- Step 2b: Admit new requests if no chunked prefill is in progress ---
        if prefill_seqs.is_empty() {
            while self.active.len() < self.config.max_batch_size {
                let Some(request) = self.waiting.pop_front() else {
                    break;
                };

                // Calculate how many blocks the full prompt needs.
                let prompt_len = request.prompt_tokens.len();
                let blocks_needed = blocks_for_tokens(prompt_len, self.config.block_size);

                if block_manager.num_free_blocks() < blocks_needed {
                    // Not enough blocks — put the request back and stop admitting.
                    self.waiting.push_front(request);
                    break;
                }

                // Allocate blocks for this sequence.
                for _ in 0..blocks_needed {
                    // unwrap is safe: we checked num_free_blocks above.
                    block_manager.append_block(request.seq_id).unwrap();
                }

                let block_table = block_manager.block_table(request.seq_id).to_vec();

                // Determine if we need chunked prefill.
                let needs_chunking = prompt_len > self.config.max_prefill_chunk;
                let chunk_end = if needs_chunking {
                    self.config.max_prefill_chunk
                } else {
                    prompt_len
                };

                let prefilled = !needs_chunking;

                prefill_seqs.push(SeqMeta {
                    seq_id: request.seq_id,
                    prompt_len,
                    current_len: 0,
                    block_table,
                    prefill_progress: chunk_end,
                });

                if needs_chunking {
                    prefill_chunk_range = Some((0, chunk_end));
                }

                self.active.insert(
                    request.seq_id,
                    ActiveSequence {
                        prompt_len,
                        generated_len: 0,
                        max_output_len: request.max_output_len,
                        prefilled,
                        prefill_progress: chunk_end,
                    },
                );
            }
        }

        // --- Step 3: Build decode list from already-prefilled active sequences ---
        let mut decode_seqs = Vec::new();

        for (&seq_id, seq) in &self.active {
            if seq.prefilled && !prefill_seqs.iter().any(|p| p.seq_id == seq_id) {
                let block_table = block_manager.block_table(seq_id).to_vec();
                decode_seqs.push(SeqMeta {
                    seq_id,
                    prompt_len: seq.prompt_len,
                    current_len: seq.generated_len,
                    block_table,
                    prefill_progress: seq.prompt_len,
                });
            }
        }

        SchedulerOutput {
            prefill_seqs,
            decode_seqs,
            evicted_seq_ids,
            prefill_chunk_range,
        }
    }

    /// Notify the scheduler that a sequence produced one new token.
    ///
    /// If the sequence needs more blocks for the new token, a block is
    /// allocated automatically. Returns `Err` if the block pool is exhausted.
    pub fn update_sequence(
        &mut self,
        seq_id: u64,
        block_manager: &mut BlockManager,
    ) -> Result<(), SchedulerError> {
        let seq = self
            .active
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound(seq_id))?;

        seq.generated_len += 1;

        // Total tokens = prompt + generated so far.
        let total_tokens = seq.prompt_len + seq.generated_len;
        let blocks_needed = blocks_for_tokens(total_tokens, self.config.block_size);
        let blocks_have = block_manager.block_table(seq_id).len();

        if blocks_needed > blocks_have {
            block_manager
                .append_block(seq_id)
                .map_err(|_| SchedulerError::OutOfBlocks)?;
        }

        Ok(())
    }

    /// Manually mark a sequence as finished (will be evicted on next schedule).
    ///
    /// This is useful when the caller detects completion outside of the
    /// normal EOS / max-length checks.
    pub fn mark_finished(&mut self, seq_id: u64, block_manager: &mut BlockManager) {
        if self.active.remove(&seq_id).is_some() {
            block_manager.free_sequence(seq_id);
        }
    }

    /// Returns `true` if there are no active sequences and no waiting requests.
    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of blocks needed to store `num_tokens` tokens with given block size.
fn blocks_for_tokens(num_tokens: usize, block_size: usize) -> usize {
    num_tokens.div_ceil(block_size)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from scheduler operations.
#[derive(Debug)]
pub enum SchedulerError {
    /// Sequence ID not found in the active batch.
    SequenceNotFound(u64),
    /// No free KV blocks available.
    OutOfBlocks,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::SequenceNotFound(id) => {
                write!(f, "scheduler: sequence {id} not found in active batch")
            }
            SchedulerError::OutOfBlocks => {
                write!(f, "scheduler: no free KV blocks available")
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_core::dtype::DType;
    use rmlx_metal::MtlDevice;

    use std::sync::OnceLock;

    fn test_device() -> &'static MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
        })
    }

    /// Helper: create a BlockManager with small parameters for testing.
    fn test_block_manager(num_blocks: usize) -> BlockManager {
        BlockManager::new(
            test_device(),
            num_blocks,
            4, // block_size: 4 tokens per block
            2, // num_layers
            2, // num_kv_heads
            4, // head_dim
            DType::Float32,
        )
    }

    fn test_config() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 4,
            eos_token_id: 2, // typical EOS
            block_size: 4,
            max_prefill_chunk: 512, // large enough that existing tests don't trigger chunking
        }
    }

    fn make_request(seq_id: u64, prompt_len: usize, max_output_len: usize) -> GenerationRequest {
        GenerationRequest {
            seq_id,
            prompt_tokens: vec![1; prompt_len],
            max_output_len,
        }
    }

    #[test]
    fn test_add_request_and_schedule() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        scheduler.add_request(make_request(1, 8, 10));
        scheduler.add_request(make_request(2, 4, 5));
        assert_eq!(scheduler.num_waiting(), 2);

        let output = scheduler.schedule(&mut bm, &HashMap::new());

        // Both requests should be admitted as prefill.
        assert_eq!(output.prefill_seqs.len(), 2);
        assert_eq!(output.decode_seqs.len(), 0);
        assert!(output.evicted_seq_ids.is_empty());
        assert_eq!(scheduler.num_active(), 2);
        assert_eq!(scheduler.num_waiting(), 0);

        // Verify seq metadata.
        let seq1 = output.prefill_seqs.iter().find(|s| s.seq_id == 1).unwrap();
        assert_eq!(seq1.prompt_len, 8);
        assert_eq!(seq1.current_len, 0);
        assert_eq!(seq1.block_table.len(), 2); // 8 tokens / 4 block_size = 2 blocks

        let seq2 = output.prefill_seqs.iter().find(|s| s.seq_id == 2).unwrap();
        assert_eq!(seq2.prompt_len, 4);
        assert_eq!(seq2.block_table.len(), 1); // 4 tokens / 4 block_size = 1 block
    }

    #[test]
    fn test_prefill_then_decode() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        scheduler.add_request(make_request(1, 4, 10));

        // First schedule: prefill.
        let out1 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out1.prefill_seqs.len(), 1);
        assert_eq!(out1.decode_seqs.len(), 0);

        // Simulate producing one token.
        scheduler.update_sequence(1, &mut bm).unwrap();

        // Second schedule: decode.
        let out2 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out2.prefill_seqs.len(), 0);
        assert_eq!(out2.decode_seqs.len(), 1);
        assert_eq!(out2.decode_seqs[0].current_len, 1);
    }

    #[test]
    fn test_max_batch_size_enforcement() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        let mut bm = test_block_manager(32);

        // Add 3 requests, batch size is 2.
        scheduler.add_request(make_request(1, 4, 10));
        scheduler.add_request(make_request(2, 4, 10));
        scheduler.add_request(make_request(3, 4, 10));

        let output = scheduler.schedule(&mut bm, &HashMap::new());

        // Only 2 should be admitted.
        assert_eq!(output.prefill_seqs.len(), 2);
        assert_eq!(scheduler.num_active(), 2);
        assert_eq!(scheduler.num_waiting(), 1);

        // Third request remains waiting.
        let admitted_ids: Vec<u64> = output.prefill_seqs.iter().map(|s| s.seq_id).collect();
        assert!(admitted_ids.contains(&1));
        assert!(admitted_ids.contains(&2));
    }

    #[test]
    fn test_sequence_completion_by_max_len() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        // Request with max_output_len = 2.
        scheduler.add_request(make_request(1, 4, 2));

        // Prefill.
        scheduler.schedule(&mut bm, &HashMap::new());

        // Generate 2 tokens (reaching max_output_len).
        scheduler.update_sequence(1, &mut bm).unwrap();
        scheduler.update_sequence(1, &mut bm).unwrap();

        // Schedule should evict the sequence.
        let output = scheduler.schedule(&mut bm, &HashMap::new());
        assert!(output.evicted_seq_ids.contains(&1));
        assert_eq!(scheduler.num_active(), 0);
    }

    #[test]
    fn test_sequence_completion_by_eos() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        scheduler.add_request(make_request(1, 4, 100));

        // Prefill.
        scheduler.schedule(&mut bm, &HashMap::new());

        // One decode step.
        scheduler.update_sequence(1, &mut bm).unwrap();

        // Signal EOS.
        let mut finished = HashMap::new();
        finished.insert(1u64, 2u32); // eos_token_id = 2

        let output = scheduler.schedule(&mut bm, &finished);
        assert!(output.evicted_seq_ids.contains(&1));
        assert_eq!(scheduler.num_active(), 0);
    }

    #[test]
    fn test_eviction_frees_blocks() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(8);

        let free_before = bm.num_free_blocks();

        scheduler.add_request(make_request(1, 8, 1)); // needs 2 blocks

        // Prefill — allocates 2 blocks.
        scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(bm.num_free_blocks(), free_before - 2);

        // Generate 1 token (hits max_output_len).
        scheduler.update_sequence(1, &mut bm).unwrap();

        // Schedule evicts -> blocks freed.
        scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(bm.num_free_blocks(), free_before);
    }

    #[test]
    fn test_block_allocation_failure() {
        let config = SchedulerConfig {
            max_batch_size: 10,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        // Only 2 blocks available total.
        let mut bm = test_block_manager(2);

        // First request: 8 tokens = 2 blocks -> exhausts pool.
        scheduler.add_request(make_request(1, 8, 10));
        // Second request: 4 tokens = 1 block -> no blocks available.
        scheduler.add_request(make_request(2, 4, 10));

        let output = scheduler.schedule(&mut bm, &HashMap::new());

        // Only the first request should be admitted.
        assert_eq!(output.prefill_seqs.len(), 1);
        assert_eq!(output.prefill_seqs[0].seq_id, 1);
        // Second request stays in waiting queue.
        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_active(), 1);
    }

    #[test]
    fn test_waiting_request_admitted_after_eviction() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        let mut bm = test_block_manager(8);

        scheduler.add_request(make_request(1, 4, 1));
        scheduler.add_request(make_request(2, 4, 10));

        // First schedule: only seq 1 admitted (batch size = 1).
        scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(scheduler.num_active(), 1);
        assert_eq!(scheduler.num_waiting(), 1);

        // Generate 1 token -> seq 1 hits max len.
        scheduler.update_sequence(1, &mut bm).unwrap();

        // Second schedule: seq 1 evicted, seq 2 admitted.
        let output = scheduler.schedule(&mut bm, &HashMap::new());
        assert!(output.evicted_seq_ids.contains(&1));
        assert_eq!(output.prefill_seqs.len(), 1);
        assert_eq!(output.prefill_seqs[0].seq_id, 2);
        assert_eq!(scheduler.num_active(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_mark_finished() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        scheduler.add_request(make_request(1, 4, 100));
        scheduler.schedule(&mut bm, &HashMap::new());

        assert_eq!(scheduler.num_active(), 1);

        scheduler.mark_finished(1, &mut bm);
        assert_eq!(scheduler.num_active(), 0);
    }

    #[test]
    fn test_is_idle() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        assert!(scheduler.is_idle());

        scheduler.add_request(make_request(1, 4, 1));
        assert!(!scheduler.is_idle());

        scheduler.schedule(&mut bm, &HashMap::new());
        assert!(!scheduler.is_idle()); // still active

        scheduler.update_sequence(1, &mut bm).unwrap();
        scheduler.schedule(&mut bm, &HashMap::new()); // evicts
        assert!(scheduler.is_idle());
    }

    #[test]
    fn test_block_table_grows_during_decode() {
        let mut scheduler = Scheduler::new(test_config());
        let mut bm = test_block_manager(32);

        // Prompt exactly fills 1 block (4 tokens, block_size=4).
        scheduler.add_request(make_request(1, 4, 10));
        scheduler.schedule(&mut bm, &HashMap::new());

        // Block table should have 1 block.
        assert_eq!(bm.block_table(1).len(), 1);

        // Generate token 5 -> needs a second block.
        scheduler.update_sequence(1, &mut bm).unwrap();
        assert_eq!(bm.block_table(1).len(), 2);
    }

    // ===== Chunked prefill tests =====

    #[test]
    fn test_chunked_prefill_splits_long_prompt() {
        // prompt_len=2048, chunk=512 -> should take 4 scheduling iterations.
        let config = SchedulerConfig {
            max_batch_size: 4,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        // Need enough blocks: 2048 tokens / 4 block_size = 512 blocks.
        let mut bm = test_block_manager(1024);

        scheduler.add_request(make_request(1, 2048, 10));

        // Iteration 1: first chunk [0, 512)
        let out1 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out1.prefill_seqs.len(), 1);
        assert_eq!(out1.prefill_seqs[0].seq_id, 1);
        assert_eq!(out1.prefill_seqs[0].prefill_progress, 512);
        assert!(out1.prefill_seqs[0].is_chunked_prefill());
        assert_eq!(out1.prefill_chunk_range, Some((0, 512)));
        assert_eq!(out1.decode_seqs.len(), 0);

        // Iteration 2: second chunk [512, 1024)
        let out2 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out2.prefill_seqs.len(), 1);
        assert_eq!(out2.prefill_seqs[0].prefill_progress, 1024);
        assert!(out2.prefill_seqs[0].is_chunked_prefill());
        assert_eq!(out2.prefill_chunk_range, Some((512, 1024)));

        // Iteration 3: third chunk [1024, 1536)
        let out3 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out3.prefill_seqs.len(), 1);
        assert_eq!(out3.prefill_seqs[0].prefill_progress, 1536);
        assert!(out3.prefill_seqs[0].is_chunked_prefill());
        assert_eq!(out3.prefill_chunk_range, Some((1024, 1536)));

        // Iteration 4: final chunk [1536, 2048) — prefill completes
        let out4 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out4.prefill_seqs.len(), 1);
        assert_eq!(out4.prefill_seqs[0].prefill_progress, 2048);
        assert!(!out4.prefill_seqs[0].is_chunked_prefill()); // complete
        assert_eq!(out4.prefill_chunk_range, Some((1536, 2048)));

        // Iteration 5: now in decode mode
        scheduler.update_sequence(1, &mut bm).unwrap();
        let out5 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out5.prefill_seqs.len(), 0);
        assert_eq!(out5.decode_seqs.len(), 1);
        assert!(out5.prefill_chunk_range.is_none());
    }

    #[test]
    fn test_chunked_prefill_interleaves_decode() {
        // 1 long prefill (prompt_len=1024, chunk=512) + 1 short decode request.
        // Decode should run between prefill chunks.
        let config = SchedulerConfig {
            max_batch_size: 4,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        let mut bm = test_block_manager(1024);

        // Add and prefill a short sequence first so it enters decode.
        scheduler.add_request(make_request(10, 8, 100));
        let out_init = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out_init.prefill_seqs.len(), 1);
        assert_eq!(out_init.prefill_seqs[0].seq_id, 10);
        // Simulate one token so it enters decode mode.
        scheduler.update_sequence(10, &mut bm).unwrap();

        // Now add the long prefill request.
        scheduler.add_request(make_request(20, 1024, 50));

        // Iteration: admits seq 20 with first chunk [0, 512).
        // seq 10 is already prefilled -> should appear in decode.
        let out1 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out1.prefill_seqs.len(), 1);
        assert_eq!(out1.prefill_seqs[0].seq_id, 20);
        assert_eq!(out1.prefill_chunk_range, Some((0, 512)));
        // Decode batch should include seq 10.
        assert_eq!(out1.decode_seqs.len(), 1);
        assert_eq!(out1.decode_seqs[0].seq_id, 10);

        // Simulate decode token for seq 10.
        scheduler.update_sequence(10, &mut bm).unwrap();

        // Iteration: continue chunk [512, 1024) for seq 20,
        // seq 10 still in decode.
        let out2 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out2.prefill_seqs.len(), 1);
        assert_eq!(out2.prefill_seqs[0].seq_id, 20);
        assert_eq!(out2.prefill_chunk_range, Some((512, 1024)));
        // Decode interleaved.
        assert_eq!(out2.decode_seqs.len(), 1);
        assert_eq!(out2.decode_seqs[0].seq_id, 10);
    }

    #[test]
    fn test_short_prompt_not_chunked() {
        // prompt_len=256, chunk=512 -> no chunking should occur.
        let config = SchedulerConfig {
            max_batch_size: 4,
            eos_token_id: 2,
            block_size: 4,
            max_prefill_chunk: 512,
        };
        let mut scheduler = Scheduler::new(config);
        let mut bm = test_block_manager(256);

        scheduler.add_request(make_request(1, 256, 10));

        let out = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out.prefill_seqs.len(), 1);
        assert_eq!(out.prefill_seqs[0].seq_id, 1);
        assert_eq!(out.prefill_seqs[0].prefill_progress, 256);
        assert!(!out.prefill_seqs[0].is_chunked_prefill()); // complete in one go
        assert!(out.prefill_chunk_range.is_none()); // not chunked

        // Next iteration should be decode.
        scheduler.update_sequence(1, &mut bm).unwrap();
        let out2 = scheduler.schedule(&mut bm, &HashMap::new());
        assert_eq!(out2.prefill_seqs.len(), 0);
        assert_eq!(out2.decode_seqs.len(), 1);
    }
}
