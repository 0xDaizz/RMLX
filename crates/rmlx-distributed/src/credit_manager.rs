//! UC recv-credit window manager.
//!
//! In UC (Unreliable Connection) mode, a recv must be posted before the
//! matching send arrives — otherwise data is silently dropped. The
//! `CreditManager` maintains a per-(peer, tag) window of pre-posted recv
//! credits so that incoming sends always find a matching recv.

use rustc_hash::FxHashMap;

use rmlx_rdma::exchange_tag::ExchangeTag;

use crate::group::DistributedError;
use crate::transport::{RdmaConnectionTransport, RecvCredit};

/// Key for credit table: (peer_rank, tag).
type CreditKey = (u32, ExchangeTag);

/// A single pre-posted recv credit slot.
struct CreditSlot {
    /// The full RecvCredit — keeps buffer and MR alive until slot is consumed.
    credit: RecvCredit,
    /// Whether this slot has been consumed (completion arrived).
    consumed: bool,
}

/// Per-peer, per-tag recv credit window manager.
///
/// Ensures recv is always posted before send (UC safety). The manager
/// tracks a pool of pre-posted recv credits for each (peer, tag) pair
/// and provides methods to ensure minimums, consume completions, and
/// replenish consumed credits.
pub struct CreditManager {
    /// (peer_id, tag) -> Vec<CreditSlot>
    credits: FxHashMap<CreditKey, Vec<CreditSlot>>,
    /// Minimum credits to maintain per (peer, tag) pair.
    min_credits: usize,
    /// Buffer size for each recv credit (bytes).
    buf_size: usize,
}

impl CreditManager {
    /// Create a new credit manager.
    ///
    /// - `min_credits`: minimum recv credits to pre-post per (peer, tag) pair
    /// - `buf_size`: buffer size in bytes for each recv credit
    pub fn new(min_credits: usize, buf_size: usize) -> Self {
        Self {
            credits: FxHashMap::default(),
            min_credits,
            buf_size,
        }
    }

    /// Ensure at least `min_credits` recv credits are posted for (peer, tag).
    ///
    /// Posts additional recvs if the current active (non-consumed) count is
    /// below the minimum. This is the primary entry point — call it before
    /// any send to a peer on a given tag.
    ///
    /// # Safety
    /// Delegates to `RdmaConnectionTransport::pre_post_recv_credits`.
    /// MR lifetime is managed by RAII via `RecvCredit`.
    pub unsafe fn ensure_credits(
        &mut self,
        peer: u32,
        tag: ExchangeTag,
        transport: &RdmaConnectionTransport,
    ) -> Result<(), DistributedError> {
        let key = (peer, tag);
        let slots = self.credits.entry(key).or_default();

        // Count active (non-consumed) credits
        let active = slots.iter().filter(|s| !s.consumed).count();
        if active >= self.min_credits {
            return Ok(());
        }

        let needed = self.min_credits - active;

        // SAFETY: caller guarantees UC safety invariants; see fn-level doc.
        let new_credits =
            unsafe { transport.pre_post_recv_credits(peer, tag, self.buf_size, needed)? };

        for credit in new_credits {
            slots.push(CreditSlot {
                credit,
                consumed: false,
            });
        }

        Ok(())
    }

    /// Mark one credit consumed for (peer, tag) on CQ completion.
    ///
    /// Scans for the first non-consumed slot whose PendingOp has completed
    /// and marks it consumed. Call this when a recv completion is observed.
    pub fn on_completion(&mut self, peer: u32, tag: ExchangeTag) {
        let key = (peer, tag);
        if let Some(slots) = self.credits.get_mut(&key) {
            for slot in slots.iter_mut() {
                if !slot.consumed && !slot.credit.pending.is_pending() {
                    slot.consumed = true;
                    return;
                }
            }
        }
    }

    /// Replenish consumed credits for (peer, tag).
    ///
    /// Removes consumed slots and posts fresh recv credits to restore
    /// the count to `min_credits`.
    ///
    /// # Safety
    /// Same safety requirements as `ensure_credits`.
    pub unsafe fn replenish(
        &mut self,
        peer: u32,
        tag: ExchangeTag,
        transport: &RdmaConnectionTransport,
    ) -> Result<(), DistributedError> {
        let key = (peer, tag);
        if let Some(slots) = self.credits.get_mut(&key) {
            // Remove consumed slots
            slots.retain(|s| !s.consumed);
        }

        // ensure_credits will post the deficit
        // SAFETY: delegated to ensure_credits; see fn-level doc.
        unsafe { self.ensure_credits(peer, tag, transport) }
    }

    /// Number of active (non-consumed) credits for (peer, tag).
    pub fn available_credits(&self, peer: u32, tag: ExchangeTag) -> usize {
        let key = (peer, tag);
        self.credits
            .get(&key)
            .map(|slots| slots.iter().filter(|s| !s.consumed).count())
            .unwrap_or(0)
    }

    /// Total number of tracked slots (active + consumed) across all keys.
    pub fn total_slots(&self) -> usize {
        self.credits.values().map(|v| v.len()).sum()
    }

    /// Configured minimum credits per (peer, tag).
    pub fn min_credits(&self) -> usize {
        self.min_credits
    }

    /// Configured buffer size per credit.
    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manager_has_no_credits() {
        let cm = CreditManager::new(4, 4096);
        assert_eq!(cm.available_credits(0, ExchangeTag::Data), 0);
        assert_eq!(cm.total_slots(), 0);
        assert_eq!(cm.min_credits(), 4);
        assert_eq!(cm.buf_size(), 4096);
    }

    #[test]
    fn on_completion_noop_when_empty() {
        let mut cm = CreditManager::new(4, 4096);
        // Should not panic when no credits exist
        cm.on_completion(0, ExchangeTag::Data);
        assert_eq!(cm.available_credits(0, ExchangeTag::Data), 0);
    }

    #[test]
    fn different_keys_are_independent() {
        let cm = CreditManager::new(4, 4096);
        // Different (peer, tag) pairs have independent credit pools
        assert_eq!(cm.available_credits(0, ExchangeTag::Data), 0);
        assert_eq!(cm.available_credits(0, ExchangeTag::MoeDispatchCount), 0);
        assert_eq!(cm.available_credits(1, ExchangeTag::Data), 0);
    }
}
