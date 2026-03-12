//! Bump arena for per-forward-pass metadata allocations.
//!
//! `ForwardArena` wraps `bumpalo::Bump` to provide fast, allocation-free
//! intermediate storage that is bulk-freed at the start of each forward pass.

use bumpalo::Bump;

/// A bump-allocator arena scoped to a single forward pass.
///
/// Call [`reset`] at the start of each forward pass to reclaim all memory
/// without individual deallocations. The backing pages are retained for reuse.
pub struct ForwardArena {
    bump: Bump,
}

impl ForwardArena {
    /// Create a new arena with default initial capacity.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Create a new arena with the specified initial capacity in bytes.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bump: Bump::with_capacity(cap),
        }
    }

    /// Reset the arena, reclaiming all allocations. Backing memory is retained.
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Returns a reference to the inner `Bump` allocator for direct use.
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Allocate a value in the arena.
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.bump.alloc(val)
    }

    /// Allocate a slice by copying from an existing slice.
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(src)
    }

    /// Returns the total bytes allocated in this arena.
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for ForwardArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc_and_reset() {
        let mut arena = ForwardArena::with_capacity(1024);

        // Allocate some values
        let x = arena.alloc(42u32);
        assert_eq!(*x, 42);

        let slice = arena.alloc_slice_copy(&[1u32, 2, 3, 4]);
        assert_eq!(slice, &[1, 2, 3, 4]);

        assert!(arena.allocated_bytes() > 0);

        // Reset reclaims all
        arena.reset();
        // After reset, allocated_bytes returns 0 (or near-0)
        // but capacity is retained
    }

    #[test]
    fn test_arena_default() {
        let arena = ForwardArena::default();
        assert_eq!(arena.allocated_bytes(), 0);
    }
}
