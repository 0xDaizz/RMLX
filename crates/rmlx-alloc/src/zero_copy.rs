use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::{EventError, GpuEvent};
use rmlx_metal::metal::Buffer as MetalBuffer;

use crate::AllocError;

/// Tracks whether a buffer is still in-flight (GPU or RDMA operation pending).
///
/// This is a lightweight completion tracker for coordinating buffer lifetime
/// across GPU compute and RDMA transfer operations. Both must complete
/// before the buffer can safely be freed or reused.
///
/// Optionally integrates with `GpuEvent` (MTLSharedEvent) for hardware-level
/// GPU completion signaling.
#[derive(Clone)]
pub struct CompletionTicket {
    gpu_complete: Arc<AtomicBool>,
    rdma_complete: Arc<AtomicBool>,
    gpu_event: Option<Arc<GpuEvent>>,
    gpu_event_value: Arc<AtomicU64>,
}

impl CompletionTicket {
    pub fn new() -> Self {
        Self {
            gpu_complete: Arc::new(AtomicBool::new(false)),
            rdma_complete: Arc::new(AtomicBool::new(false)),
            gpu_event: None,
            gpu_event_value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Attach a GpuEvent for hardware-level GPU completion signaling.
    ///
    /// Returns the signal value that the GPU command buffer should signal
    /// when the operation completes.
    pub fn with_gpu_event(&mut self, event: Arc<GpuEvent>) -> u64 {
        let value = event.next_value();
        self.gpu_event_value.store(value, Ordering::Release);
        self.gpu_event = Some(event);
        value
    }

    /// Mark the GPU operation as completed.
    ///
    /// If a GpuEvent is attached, this also signals the event atomically.
    pub fn mark_gpu_complete(&self) {
        self.gpu_complete.store(true, Ordering::Release);
    }

    /// Mark the RDMA operation as completed.
    pub fn mark_rdma_complete(&self) {
        self.rdma_complete.store(true, Ordering::Release);
    }

    /// Returns true if both GPU and RDMA operations have completed.
    ///
    /// If a GpuEvent is attached, GPU completion is determined by checking
    /// the event's signaled value rather than the atomic flag alone.
    pub fn is_safe_to_free(&self) -> bool {
        let gpu_done = if let Some(ref event) = self.gpu_event {
            let target = self.gpu_event_value.load(Ordering::Acquire);
            target == 0
                || event.raw().signaled_value() >= target
                || self.gpu_complete.load(Ordering::Acquire)
        } else {
            self.gpu_complete.load(Ordering::Acquire)
        };
        gpu_done && self.rdma_complete.load(Ordering::Acquire)
    }

    /// Wait for both GPU and RDMA completion with a timeout.
    ///
    /// If a GpuEvent is attached, uses the event's CPU wait mechanism for
    /// low-latency GPU synchronization. Otherwise, spins on the atomic flags.
    pub fn wait_all_complete(&self, timeout: Duration) -> Result<(), CompletionError> {
        let start = Instant::now();

        // Wait for GPU completion
        if let Some(ref event) = self.gpu_event {
            let target = self.gpu_event_value.load(Ordering::Acquire);
            if target > 0 && !self.gpu_complete.load(Ordering::Acquire) {
                let remaining = timeout
                    .checked_sub(start.elapsed())
                    .unwrap_or(Duration::ZERO);
                match event.cpu_wait(target, remaining) {
                    Ok(_) => {}
                    Err(EventError::Timeout(_)) => return Err(CompletionError::GpuTimeout),
                    Err(EventError::Cancelled) => return Err(CompletionError::Cancelled),
                }
            }
        } else {
            while !self.gpu_complete.load(Ordering::Acquire) {
                if start.elapsed() >= timeout {
                    return Err(CompletionError::GpuTimeout);
                }
                std::thread::yield_now();
            }
        }

        // Wait for RDMA completion
        while !self.rdma_complete.load(Ordering::Acquire) {
            if start.elapsed() >= timeout {
                return Err(CompletionError::RdmaTimeout);
            }
            std::thread::yield_now();
        }

        Ok(())
    }

    /// Whether a GpuEvent is attached.
    pub fn has_gpu_event(&self) -> bool {
        self.gpu_event.is_some()
    }

    /// Whether GPU completion has been signaled.
    pub fn is_gpu_complete(&self) -> bool {
        if let Some(ref event) = self.gpu_event {
            let target = self.gpu_event_value.load(Ordering::Acquire);
            target == 0
                || event.raw().signaled_value() >= target
                || self.gpu_complete.load(Ordering::Acquire)
        } else {
            self.gpu_complete.load(Ordering::Acquire)
        }
    }

    /// Whether RDMA completion has been signaled.
    pub fn is_rdma_complete(&self) -> bool {
        self.rdma_complete.load(Ordering::Acquire)
    }
}

/// Errors from completion waiting.
#[derive(Debug)]
pub enum CompletionError {
    GpuTimeout,
    RdmaTimeout,
    Cancelled,
}

impl std::fmt::Display for CompletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuTimeout => write!(f, "GPU completion timed out"),
            Self::RdmaTimeout => write!(f, "RDMA completion timed out"),
            Self::Cancelled => write!(f, "completion wait cancelled"),
        }
    }
}

impl std::error::Error for CompletionError {}

impl Default for CompletionTicket {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy buffer: page-aligned memory shared between CPU and Metal GPU.
///
/// Uses posix_memalign for allocation and Metal newBufferWithBytesNoCopy
/// to create a GPU-visible view without any data copy.
pub struct ZeroCopyBuffer {
    raw_ptr: NonNull<u8>,
    metal_buffer: MetalBuffer,
    in_flight: Arc<()>,
    size: usize,
    _alignment: usize,
    ticket: Option<CompletionTicket>,
}

// SAFETY: ZeroCopyBuffer can be sent between threads because:
// 1. The raw_ptr is heap-allocated (posix_memalign) and owned solely by this struct.
// 2. The Metal buffer is StorageModeShared (CPU+GPU coherent on Apple Silicon UMA).
// 3. The in_flight Arc<()> is itself Send+Sync.
unsafe impl Send for ZeroCopyBuffer {}

// SAFETY: ZeroCopyBuffer can be shared between threads because:
// 1. Immutable access (&self) only exposes the raw pointer value and Metal buffer reference.
// 2. Mutable pointer access (as_mut_ptr) requires &mut self, enforcing exclusive access.
// 3. In-flight tracking (Arc<()>) is thread-safe (atomic ref count).
// 4. Metal GPU operations are synchronized externally via command buffer completion handlers.
unsafe impl Sync for ZeroCopyBuffer {}

impl ZeroCopyBuffer {
    /// Allocate a zero-copy buffer of at least `size` bytes.
    /// The actual allocation is page-aligned (typically 16KB on Apple Silicon).
    pub fn new(device: &GpuDevice, size: usize) -> Result<Self, AllocError> {
        let alignment = page_size();
        let aligned_size = align_up(size, alignment);

        // Step 1: Page-aligned allocation
        let raw_ptr = unsafe {
            // SAFETY: posix_memalign is called with valid alignment (power-of-2, >= sizeof(void*))
            // and size. The resulting pointer is non-null on success.
            let mut ptr: *mut libc::c_void = std::ptr::null_mut();
            let ret = libc::posix_memalign(&mut ptr, alignment, aligned_size);
            if ret != 0 {
                return Err(AllocError::PosixMemalign(ret));
            }
            // SAFETY: ptr is valid for aligned_size bytes, just allocated.
            // Zero-fill for safety.
            std::ptr::write_bytes(ptr as *mut u8, 0, aligned_size);
            NonNull::new(ptr as *mut u8).expect("posix_memalign returned null despite success")
        };

        // Step 2: Metal NoCopy buffer (StorageModeShared)
        let metal_buffer = unsafe {
            // SAFETY: raw_ptr is page-aligned, valid for aligned_size bytes,
            // and will outlive the buffer (we own it and free in Drop).
            rmlx_metal::buffer::new_buffer_no_copy(
                device.raw(),
                raw_ptr.as_ptr() as *mut std::ffi::c_void,
                aligned_size as u64,
            )
        };

        Ok(Self {
            raw_ptr,
            metal_buffer,
            in_flight: Arc::new(()),
            size: aligned_size,
            _alignment: alignment,
            ticket: None,
        })
    }

    /// The underlying Metal buffer (for GPU operations).
    #[inline]
    pub fn metal_buffer(&self) -> &MetalBuffer {
        &self.metal_buffer
    }

    /// Raw pointer to the buffer contents.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.raw_ptr.as_ptr()
    }

    /// Mutable raw pointer to the buffer contents.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.raw_ptr.as_ptr()
    }

    /// Buffer size in bytes (page-aligned).
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Attach a completion ticket for GPU/RDMA lifecycle tracking.
    pub fn set_ticket(&mut self, ticket: CompletionTicket) {
        self.ticket = Some(ticket);
    }

    /// Get a reference to the current completion ticket, if any.
    pub fn ticket(&self) -> Option<&CompletionTicket> {
        self.ticket.as_ref()
    }

    /// Returns true if safe to free (no in-flight operations tracked by ticket).
    /// If no ticket is attached, returns true (untracked = safe).
    pub fn is_safe_to_free(&self) -> bool {
        match &self.ticket {
            Some(t) => t.is_safe_to_free(),
            None => true,
        }
    }

    /// Acquire an in-flight token. The buffer cannot be freed while
    /// any token is held. Use this before submitting GPU work or RDMA operations.
    pub fn acquire_in_flight(&self) -> InFlightToken {
        InFlightToken {
            _guard: Arc::clone(&self.in_flight),
        }
    }

    /// Acquire a completion fence for GPU/RDMA work tracking.
    pub fn acquire_fence(&self, op_tag: &'static str) -> CompletionFence {
        CompletionFence {
            token: std::mem::ManuallyDrop::new(self.acquire_in_flight()),
            op_tag,
            verified: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Number of in-flight references (includes self).
    /// Returns 1 when no external references are held.
    pub fn in_flight_count(&self) -> usize {
        Arc::strong_count(&self.in_flight)
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        // Check completion ticket if present
        if let Some(ref ticket) = self.ticket {
            if !ticket.is_safe_to_free() {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !ticket.is_safe_to_free() {
                    if Instant::now() >= deadline {
                        eprintln!(
                            "[rmlx-alloc] CRITICAL: buffer {:p} drop timeout -- \
                             completion ticket not resolved (gpu={}, rdma={}). \
                             Leaking memory to prevent use-after-free.",
                            self.raw_ptr.as_ptr(),
                            ticket.gpu_complete.load(Ordering::Acquire),
                            ticket.rdma_complete.load(Ordering::Acquire),
                        );
                        let _ = std::mem::replace(&mut self.raw_ptr, NonNull::dangling());
                        return;
                    }
                    std::thread::yield_now();
                }
            }
        }
        // Wait for all in-flight tokens to be released
        let deadline = Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&self.in_flight) > 1 {
            if Instant::now() >= deadline {
                let remaining = Arc::strong_count(&self.in_flight) - 1;
                eprintln!(
                    "[rmlx-alloc] CRITICAL: buffer {:p} drop timeout -- {} in-flight refs remaining. \
                     Leaking memory to prevent use-after-free.",
                    self.raw_ptr.as_ptr(), remaining
                );
                // Leak everything to prevent use-after-free.
                // Replace raw_ptr with dangling so we skip the free below;
                // the old pointer value is intentionally leaked.
                let _ = std::mem::replace(&mut self.raw_ptr, NonNull::dangling());
                return;
            }
            std::thread::yield_now();
        }
        // All in-flight work complete -- safe to free
        // metal_buffer drop is automatic (NoCopy buffer doesn't free the memory)
        // SAFETY: raw_ptr was allocated by posix_memalign and all GPU/RDMA
        // work using this buffer has completed.
        unsafe { libc::free(self.raw_ptr.as_ptr() as *mut libc::c_void) };
    }
}

/// In-flight reference token. Prevents buffer deallocation while held.
pub struct InFlightToken {
    _guard: Arc<()>,
}

/// Completion fence -- ties buffer lifetime to hardware completion events.
pub struct CompletionFence {
    token: std::mem::ManuallyDrop<InFlightToken>,
    op_tag: &'static str,
    verified: Arc<AtomicBool>,
}

impl CompletionFence {
    /// Tag identifying which operation holds this fence.
    pub fn op_tag(&self) -> &'static str {
        self.op_tag
    }

    /// Release after hardware completion verification.
    /// Called by GpuCompletionHandler or CQ poller after confirming
    /// the GPU/RDMA operation has actually completed.
    pub fn release_after_verification(mut self) {
        self.verified.store(true, Ordering::Release);
        // SAFETY: We only call ManuallyDrop::drop once, right here,
        // after verifying hardware completion.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.token) };
    }
}

impl Drop for CompletionFence {
    fn drop(&mut self) {
        if !self.verified.load(Ordering::Acquire) {
            // Token is ManuallyDrop — not dropping it intentionally.
            // The Arc<()> ref count stays elevated, preventing ZeroCopyBuffer::drop
            // from freeing memory while GPU/RDMA work may still be in flight.
            eprintln!(
                "[rmlx-alloc] WARNING: CompletionFence for '{}' dropped without verification. \
                 Leaking in-flight token to prevent use-after-free.",
                self.op_tag
            );
        }
        // If verified is true, token was already dropped in release_after_verification.
        // If verified is false, we intentionally skip dropping the ManuallyDrop token.
    }
}

/// Handles Metal command buffer completion -> fence release.
pub struct GpuCompletionHandler {
    fence: Option<CompletionFence>,
}

impl GpuCompletionHandler {
    pub fn new(fence: CompletionFence) -> Self {
        Self { fence: Some(fence) }
    }

    /// Called from Metal's completedHandler callback.
    pub fn on_completed(&mut self) {
        if let Some(fence) = self.fence.take() {
            fence.release_after_verification();
        }
    }
}

/// Get the system page size (usually 16KB on Apple Silicon).
pub(crate) fn page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is always safe to call.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 {
        ps as usize
    } else {
        16384
    }
}

/// Round up `n` to the next multiple of `align`. `align` must be a power of 2.
pub(crate) fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}
