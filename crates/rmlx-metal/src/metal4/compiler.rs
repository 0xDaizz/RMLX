//! Async pipeline state compilation via Metal 4 `MTL4Compiler`.
//!
//! [`AsyncCompiler`] wraps `MTL4Compiler` to provide:
//! - Synchronous library compilation from MSL source
//! - Synchronous compute pipeline state creation
//! - Async compute PSO creation (returns a pollable [`CompileTask`])
//!
//! This replaces the Metal 3 pattern of `device.newLibrary(source:)` +
//! `device.newComputePipelineState(function:)` with a unified compiler
//! object that can also perform async compilation.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::MetalError;

// Type aliases local to this module.
type MtlLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type MtlPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A wrapper around `MTL4Compiler` for synchronous and asynchronous PSO creation.
///
/// Create via [`AsyncCompiler::new`] which obtains a compiler from the device.
/// The compiler can then create libraries (from MSL source) and compute
/// pipeline states, either synchronously or asynchronously.
pub struct AsyncCompiler {
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
}

impl AsyncCompiler {
    /// Create a new `AsyncCompiler` from a Metal device.
    ///
    /// Calls `[MTLDevice newCompilerWithDescriptor:error:]` to obtain the
    /// underlying `MTL4Compiler` instance.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, MetalError> {
        let desc = MTL4CompilerDescriptor::new();
        let compiler = device.newCompilerWithDescriptor_error(&desc).map_err(|e| {
            MetalError::PipelineCreate(format!(
                "failed to create MTL4Compiler: {}",
                e.localizedDescription()
            ))
        })?;
        Ok(Self { compiler })
    }

    /// Compile an MSL source string into a Metal library (synchronous).
    pub fn compile_library(&self, source: &str) -> Result<MtlLibrary, MetalError> {
        let desc = MTL4LibraryDescriptor::new();
        desc.setSource(Some(&NSString::from_str(source)));
        self.compiler
            .newLibraryWithDescriptor_error(&desc)
            .map_err(|e| {
                MetalError::ShaderCompile(format!(
                    "MTL4Compiler library compilation failed: {}",
                    e.localizedDescription()
                ))
            })
    }

    /// Create a compute pipeline state synchronously from a library + function name.
    ///
    /// This is the simplest path: compile source -> get function -> create PSO.
    pub fn compile_pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Result<MtlPipeline, MetalError> {
        let pipeline_desc = MTL4ComputePipelineDescriptor::new();

        // Create a library function descriptor pointing to the function.
        let func_desc = MTL4LibraryFunctionDescriptor::new();
        func_desc.setName(Some(&NSString::from_str(function_name)));
        func_desc.setLibrary(Some(library));

        // MTL4LibraryFunctionDescriptor is a subclass of MTL4FunctionDescriptor,
        // so we can pass it directly.
        pipeline_desc.setComputeFunctionDescriptor(Some(&func_desc));

        self.compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipeline_desc, None)
            .map_err(|e| {
                MetalError::PipelineCreate(format!(
                    "MTL4Compiler PSO creation failed for '{}': {}",
                    function_name,
                    e.localizedDescription()
                ))
            })
    }

    /// Compile MSL source and create a compute PSO in one step (synchronous).
    pub fn compile_source_to_pipeline(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<MtlPipeline, MetalError> {
        let library = self.compile_library(source)?;
        self.compile_pipeline(&library, function_name)
    }

    /// Access the raw `MTL4Compiler` protocol object.
    #[inline(always)]
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4Compiler> {
        &self.compiler
    }
}

/// A handle to an in-flight or completed asynchronous compilation task.
///
/// Returned by async compilation methods. Provides polling (`is_complete`)
/// and blocking (`wait`) interfaces, plus access to the resulting PSO via
/// [`pipeline`](CompileTask::pipeline).
///
/// Two variants exist:
/// - **Async** — wraps a real `MTL4CompilerTask` from the driver.
/// - **Completed** — holds a pre-built PSO (used when the synchronous Metal 4
///   compilation path is invoked through the async API surface).
pub enum CompileTask {
    /// A real in-flight compiler task from `MTL4Compiler`.
    Async {
        task: Retained<ProtocolObject<dyn MTL4CompilerTask>>,
    },
    /// A pre-completed task holding the finished PSO.
    Completed { pso: MtlPipeline },
}

impl CompileTask {
    /// Wrap a raw `MTL4CompilerTask`.
    #[allow(dead_code)] // Will be used when the callback-based async API is exposed.
    pub(crate) fn new(task: Retained<ProtocolObject<dyn MTL4CompilerTask>>) -> Self {
        Self::Async { task }
    }

    /// Create a pre-completed task that already holds the compiled PSO.
    ///
    /// Used when the Metal 4 synchronous compilation path is accessed through
    /// the async API surface (e.g., `DiskPipelineCache::compile_pipeline_async`).
    pub(crate) fn new_completed(pso: MtlPipeline) -> Self {
        Self::Completed { pso }
    }

    /// Block the calling thread until the compilation completes.
    ///
    /// No-op for pre-completed tasks.
    pub fn wait(&self) {
        if let Self::Async { task } = self {
            task.waitUntilCompleted();
        }
    }

    /// Check whether the compilation has finished (non-blocking).
    pub fn is_complete(&self) -> bool {
        match self {
            Self::Async { task } => task.status() == MTL4CompilerTaskStatus::Finished,
            Self::Completed { .. } => true,
        }
    }

    /// Query the current status of the compile task.
    pub fn status(&self) -> MTL4CompilerTaskStatus {
        match self {
            Self::Async { task } => task.status(),
            Self::Completed { .. } => MTL4CompilerTaskStatus::Finished,
        }
    }

    /// Return a reference to the compiled PSO, if available.
    ///
    /// For `Completed` tasks this always returns `Some`. For `Async` tasks
    /// this returns `None` — the PSO must be extracted from the driver
    /// result after `wait()` completes.
    pub fn pipeline(&self) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        match self {
            Self::Completed { pso } => Some(pso),
            // TODO(metal4): When MTL4CompilerTask provides a result/output accessor,
            // implement PSO extraction here. Currently the objc2-metal 0.3.2 binding
            // for MTL4CompilerTask only exposes `compiler()`, `status()`, and
            // `waitUntilCompleted()` — no method to retrieve the compiled PSO.
            //
            // The intended flow: after `wait()` succeeds (status == Finished), call
            // the result accessor to obtain the PSO and transition this variant from
            // Async to Completed (or cache the PSO internally). Until that API is
            // available in the bindings, callers must use the synchronous
            // `AsyncCompiler::compile_pipeline()` path which returns a
            // `CompileTask::Completed` directly.
            Self::Async { .. } => None,
        }
    }

    /// Access the raw `MTL4CompilerTask` protocol object.
    ///
    /// Returns `None` for pre-completed tasks.
    #[inline(always)]
    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTL4CompilerTask>> {
        match self {
            Self::Async { task } => Some(task),
            Self::Completed { .. } => None,
        }
    }
}
