# Running Your First Metal Compute Kernel

This tutorial walks you through performing a vector addition on the GPU using rmlx-metal, step by step.

---

## Overview

The full pipeline for executing a Metal compute kernel is as follows:

```
Acquire device -> Create buffers -> Compile shader -> Create pipeline -> Dispatch -> Verify results
```

The complete code for this tutorial can be found in `crates/rmlx-metal/tests/metal_compute.rs`.

---

## Step 1: Project Setup

Add the `rmlx-metal` dependency to your `Cargo.toml`:

```toml
[dependencies]
rmlx-metal = { path = "../rmlx/crates/rmlx-metal" }
```

Import the required modules:

```rust
use rmlx_metal::buffer::read_buffer;
use rmlx_metal::command::encode_compute_1d;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::library::compile_source;
use rmlx_metal::pipeline::PipelineCache;
use rmlx_metal::queue::GpuQueue;
```

---

## Step 2: Acquire the GPU Device

Acquire the system's default Metal GPU device:

```rust
let device = match GpuDevice::system_default() {
    Ok(d) => d,
    Err(_) => {
        eprintln!("skipping: no Metal device available");
        return;
    }
};
let queue = GpuQueue::new(&device);
```

- `GpuDevice::system_default()` returns the system's default Metal device.
- On Apple Silicon Macs, the integrated GPU is selected as the default device.
- On environments without a Metal device (VMs, Intel Macs, etc.), it returns an error.
- `GpuQueue` creates a command queue for submitting GPU commands.

---

## Step 3: Create Buffers

Create the input and output buffers needed for GPU computation:

```rust
// Input buffers: created with data
let buffer_a = device.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0]);
let buffer_b = device.new_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0]);

// Output buffer: empty buffer created with size only
let buffer_out = device.new_buffer(
    16, // 4 floats * 4 bytes = 16 bytes
    rmlx_metal::metal::MTLResourceOptions::StorageModeShared,
);
```

- `new_buffer_with_data` copies data from a Rust slice into a GPU buffer.
- `new_buffer` creates an empty buffer of the specified size.
- `StorageModeShared` is a mode where the CPU and GPU share the same memory.
  On Apple Silicon's UMA architecture, no actual data copy occurs.

---

## Step 4: Write and Compile the MSL Shader

Write a vector addition kernel in Metal Shading Language (MSL):

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
```

### MSL Attribute Reference

| Attribute | Meaning |
|-----------|---------|
| `kernel void` | Declares a compute function that runs on the GPU |
| `device const float *a` | Read-only buffer pointer in GPU device memory |
| `device float *out` | Writable buffer pointer in GPU device memory |
| `[[buffer(0)]]` | Maps to the buffer bound with `set_buffer(0, ...)` from Rust. Indices map in order: 0, 1, 2 |
| `[[thread_position_in_grid]]` | The global index of the current thread. The GPU automatically assigns a unique value to each thread. In this example, it will be 0, 1, 2, 3 |

JIT-compile this shader:

```rust
const VECTOR_ADD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
"#;

// JIT compilation: compiles MSL source at runtime
let library = compile_source(device.raw(), VECTOR_ADD_SOURCE)
    .expect("shader compilation");
```

> **Note**: `compile_source` calls `MTLDevice::newLibraryWithSource` to compile MSL source at runtime.
> It works without Xcode and produces the same result as using an AOT-compiled `.metallib` file.

---

## Step 5: Create the Pipeline

Find the kernel function from the compiled library and create a compute pipeline state:

```rust
let mut cache = PipelineCache::new(device.raw());
let pipeline = cache
    .get_or_create("vector_add_float", &library)
    .expect("pipeline creation");
```

- `PipelineCache` caches pipelines for the same kernel to prevent redundant creation.
- `get_or_create` finds the kernel function by name (`"vector_add_float"`) in the library and creates a pipeline.
- If a cached pipeline already exists, it returns the existing one.

---

## Step 6: Dispatch and Execute

Submit the kernel to the GPU and execute it:

```rust
// Create a command buffer
let cmd_buf = queue.new_command_buffer();

// Encode 1D compute: buffer binding + thread count
encode_compute_1d(
    cmd_buf,
    pipeline,
    &[(&buffer_a, 0), (&buffer_b, 0), (&buffer_out, 0)],
    4, // thread count = element count
);

// Submit to GPU and wait for completion
cmd_buf.commit();
cmd_buf.wait_until_completed();
```

- `encode_compute_1d` creates a compute command encoder, binds buffers, and dispatches threads in a single call.
- In the tuple `(&buffer_a, 0)`, the second value `0` is the offset (in bytes) within the buffer.
- The thread count `4` matches the number of elements to process. Each thread handles one element.
- `commit()` submits the command buffer to the GPU queue.
- `wait_until_completed()` blocks the current thread until GPU execution finishes.

---

## Step 7: Verify the Results

Read the GPU-computed results from the CPU:

```rust
// SAFETY: buffer_out uses StorageModeShared and GPU work has completed
// (wait_until_completed returned). We read exactly 4 floats = 16 bytes
// which matches the buffer allocation size.
let result: &[f32] = unsafe { read_buffer(&buffer_out, 4) };
assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
```

- `read_buffer` is an `unsafe` function. It interprets the GPU buffer contents as a Rust slice.
- Since `StorageModeShared` is used, the CPU and GPU share the same memory, so no separate data transfer is needed.
- It must be called after `wait_until_completed()` to ensure the GPU computation results are available.

**Result**: `[1.0+5.0, 2.0+6.0, 3.0+7.0, 4.0+8.0]` = `[6.0, 8.0, 10.0, 12.0]`

---

## Full Code

Below is the complete code combining all the steps above:

```rust
use rmlx_metal::buffer::read_buffer;
use rmlx_metal::command::encode_compute_1d;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::library::compile_source;
use rmlx_metal::pipeline::PipelineCache;
use rmlx_metal::queue::GpuQueue;

const VECTOR_ADD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
"#;

fn main() {
    // 1. Acquire GPU device
    let device = GpuDevice::system_default().expect("Metal device required");
    let queue = GpuQueue::new(&device);

    // 2. Create input/output buffers
    let buffer_a = device.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0]);
    let buffer_b = device.new_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0]);
    let buffer_out = device.new_buffer(
        16,
        rmlx_metal::metal::MTLResourceOptions::StorageModeShared,
    );

    // 3. JIT compile shader + create pipeline
    let library = compile_source(device.raw(), VECTOR_ADD_SOURCE)
        .expect("shader compilation");
    let mut cache = PipelineCache::new(device.raw());
    let pipeline = cache
        .get_or_create("vector_add_float", &library)
        .expect("pipeline creation");

    // 4. GPU dispatch
    let cmd_buf = queue.new_command_buffer();
    encode_compute_1d(
        cmd_buf,
        pipeline,
        &[(&buffer_a, 0), (&buffer_b, 0), (&buffer_out, 0)],
        4,
    );

    // 5. Execute and wait
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // 6. Verify results
    // SAFETY: StorageModeShared buffer, GPU work completed, reading 4 f32 = 16 bytes
    let result: &[f32] = unsafe { read_buffer(&buffer_out, 4) };
    println!("Result: {:?}", result); // [6.0, 8.0, 10.0, 12.0]
}
```

---

## Running the Test

The same code from this tutorial is included as an integration test:

```bash
# Run the integration test
cargo test -p rmlx-metal -- test_basic_metal_compute

# Run with output (to see println! output)
cargo test -p rmlx-metal -- test_basic_metal_compute --nocapture
```

---

## Next Steps

This tutorial covered the basics of rmlx's Metal compute pipeline.
For the full implementation plan, see the [Implementation Roadmap](../roadmap/phases.md).

- **Phase 2** adds 10 core GPU compute kernels including matmul, softmax, and RoPE
- **Phase 3** implements MTLSharedEvent-based synchronization and the dual queue pipeline
- **Phase 4** adds MoE kernels for Expert Parallelism
- [GPU Pipeline](../gpu-pipeline.md) — Learn how ExecGraph batches operations for 16.15x speedup
