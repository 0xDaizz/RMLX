//! Integration test: basic Metal compute kernel execution
//!
//! Verifies the full rmlx-metal pipeline: device acquisition, buffer creation,
//! JIT shader compilation, pipeline caching, compute dispatch, and result readback.

use objc2_metal::MTLCommandBuffer as _;
use rmlx_metal::buffer::read_buffer;
use rmlx_metal::command::encode_compute_1d;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::library::compile_source;
use rmlx_metal::pipeline::PipelineCache;
use rmlx_metal::queue::GpuQueue;

/// MSL source for a simple vector addition kernel.
const VECTOR_ADD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
"#;

#[test]
fn test_basic_metal_compute() {
    // 1. Get device — skip test on headless/virtualized hosts without Metal
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping test: no Metal device available");
            return;
        }
    };
    let queue = GpuQueue::new(&device);

    // 2. Create input/output buffers
    let buffer_a = device.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0]);
    let buffer_b = device.new_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0]);
    let buffer_out = device.new_buffer(
        16, // 4 floats * 4 bytes
        rmlx_metal::MTLResourceOptions::StorageModeShared,
    );

    // 3. Compile shader from source (JIT) and get pipeline
    let library = compile_source(device.raw(), VECTOR_ADD_SOURCE).expect("shader compilation");
    let cache = PipelineCache::new(device.raw());
    let pipeline = cache
        .get_or_create("vector_add_float", &library)
        .expect("pipeline creation");

    // 4. Encode and dispatch
    let cmd_buf = queue.new_command_buffer();
    encode_compute_1d(
        &cmd_buf,
        &pipeline,
        &[(&buffer_a, 0), (&buffer_b, 0), (&buffer_out, 0)],
        4,
    );

    // 5. Execute and wait
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // 6. Verify results
    // SAFETY: buffer_out uses StorageModeShared and GPU work has completed
    // (wait_until_completed returned). We read exactly 4 floats = 16 bytes
    // which matches the buffer allocation size.
    let result: &[f32] = unsafe { read_buffer(&buffer_out, 4) };
    assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
}
