//! Integration tests for fused kernels (SwiGLU+Down and RMS+GEMV).
//! Tests require Metal GPU — gracefully skip if unavailable.

use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::fused;

fn setup() -> Option<(metal::Device, metal::CommandQueue, KernelRegistry)> {
    let gpu = rmlx_metal::device::GpuDevice::system_default().ok()?;
    let dev = gpu.raw().clone();
    let queue = dev.new_command_queue();
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).ok()?;
    fused::register_fused_kernels(&registry).ok()?;
    Some((dev, queue, registry))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_f16_buf(buf: &metal::Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const u16;
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    slice
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect()
}

fn read_f32_buf(buf: &metal::Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        v.push(f * 0.1); // small values to avoid overflow
    }
    v
}

fn make_f16_buf(dev: &metal::Device, data: &[f32]) -> metal::Buffer {
    let f16_data: Vec<u16> = data
        .iter()
        .map(|&f| half::f16::from_f32(f).to_bits())
        .collect();
    dev.new_buffer_with_data(
        f16_data.as_ptr() as *const std::ffi::c_void,
        (f16_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn make_f32_buf(dev: &metal::Device, data: &[f32]) -> metal::Buffer {
    dev.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_f16_out(dev: &metal::Device, count: usize) -> metal::Buffer {
    dev.new_buffer(
        (count * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_f32_out(dev: &metal::Device, count: usize) -> metal::Buffer {
    dev.new_buffer(
        (count * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// CPU reference: SwiGLU(gate_up) then GEMV with bias.
fn cpu_swiglu_down(down_w: &[f32], gate_up: &[f32], bias: &[f32], m: usize, k: usize) -> Vec<f32> {
    // Step 1: SwiGLU
    let mut silu_out = vec![0.0f32; k];
    for i in 0..k {
        let gate = gate_up[i];
        let up = gate_up[k + i];
        let sigmoid = 1.0 / (1.0 + (-gate).exp());
        silu_out[i] = gate * sigmoid * up;
    }
    // Step 2: GEMV + bias
    let mut expected = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f64; // accumulate in f64 for reference accuracy
        for col in 0..k {
            acc += (down_w[row * k + col] * silu_out[col]) as f64;
        }
        expected[row] = acc as f32 + bias[row];
    }
    expected
}

/// CPU reference: RMS norm then GEMV.
fn cpu_rms_gemv(
    input: &[f32],
    norm_w: &[f32],
    mat: &[f32],
    m: usize,
    k: usize,
    eps: f32,
) -> Vec<f32> {
    // Step 1: RMS norm
    let ss: f32 = input.iter().map(|x| x * x).sum::<f32>() / k as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    let normed: Vec<f32> = (0..k).map(|i| input[i] * inv_rms * norm_w[i]).collect();
    // Step 2: GEMV
    let mut expected = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f64;
        for col in 0..k {
            acc += (mat[row * k + col] * normed[col]) as f64;
        }
        expected[row] = acc as f32;
    }
    expected
}

fn assert_close(expected: &[f32], actual: &[f32], tol: f32, label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
    let mut max_err: f32 = 0.0;
    let mut max_idx = 0;
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let err = (e - a).abs();
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }
    eprintln!(
        "{label}: max abs error = {max_err:.6e} at index {max_idx} (expected={:.6}, actual={:.6})",
        expected[max_idx], actual[max_idx]
    );
    assert!(
        max_err < tol,
        "{label}: max abs error {max_err:.6e} exceeds tolerance {tol:.6e} at index {max_idx}"
    );
}

// ===========================================================================
// Fusion B: fused_swiglu_down tests
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn run_fused_swiglu_down(
    dev: &metal::Device,
    queue: &metal::CommandQueue,
    registry: &KernelRegistry,
    m: usize,
    k: usize,
    dtype: DType,
    tol: f32,
    label: &str,
) {
    let down_w_f32 = rand_vec(m * k, 42);
    let gate_up_f32 = rand_vec(2 * k, 123);
    let bias_f32 = rand_vec(m, 999);

    let expected = cpu_swiglu_down(&down_w_f32, &gate_up_f32, &bias_f32, m, k);

    match dtype {
        DType::Float16 => {
            let mat_buf = make_f16_buf(dev, &down_w_f32);
            let gate_up_buf = make_f16_buf(dev, &gate_up_f32);
            let bias_buf = make_f16_buf(dev, &bias_f32);
            let out_buf = alloc_f16_out(dev, m);

            let kernel_name =
                fused::fused_swiglu_down_kernel_name(dtype, m as u32).expect("kernel name");
            let pso = registry.get_pipeline(kernel_name, dtype).expect("get PSO");
            let (grid, tg) = fused::fused_swiglu_down_dispatch_sizes(m as u32, &pso);

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            fused::fused_swiglu_down_preresolved_into_encoder(
                &pso,
                &mat_buf,
                0,
                &gate_up_buf,
                0,
                &out_buf,
                0,
                m as u32,
                k as u32,
                &bias_buf,
                0,
                grid,
                tg,
                encoder,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let actual = read_f16_buf(&out_buf, m);
            assert_close(&expected, &actual, tol, label);
        }
        DType::Float32 => {
            let mat_buf = make_f32_buf(dev, &down_w_f32);
            let gate_up_buf = make_f32_buf(dev, &gate_up_f32);
            let bias_buf = make_f32_buf(dev, &bias_f32);
            let out_buf = alloc_f32_out(dev, m);

            let kernel_name =
                fused::fused_swiglu_down_kernel_name(dtype, m as u32).expect("kernel name");
            let pso = registry.get_pipeline(kernel_name, dtype).expect("get PSO");
            let (grid, tg) = fused::fused_swiglu_down_dispatch_sizes(m as u32, &pso);

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            fused::fused_swiglu_down_preresolved_into_encoder(
                &pso,
                &mat_buf,
                0,
                &gate_up_buf,
                0,
                &out_buf,
                0,
                m as u32,
                k as u32,
                &bias_buf,
                0,
                grid,
                tg,
                encoder,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let actual = read_f32_buf(&out_buf, m);
            assert_close(&expected, &actual, tol, label);
        }
        _ => panic!("unsupported dtype for test: {:?}", dtype),
    }
}

#[test]
fn test_fused_swiglu_down_f16_matches_sequential() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_swiglu_down(
        &dev,
        &queue,
        &registry,
        4096,
        11008,
        DType::Float16,
        5e-2,
        "swiglu_down_f16_4096x11008",
    );
}

#[test]
fn test_fused_swiglu_down_f32_matches_sequential() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_swiglu_down(
        &dev,
        &queue,
        &registry,
        4096,
        11008,
        DType::Float32,
        1e-3,
        "swiglu_down_f32_4096x11008",
    );
}

#[test]
fn test_fused_swiglu_down_small_m() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_swiglu_down(
        &dev,
        &queue,
        &registry,
        64,
        128,
        DType::Float16,
        1e-2,
        "swiglu_down_f16_small_64x128",
    );
}

#[test]
fn test_fused_swiglu_down_edge_k_not_multiple_of_4() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_swiglu_down(
        &dev,
        &queue,
        &registry,
        256,
        13,
        DType::Float16,
        1e-2,
        "swiglu_down_f16_edge_256x13",
    );
}

// ===========================================================================
// Fusion A: fused_rms_gemv tests
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn run_fused_rms_gemv(
    dev: &metal::Device,
    queue: &metal::CommandQueue,
    registry: &KernelRegistry,
    m: usize,
    k: usize,
    dtype: DType,
    tol: f32,
    label: &str,
) {
    let eps: f32 = 1e-5;
    let w_stride: u32 = 1;

    let input_f32 = rand_vec(k, 77);
    let norm_w_f32 = rand_vec(k, 88);
    let mat_f32 = rand_vec(m * k, 55);

    let expected = cpu_rms_gemv(&input_f32, &norm_w_f32, &mat_f32, m, k, eps);

    match dtype {
        DType::Float16 => {
            let input_buf = make_f16_buf(dev, &input_f32);
            let norm_w_buf = make_f16_buf(dev, &norm_w_f32);
            let mat_buf = make_f16_buf(dev, &mat_f32);
            let out_buf = alloc_f16_out(dev, m);

            let kernel_name =
                fused::fused_rms_gemv_kernel_name(dtype, m as u32).expect("kernel name");
            let pso = registry.get_pipeline(kernel_name, dtype).expect("get PSO");
            let (grid, tg) = fused::fused_rms_gemv_dispatch_sizes(m as u32, &pso);

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            fused::fused_rms_gemv_preresolved_into_encoder(
                &pso,
                &input_buf,
                0,
                &norm_w_buf,
                0,
                &mat_buf,
                0,
                &out_buf,
                0,
                m as u32,
                k as u32,
                eps,
                w_stride,
                grid,
                tg,
                encoder,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let actual = read_f16_buf(&out_buf, m);
            assert_close(&expected, &actual, tol, label);
        }
        DType::Float32 => {
            let input_buf = make_f32_buf(dev, &input_f32);
            let norm_w_buf = make_f32_buf(dev, &norm_w_f32);
            let mat_buf = make_f32_buf(dev, &mat_f32);
            let out_buf = alloc_f32_out(dev, m);

            let kernel_name =
                fused::fused_rms_gemv_kernel_name(dtype, m as u32).expect("kernel name");
            let pso = registry.get_pipeline(kernel_name, dtype).expect("get PSO");
            let (grid, tg) = fused::fused_rms_gemv_dispatch_sizes(m as u32, &pso);

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            fused::fused_rms_gemv_preresolved_into_encoder(
                &pso,
                &input_buf,
                0,
                &norm_w_buf,
                0,
                &mat_buf,
                0,
                &out_buf,
                0,
                m as u32,
                k as u32,
                eps,
                w_stride,
                grid,
                tg,
                encoder,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let actual = read_f32_buf(&out_buf, m);
            assert_close(&expected, &actual, tol, label);
        }
        _ => panic!("unsupported dtype for test: {:?}", dtype),
    }
}

#[test]
fn test_fused_rms_gemv_f16_qkv_shape() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_rms_gemv(
        &dev,
        &queue,
        &registry,
        6144,
        4096,
        DType::Float16,
        5e-2,
        "rms_gemv_f16_qkv_6144x4096",
    );
}

#[test]
fn test_fused_rms_gemv_f16_gateup_shape() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_rms_gemv(
        &dev,
        &queue,
        &registry,
        22016,
        4096,
        DType::Float16,
        5e-2,
        "rms_gemv_f16_gateup_22016x4096",
    );
}

#[test]
fn test_fused_rms_gemv_f32_matches_sequential() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_rms_gemv(
        &dev,
        &queue,
        &registry,
        6144,
        4096,
        DType::Float32,
        1e-3,
        "rms_gemv_f32_6144x4096",
    );
}

#[test]
fn test_fused_rms_gemv_small_m() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_rms_gemv(
        &dev,
        &queue,
        &registry,
        64,
        128,
        DType::Float16,
        1e-2,
        "rms_gemv_f16_small_64x128",
    );
}

#[test]
fn test_fused_rms_gemv_edge_cases() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    run_fused_rms_gemv(
        &dev,
        &queue,
        &registry,
        256,
        17,
        DType::Float16,
        2e-2,
        "rms_gemv_f16_edge_256x17",
    );
}
