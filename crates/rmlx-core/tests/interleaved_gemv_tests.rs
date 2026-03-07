//! Correctness tests for interleaved [M/TM, K, TM] layout GEMV kernels.
//! Each test verifies that the interleaved kernel produces the same result as
//! the row-major kernel by reformatting the weight matrix and comparing outputs.
//! Tests require Metal GPU — gracefully skip if unavailable.

use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::fused;
use rmlx_core::ops::gemv;

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
        v.push(f * 0.1);
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

/// Reformat [M, K] row-major weights to interleaved [M/TM, K, TM] layout.
/// TM=4: for each group of 4 rows, interleave K values.
fn reformat_to_interleaved(weights: &[f32], m: usize, k: usize) -> Vec<f32> {
    let tm = 4usize;
    assert_eq!(m % tm, 0, "M must be divisible by TM=4");
    let num_groups = m / tm;
    let mut out = vec![0.0f32; m * k];
    for group in 0..num_groups {
        for ki in 0..k {
            for r in 0..tm {
                out[group * k * tm + ki * tm + r] = weights[(group * tm + r) * k + ki];
            }
        }
    }
    out
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
// Test 1: gemv_bm8_f16 row vs gemv_bm8_f16_interleaved
// ===========================================================================

#[test]
fn test_gemv_bm8_f16_interleaved_matches_row_major() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };

    let m: usize = 4096;
    let k: usize = 4096;

    let w_f32 = rand_vec(m * k, 42);
    let x_f32 = rand_vec(k, 123);

    // Row-major GEMV
    let w_buf_row = make_f16_buf(&dev, &w_f32);
    let x_buf = make_f16_buf(&dev, &x_f32);
    let out_row = alloc_f16_out(&dev, m);

    let row_name = gemv::gemv_kernel_name(DType::Float16, m as u32).expect("row kernel name");
    let row_pso = registry.get_pipeline(row_name, DType::Float16).expect("row PSO");
    let (grid, tg) = gemv::gemv_dispatch_sizes(m as u32, &row_pso);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    gemv::gemv_preresolved_into_encoder(
        &row_pso, &w_buf_row, 0, &x_buf, 0, &out_row, 0,
        m as u32, k as u32, grid, tg, encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let row_result = read_f16_buf(&out_row, m);

    // Interleaved GEMV: reformat weight to [M/TM, K, TM]
    let w_interleaved_f32 = reformat_to_interleaved(&w_f32, m, k);
    let w_buf_il = make_f16_buf(&dev, &w_interleaved_f32);
    let out_il = alloc_f16_out(&dev, m);

    let il_name = gemv::gemv_interleaved_kernel_name(DType::Float16, m as u32).expect("interleaved kernel name");
    let il_pso = registry.get_pipeline(il_name, DType::Float16).expect("interleaved PSO");
    let (grid2, tg2) = gemv::gemv_dispatch_sizes(m as u32, &il_pso);

    let cb2 = queue.new_command_buffer();
    let encoder2 = cb2.new_compute_command_encoder();
    gemv::gemv_preresolved_into_encoder(
        &il_pso, &w_buf_il, 0, &x_buf, 0, &out_il, 0,
        m as u32, k as u32, grid2, tg2, encoder2,
    );
    encoder2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();
    let il_result = read_f16_buf(&out_il, m);

    assert_close(&row_result, &il_result, 1e-2, "gemv_bm8_f16_interleaved_vs_row");
}

// ===========================================================================
// Test 2: gemv_bm8_f32 row vs gemv_bm8_f32_interleaved
// ===========================================================================

#[test]
fn test_gemv_bm8_f32_interleaved_matches_row_major() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };

    let m: usize = 4096;
    let k: usize = 4096;

    let w_f32 = rand_vec(m * k, 42);
    let x_f32 = rand_vec(k, 123);

    // Row-major
    let w_buf_row = make_f32_buf(&dev, &w_f32);
    let x_buf = make_f32_buf(&dev, &x_f32);
    let out_row = alloc_f32_out(&dev, m);

    let row_name = gemv::gemv_kernel_name(DType::Float32, m as u32).expect("row kernel name");
    let row_pso = registry.get_pipeline(row_name, DType::Float32).expect("row PSO");
    let (grid, tg) = gemv::gemv_dispatch_sizes(m as u32, &row_pso);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    gemv::gemv_preresolved_into_encoder(
        &row_pso, &w_buf_row, 0, &x_buf, 0, &out_row, 0,
        m as u32, k as u32, grid, tg, encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let row_result = read_f32_buf(&out_row, m);

    // Interleaved
    let w_interleaved_f32 = reformat_to_interleaved(&w_f32, m, k);
    let w_buf_il = make_f32_buf(&dev, &w_interleaved_f32);
    let out_il = alloc_f32_out(&dev, m);

    let il_name = gemv::gemv_interleaved_kernel_name(DType::Float32, m as u32).expect("interleaved kernel name");
    let il_pso = registry.get_pipeline(il_name, DType::Float32).expect("interleaved PSO");
    let (grid2, tg2) = gemv::gemv_dispatch_sizes(m as u32, &il_pso);

    let cb2 = queue.new_command_buffer();
    let encoder2 = cb2.new_compute_command_encoder();
    gemv::gemv_preresolved_into_encoder(
        &il_pso, &w_buf_il, 0, &x_buf, 0, &out_il, 0,
        m as u32, k as u32, grid2, tg2, encoder2,
    );
    encoder2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();
    let il_result = read_f32_buf(&out_il, m);

    assert_close(&row_result, &il_result, 1e-5, "gemv_bm8_f32_interleaved_vs_row");
}

// ===========================================================================
// Test 3: gemv_bias_bm8_f16 row vs gemv_bias_bm8_f16_interleaved
// ===========================================================================

#[test]
fn test_gemv_bias_bm8_f16_interleaved_matches_row_major() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };

    let m: usize = 4096;
    let k: usize = 4096;

    let w_f32 = rand_vec(m * k, 42);
    let x_f32 = rand_vec(k, 123);
    let bias_f32 = rand_vec(m, 999);

    // Row-major with bias
    let w_buf_row = make_f16_buf(&dev, &w_f32);
    let x_buf = make_f16_buf(&dev, &x_f32);
    let bias_buf = make_f16_buf(&dev, &bias_f32);
    let out_row = alloc_f16_out(&dev, m);

    let row_name = gemv::gemv_bias_kernel_name(DType::Float16, m as u32).expect("row kernel name");
    let row_pso = registry.get_pipeline(row_name, DType::Float16).expect("row PSO");
    let (grid, tg) = gemv::gemv_dispatch_sizes(m as u32, &row_pso);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    gemv::gemv_bias_preresolved_into_encoder(
        &row_pso, &w_buf_row, 0, &x_buf, 0, &out_row, 0,
        m as u32, k as u32, &bias_buf, 0, grid, tg, encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let row_result = read_f16_buf(&out_row, m);

    // Interleaved with bias
    let w_interleaved_f32 = reformat_to_interleaved(&w_f32, m, k);
    let w_buf_il = make_f16_buf(&dev, &w_interleaved_f32);
    let out_il = alloc_f16_out(&dev, m);

    let il_name = gemv::gemv_bias_interleaved_kernel_name(DType::Float16, m as u32).expect("interleaved kernel name");
    let il_pso = registry.get_pipeline(il_name, DType::Float16).expect("interleaved PSO");
    let (grid2, tg2) = gemv::gemv_dispatch_sizes(m as u32, &il_pso);

    let cb2 = queue.new_command_buffer();
    let encoder2 = cb2.new_compute_command_encoder();
    gemv::gemv_bias_preresolved_into_encoder(
        &il_pso, &w_buf_il, 0, &x_buf, 0, &out_il, 0,
        m as u32, k as u32, &bias_buf, 0, grid2, tg2, encoder2,
    );
    encoder2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();
    let il_result = read_f16_buf(&out_il, m);

    assert_close(&row_result, &il_result, 1e-2, "gemv_bias_bm8_f16_interleaved_vs_row");
}

// ===========================================================================
// Test 4: fused_rms_gemv_bm8_f16 row vs fused_rms_gemv_bm8_f16_interleaved
// ===========================================================================

#[test]
fn test_fused_rms_gemv_bm8_f16_interleaved_matches_row_major() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };

    let m: usize = 4096;
    let k: usize = 4096;
    let eps: f32 = 1e-5;
    let w_stride: u32 = 1;

    let input_f32 = rand_vec(k, 77);
    let norm_w_f32 = rand_vec(k, 88);
    let mat_f32 = rand_vec(m * k, 55);

    // Row-major fused RMS+GEMV
    let input_buf = make_f16_buf(&dev, &input_f32);
    let norm_w_buf = make_f16_buf(&dev, &norm_w_f32);
    let mat_buf_row = make_f16_buf(&dev, &mat_f32);
    let out_row = alloc_f16_out(&dev, m);

    let row_name = fused::fused_rms_gemv_kernel_name(DType::Float16, m as u32).expect("row kernel");
    let row_pso = registry.get_pipeline(row_name, DType::Float16).expect("row PSO");
    let (grid, tg) = fused::fused_rms_gemv_dispatch_sizes(m as u32, &row_pso);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    fused::fused_rms_gemv_preresolved_into_encoder(
        &row_pso, &input_buf, 0, &norm_w_buf, 0, &mat_buf_row, 0,
        &out_row, 0, m as u32, k as u32, eps, w_stride, grid, tg, encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let row_result = read_f16_buf(&out_row, m);

    // Interleaved fused RMS+GEMV
    let mat_il_f32 = reformat_to_interleaved(&mat_f32, m, k);
    let mat_buf_il = make_f16_buf(&dev, &mat_il_f32);
    let out_il = alloc_f16_out(&dev, m);

    let il_name = fused::fused_rms_gemv_interleaved_kernel_name(DType::Float16).expect("interleaved kernel");
    let il_pso = registry.get_pipeline(il_name, DType::Float16).expect("interleaved PSO");
    let (grid2, tg2) = fused::fused_rms_gemv_dispatch_sizes(m as u32, &il_pso);

    let cb2 = queue.new_command_buffer();
    let encoder2 = cb2.new_compute_command_encoder();
    fused::fused_rms_gemv_preresolved_into_encoder(
        &il_pso, &input_buf, 0, &norm_w_buf, 0, &mat_buf_il, 0,
        &out_il, 0, m as u32, k as u32, eps, w_stride, grid2, tg2, encoder2,
    );
    encoder2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();
    let il_result = read_f16_buf(&out_il, m);

    assert_close(&row_result, &il_result, 1e-2, "fused_rms_gemv_bm8_f16_interleaved_vs_row");
}

// ===========================================================================
// Test 5: fused_swiglu_down_bm8_f16 row vs fused_swiglu_down_bm8_f16_interleaved
// ===========================================================================

#[test]
fn test_fused_swiglu_down_bm8_f16_interleaved_matches_row_major() {
    let Some((dev, queue, registry)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };

    let m: usize = 4096; // hidden dim
    let k: usize = 14336; // intermediate dim

    let down_w_f32 = rand_vec(m * k, 42);
    let gate_up_f32 = rand_vec(2 * k, 123);
    let bias_f32 = rand_vec(m, 999);

    // Row-major fused SwiGLU+down
    let mat_buf_row = make_f16_buf(&dev, &down_w_f32);
    let gate_up_buf = make_f16_buf(&dev, &gate_up_f32);
    let bias_buf = make_f16_buf(&dev, &bias_f32);
    let out_row = alloc_f16_out(&dev, m);

    let row_name = fused::fused_swiglu_down_kernel_name(DType::Float16, m as u32).expect("row kernel");
    let row_pso = registry.get_pipeline(row_name, DType::Float16).expect("row PSO");
    let (grid, tg) = fused::fused_swiglu_down_dispatch_sizes(m as u32, &row_pso);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    fused::fused_swiglu_down_preresolved_into_encoder(
        &row_pso, &mat_buf_row, 0, &gate_up_buf, 0, &out_row, 0,
        m as u32, k as u32, &bias_buf, 0, grid, tg, encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let row_result = read_f16_buf(&out_row, m);

    // Interleaved fused SwiGLU+down
    let down_w_il_f32 = reformat_to_interleaved(&down_w_f32, m, k);
    let mat_buf_il = make_f16_buf(&dev, &down_w_il_f32);
    let out_il = alloc_f16_out(&dev, m);

    let il_name = fused::fused_swiglu_down_interleaved_kernel_name(DType::Float16).expect("interleaved kernel");
    let il_pso = registry.get_pipeline(il_name, DType::Float16).expect("interleaved PSO");
    let (grid2, tg2) = fused::fused_swiglu_down_dispatch_sizes(m as u32, &il_pso);

    let cb2 = queue.new_command_buffer();
    let encoder2 = cb2.new_compute_command_encoder();
    fused::fused_swiglu_down_preresolved_into_encoder(
        &il_pso, &mat_buf_il, 0, &gate_up_buf, 0, &out_il, 0,
        m as u32, k as u32, &bias_buf, 0, grid2, tg2, encoder2,
    );
    encoder2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();
    let il_result = read_f16_buf(&out_il, m);

    assert_close(&row_result, &il_result, 1e-2, "fused_swiglu_down_bm8_f16_interleaved_vs_row");
}
