//! Integration tests for rmlx-core ops.
//! Tests require Metal GPU — gracefully skip if unavailable.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

fn setup() -> Option<(KernelRegistry, metal::CommandQueue)> {
    let device = GpuDevice::system_default().ok()?;
    let queue = device.raw().new_command_queue();
    let registry = KernelRegistry::new(device);
    ops::register_all(&registry).ok()?;
    Some((registry, queue))
}

// --- Copy tests ---

#[test]
fn test_copy_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let src = Array::from_slice(registry.device().raw(), &data, vec![4]);
    let dst = ops::copy::copy(&registry, &src, &queue).expect("copy failed");
    let result: Vec<f32> = unsafe { dst.to_vec() };
    assert_eq!(result, data, "copy should be exact");
}

// --- Binary tests ---

#[test]
fn test_add_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let a = Array::from_slice(registry.device().raw(), &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
    let b = Array::from_slice(
        registry.device().raw(),
        &[10.0f32, 20.0, 30.0, 40.0],
        vec![4],
    );
    let c = ops::binary::add(&registry, &a, &b, &queue).expect("add failed");
    let result: Vec<f32> = unsafe { c.to_vec() };
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_mul_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let a = Array::from_slice(registry.device().raw(), &[2.0f32, 3.0, 4.0, 5.0], vec![4]);
    let b = Array::from_slice(
        registry.device().raw(),
        &[10.0f32, 10.0, 10.0, 10.0],
        vec![4],
    );
    let c = ops::binary::mul(&registry, &a, &b, &queue).expect("mul failed");
    let result: Vec<f32> = unsafe { c.to_vec() };
    assert_eq!(result, vec![20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn test_sub_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let a = Array::from_slice(
        registry.device().raw(),
        &[10.0f32, 20.0, 30.0, 40.0],
        vec![4],
    );
    let b = Array::from_slice(registry.device().raw(), &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
    let c = ops::binary::sub(&registry, &a, &b, &queue).expect("sub failed");
    let result: Vec<f32> = unsafe { c.to_vec() };
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_div_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let a = Array::from_slice(
        registry.device().raw(),
        &[10.0f32, 20.0, 30.0, 40.0],
        vec![4],
    );
    let b = Array::from_slice(registry.device().raw(), &[2.0f32, 4.0, 5.0, 8.0], vec![4]);
    let c = ops::binary::div(&registry, &a, &b, &queue).expect("div failed");
    let result: Vec<f32> = unsafe { c.to_vec() };
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
}

// --- Reduce tests ---

#[test]
fn test_reduce_sum_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_slice(registry.device().raw(), &data, vec![4]);
    let result = ops::reduce::sum(&registry, &input, &queue).expect("sum failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    let expected = 10.0f32;
    assert!(
        (vals[0] - expected).abs() < 1e-5,
        "sum: got {} expected {}",
        vals[0],
        expected
    );
}

#[test]
fn test_reduce_max_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.5];
    let input = Array::from_slice(registry.device().raw(), &data, vec![4]);
    let result = ops::reduce::max(&registry, &input, &queue).expect("max failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    assert_eq!(vals[0], 4.0, "max should be exact");
}

// --- Softmax test ---

#[test]
fn test_softmax_row_sum() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let input = Array::from_slice(registry.device().raw(), &data, vec![2, 4]);
    let result = ops::softmax::softmax(&registry, &input, &queue).expect("softmax failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    // Each row should sum to ~1.0
    let row0_sum: f32 = vals[0..4].iter().sum();
    let row1_sum: f32 = vals[4..8].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum = {row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum = {row1_sum}");
    // Values should be monotonically increasing within each row
    assert!(vals[0] < vals[1] && vals[1] < vals[2] && vals[2] < vals[3]);
}

// --- RMS Norm test ---

#[test]
fn test_rms_norm_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let input = Array::from_slice(registry.device().raw(), &input_data, vec![1, 4]);
    let weight = Array::from_slice(registry.device().raw(), &weight_data, vec![4]);
    let result =
        ops::rms_norm::rms_norm(&registry, &input, &weight, 1e-5, &queue).expect("rms_norm failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    // RMS = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5)
    let rms = (7.5f32 + 1e-5).sqrt();
    for (i, &v) in vals.iter().enumerate() {
        let expected = input_data[i] / rms;
        assert!(
            (v - expected).abs() < 1e-4,
            "rms_norm[{i}]: got {v}, expected {expected}"
        );
    }
}

// --- GEMV test ---

#[test]
fn test_gemv_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // 2x3 matrix * 3-vector = 2-vector
    // [[1,2,3],[4,5,6]] * [1,1,1] = [6, 15]
    let mat_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vec_data: Vec<f32> = vec![1.0, 1.0, 1.0];
    let mat = Array::from_slice(registry.device().raw(), &mat_data, vec![2, 3]);
    let v = Array::from_slice(registry.device().raw(), &vec_data, vec![3]);
    let result = ops::gemv::gemv(&registry, &mat, &v, &queue).expect("gemv failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    assert!((vals[0] - 6.0).abs() < 1e-3, "gemv[0]: got {}", vals[0]);
    assert!((vals[1] - 15.0).abs() < 1e-3, "gemv[1]: got {}", vals[1]);
}

// --- RoPE test ---

#[test]
fn test_rope_identity() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // With cos=1, sin=0, rope should be identity (scaled)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // seq=1, head_dim=4
    let cos_data: Vec<f32> = vec![1.0, 1.0]; // head_dim/2
    let sin_data: Vec<f32> = vec![0.0, 0.0];
    let input = Array::from_slice(registry.device().raw(), &input_data, vec![1, 4]);
    let cos_f = Array::from_slice(registry.device().raw(), &cos_data, vec![1, 2]);
    let sin_f = Array::from_slice(registry.device().raw(), &sin_data, vec![1, 2]);
    let result =
        ops::rope::rope(&registry, &input, &cos_f, &sin_f, 0, 1.0, &queue).expect("rope failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    for (i, (&got, &expected)) in vals.iter().zip(input_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "rope[{i}]: got {got}, expected {expected}"
        );
    }
}

// --- Quantized buffer size test ---

#[test]
fn test_zeros_q8_0_buffer_size() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    // Q8_0: 32 elements = 1 block = 34 bytes (2 bytes scale + 32 bytes data)
    let arr = Array::zeros(device.raw(), &[32], DType::Q8_0);
    assert_eq!(
        arr.byte_size(),
        34,
        "Q8_0 32 elements should be 34 bytes, not 32"
    );
    assert_eq!(
        arr.metal_buffer().length(),
        34,
        "Metal buffer should be 34 bytes"
    );

    // Q4_0: 32 elements = 1 block = 18 bytes
    let arr4 = Array::zeros(device.raw(), &[32], DType::Q4_0);
    assert_eq!(arr4.byte_size(), 18, "Q4_0 32 elements should be 18 bytes");
}

// --- Quantized matmul validation tests ---

#[test]
fn test_quantized_matmul_vec_size_mismatch() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let in_features: usize = 64;
    let out_features: usize = 32;
    // Create correctly sized Q8_0 weights buffer
    let weight_bytes = DType::Q8_0.numel_to_bytes(out_features * in_features);
    let weights = Array::new(
        dev.new_buffer(
            weight_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ),
        vec![out_features * in_features],
        vec![1],
        DType::Q8_0,
        0,
    );
    // Create vec with WRONG size (32 instead of 64)
    let wrong_vec = Array::from_slice(dev, &vec![0.0f32; 32], vec![32]);
    let result = ops::quantized::quantized_matmul(
        &registry,
        &weights,
        &wrong_vec,
        out_features,
        in_features,
        &queue,
    );
    assert!(result.is_err(), "should fail when vec size != in_features");
}

#[test]
fn test_quantized_matmul_weights_buffer_too_small() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let in_features: usize = 64;
    let out_features: usize = 32;
    // Create UNDERSIZED weights buffer (only enough for half the rows)
    let small_bytes = DType::Q8_0.numel_to_bytes((out_features / 2) * in_features);
    let weights = Array::new(
        dev.new_buffer(
            small_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ),
        vec![(out_features / 2) * in_features],
        vec![1],
        DType::Q8_0,
        0,
    );
    let input_vec = Array::from_slice(dev, &vec![0.0f32; in_features], vec![in_features]);
    let result = ops::quantized::quantized_matmul(
        &registry,
        &weights,
        &input_vec,
        out_features,
        in_features,
        &queue,
    );
    assert!(
        result.is_err(),
        "should fail when weights buffer is too small"
    );
}

// ─── GEMM tests ───

#[test]
fn test_matmul_f32() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // A: 2x3, B: 3x2 -> C: 2x2
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
    // C = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
    let a = Array::from_slice(
        registry.device().raw(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let b = Array::from_slice(
        registry.device().raw(),
        &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![3, 2],
    );
    let c = ops::matmul::matmul(&registry, &a, &b, &queue).expect("matmul failed");
    let vals: Vec<f32> = unsafe { c.to_vec() };
    assert!((vals[0] - 4.0).abs() < 1e-3, "C[0,0] = {}", vals[0]);
    assert!((vals[1] - 5.0).abs() < 1e-3, "C[0,1] = {}", vals[1]);
    assert!((vals[2] - 10.0).abs() < 1e-3, "C[1,0] = {}", vals[2]);
    assert!((vals[3] - 11.0).abs() < 1e-3, "C[1,1] = {}", vals[3]);
}

#[test]
fn test_matmul_square() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // Identity * anything = anything
    let eye = Array::from_slice(
        registry.device().raw(),
        &[1.0f32, 0.0, 0.0, 1.0],
        vec![2, 2],
    );
    let m = Array::from_slice(
        registry.device().raw(),
        &[5.0f32, 6.0, 7.0, 8.0],
        vec![2, 2],
    );
    let result = ops::matmul::matmul(&registry, &eye, &m, &queue).expect("matmul failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    assert!((vals[0] - 5.0).abs() < 1e-3);
    assert!((vals[1] - 6.0).abs() < 1e-3);
    assert!((vals[2] - 7.0).abs() < 1e-3);
    assert!((vals[3] - 8.0).abs() < 1e-3);
}

// ─── Negative tests ───

#[test]
fn test_matmul_shape_mismatch() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let a = Array::from_slice(registry.device().raw(), &[1.0f32, 2.0], vec![1, 2]);
    let b = Array::from_slice(registry.device().raw(), &[1.0f32, 2.0, 3.0], vec![1, 3]);
    // This should panic due to inner dimension mismatch
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ops::matmul::matmul(&registry, &a, &b, &queue)
    }));
    assert!(result.is_err(), "matmul should panic on shape mismatch");
}

#[test]
fn test_missing_kernel_error() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    // Empty registry -- no kernels registered
    let empty_reg = KernelRegistry::new(device);
    let result = empty_reg.get_pipeline("nonexistent_kernel", DType::Float32);
    assert!(result.is_err(), "should fail for missing kernel");
}

// --- bf16 binary ops tests ---

#[test]
fn test_add_bf16() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // bf16 values as raw u16 (IEEE bf16): 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040, 4.0=0x4080
    let a_data: Vec<u16> = vec![0x3F80, 0x4000, 0x4040, 0x4080];
    let b_data: Vec<u16> = vec![0x3F80, 0x3F80, 0x3F80, 0x3F80]; // all 1.0
    let a_buf = dev.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let b_buf = dev.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let a = Array::new(a_buf, vec![4], vec![1], DType::Bfloat16, 0);
    let b = Array::new(b_buf, vec![4], vec![1], DType::Bfloat16, 0);
    let c = ops::binary::add(&registry, &a, &b, &queue).expect("bf16 add failed");
    assert_eq!(c.dtype(), DType::Bfloat16);
    assert_eq!(c.shape(), &[4]);
    // Read raw bf16 output
    let out_ptr = c.metal_buffer().contents() as *const u16;
    let out_raw: Vec<u16> = unsafe { std::slice::from_raw_parts(out_ptr, 4).to_vec() };
    // 1+1=2.0=0x4000, 2+1=3.0=0x4040, 3+1=4.0=0x4080, 4+1=5.0=0x40A0
    assert_eq!(out_raw[0], 0x4000, "1.0+1.0 should be 2.0 (bf16 0x4000)");
    assert_eq!(out_raw[1], 0x4040, "2.0+1.0 should be 3.0 (bf16 0x4040)");
    assert_eq!(out_raw[2], 0x4080, "3.0+1.0 should be 4.0 (bf16 0x4080)");
    assert_eq!(out_raw[3], 0x40A0, "4.0+1.0 should be 5.0 (bf16 0x40A0)");
}

#[test]
fn test_mul_bf16() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // 2.0=0x4000, 3.0=0x4040
    let a_data: Vec<u16> = vec![0x4000, 0x4040]; // [2.0, 3.0]
    let b_data: Vec<u16> = vec![0x4000, 0x4000]; // [2.0, 2.0]
    let a_buf = dev.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let b_buf = dev.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let a = Array::new(a_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let b = Array::new(b_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let c = ops::binary::mul(&registry, &a, &b, &queue).expect("bf16 mul failed");
    let out_ptr = c.metal_buffer().contents() as *const u16;
    let out_raw: Vec<u16> = unsafe { std::slice::from_raw_parts(out_ptr, 2).to_vec() };
    // 2*2=4.0=0x4080, 3*2=6.0=0x40C0
    assert_eq!(out_raw[0], 0x4080, "2.0*2.0 should be 4.0");
    assert_eq!(out_raw[1], 0x40C0, "3.0*2.0 should be 6.0");
}

#[test]
fn test_sub_bf16() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // 4.0=0x4080, 2.0=0x4000
    let a_data: Vec<u16> = vec![0x4080, 0x4040]; // [4.0, 3.0]
    let b_data: Vec<u16> = vec![0x3F80, 0x3F80]; // [1.0, 1.0]
    let a_buf = dev.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let b_buf = dev.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let a = Array::new(a_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let b = Array::new(b_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let c = ops::binary::sub(&registry, &a, &b, &queue).expect("bf16 sub failed");
    let out_ptr = c.metal_buffer().contents() as *const u16;
    let out_raw: Vec<u16> = unsafe { std::slice::from_raw_parts(out_ptr, 2).to_vec() };
    // 4-1=3.0=0x4040, 3-1=2.0=0x4000
    assert_eq!(out_raw[0], 0x4040, "4.0-1.0 should be 3.0");
    assert_eq!(out_raw[1], 0x4000, "3.0-1.0 should be 2.0");
}

#[test]
fn test_div_bf16() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // 6.0=0x40C0, 4.0=0x4080
    let a_data: Vec<u16> = vec![0x40C0, 0x4080]; // [6.0, 4.0]
    let b_data: Vec<u16> = vec![0x4000, 0x4000]; // [2.0, 2.0]
    let a_buf = dev.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let b_buf = dev.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let a = Array::new(a_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let b = Array::new(b_buf, vec![2], vec![1], DType::Bfloat16, 0);
    let c = ops::binary::div(&registry, &a, &b, &queue).expect("bf16 div failed");
    let out_ptr = c.metal_buffer().contents() as *const u16;
    let out_raw: Vec<u16> = unsafe { std::slice::from_raw_parts(out_ptr, 2).to_vec() };
    // 6/2=3.0=0x4040, 4/2=2.0=0x4000
    assert_eq!(out_raw[0], 0x4040, "6.0/2.0 should be 3.0");
    assert_eq!(out_raw[1], 0x4000, "4.0/2.0 should be 2.0");
}

// --- bf16 copy test ---

#[test]
fn test_copy_bf16() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let data: Vec<u16> = vec![0x3F80, 0x4000, 0x4040, 0x4080]; // [1.0, 2.0, 3.0, 4.0]
    let buf = dev.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let src = Array::new(buf, vec![4], vec![1], DType::Bfloat16, 0);
    let dst = ops::copy::copy(&registry, &src, &queue).expect("bf16 copy failed");
    assert_eq!(dst.dtype(), DType::Bfloat16);
    let out_ptr = dst.metal_buffer().contents() as *const u16;
    let out_raw: Vec<u16> = unsafe { std::slice::from_raw_parts(out_ptr, 4).to_vec() };
    assert_eq!(out_raw, data, "bf16 copy should be exact");
}

// --- Two-pass reduce tests ---

#[test]
fn test_reduce_sum_large_array() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // 1M elements, all 1.0 => sum = 1_000_000
    let n = 1_000_000;
    let data: Vec<f32> = vec![1.0; n];
    let input = Array::from_slice(registry.device().raw(), &data, vec![n]);
    let result = ops::reduce::sum(&registry, &input, &queue).expect("large sum failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    let expected = n as f32;
    assert!(
        (vals[0] - expected).abs() / expected < 1e-3,
        "large sum: got {} expected {}",
        vals[0],
        expected
    );
}

#[test]
fn test_reduce_max_large_array() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // 1M elements: [0, 1, 2, ..., 999999]
    let n = 1_000_000;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = Array::from_slice(registry.device().raw(), &data, vec![n]);
    let result = ops::reduce::max(&registry, &input, &queue).expect("large max failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    let expected = (n - 1) as f32;
    assert_eq!(
        vals[0], expected,
        "large max: got {} expected {}",
        vals[0], expected
    );
}

#[test]
fn test_reduce_sum_medium_array() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    // 10K elements => triggers two-pass, sum should be accurate
    let n = 10_000;
    let data: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
    let input = Array::from_slice(registry.device().raw(), &data, vec![n]);
    let result = ops::reduce::sum(&registry, &input, &queue).expect("medium sum failed");
    let vals: Vec<f32> = unsafe { result.to_vec() };
    // Sum of 1..=10000 = 10000 * 10001 / 2 = 50_005_000
    let expected = 50_005_000.0f32;
    assert!(
        (vals[0] - expected).abs() / expected < 1e-3,
        "medium sum: got {} expected {}",
        vals[0],
        expected
    );
}
