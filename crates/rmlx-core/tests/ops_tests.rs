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
    let result: Vec<f32> = dst.to_vec_checked();
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
    let result: Vec<f32> = c.to_vec_checked();
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
    let result: Vec<f32> = c.to_vec_checked();
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
    let result: Vec<f32> = c.to_vec_checked();
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
    let result: Vec<f32> = c.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = c.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    // This should return Err due to inner dimension mismatch
    let result = ops::matmul::matmul(&registry, &a, &b, &queue);
    assert!(result.is_err(), "matmul should error on shape mismatch");
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
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
    let vals: Vec<f32> = result.to_vec_checked();
    // Sum of 1..=10000 = 10000 * 10001 / 2 = 50_005_000
    let expected = 50_005_000.0f32;
    assert!(
        (vals[0] - expected).abs() / expected < 1e-3,
        "medium sum: got {} expected {}",
        vals[0],
        expected
    );
}

// ─── RP0-5: rms_norm large axis_size tests ───

#[test]
fn test_rms_norm_large_sizes() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    for &axis_size in &[64, 512, 1024, 2048, 4096, 8192] {
        let input_data: Vec<f32> = (0..axis_size).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let weight_data: Vec<f32> = vec![1.0; axis_size];
        let input = Array::from_slice(dev, &input_data, vec![1, axis_size]);
        let weight = Array::from_slice(dev, &weight_data, vec![axis_size]);

        let result = ops::rms_norm::rms_norm(&registry, &input, &weight, 1e-5, &queue)
            .unwrap_or_else(|e| panic!("rms_norm failed for axis_size={axis_size}: {e}"));
        let vals: Vec<f32> = result.to_vec_checked();

        // Compute expected RMS
        let sum_sq: f32 = input_data.iter().map(|x| x * x).sum();
        let rms = (sum_sq / axis_size as f32 + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;

        // Check a few values
        for &idx in &[0, axis_size / 2, axis_size - 1] {
            let expected = input_data[idx] * inv_rms;
            let got = vals[idx];
            assert!(
                (got - expected).abs() < 1e-3,
                "rms_norm[{idx}] axis_size={axis_size}: got {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_rms_norm_multi_row_large() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let axis_size = 4096;
    let rows = 4;
    let input_data: Vec<f32> = (0..rows * axis_size)
        .map(|i| ((i % axis_size) as f32 + 1.0) * 0.001)
        .collect();
    let weight_data: Vec<f32> = vec![1.0; axis_size];
    let input = Array::from_slice(dev, &input_data, vec![rows, axis_size]);
    let weight = Array::from_slice(dev, &weight_data, vec![axis_size]);

    let result = ops::rms_norm::rms_norm(&registry, &input, &weight, 1e-5, &queue)
        .expect("multi-row rms_norm failed");
    let vals: Vec<f32> = result.to_vec_checked();

    // Each row should have its own normalization
    for r in 0..rows {
        let row_data = &input_data[r * axis_size..(r + 1) * axis_size];
        let sum_sq: f32 = row_data.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / axis_size as f32 + 1e-5).sqrt();
        let got = vals[r * axis_size];
        let expected = row_data[0] * inv_rms;
        assert!(
            (got - expected).abs() < 1e-3,
            "row {r} rms_norm[0]: got {got}, expected {expected}"
        );
    }
}

// ─── RP0-6: binary quantized returns proper error ───

#[test]
fn test_binary_quantized_returns_error() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Create Q8_0 arrays (32 elements = 1 block = 34 bytes)
    let a = Array::zeros(dev, &[32], DType::Q8_0);
    let b = Array::zeros(dev, &[32], DType::Q8_0);
    let result = ops::binary::binary_op(&registry, &a, &b, ops::binary::BinaryOp::Add, &queue);
    assert!(
        result.is_err(),
        "binary op on quantized types should return error, not panic"
    );
}

// ─── RP1-1: matmul GEMV auto-dispatch ───

#[test]
fn test_matmul_m1_n1_uses_gemv_path() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // [1,3] @ [3,1] = dot product
    let a = Array::from_slice(dev, &[1.0f32, 2.0, 3.0], vec![1, 3]);
    let b = Array::from_slice(dev, &[4.0f32, 5.0, 6.0], vec![3, 1]);
    let c = ops::matmul::matmul(&registry, &a, &b, &queue).expect("matmul M=1,N=1 failed");
    let vals: Vec<f32> = c.to_vec_checked();
    // 1*4 + 2*5 + 3*6 = 32
    assert!(
        (vals[0] - 32.0).abs() < 1e-3,
        "matmul M=1,N=1: got {}, expected 32.0",
        vals[0]
    );
    assert_eq!(c.shape(), &[1, 1]);
}

#[test]
fn test_matmul_m1_same_as_gemm() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // [1,4] @ [4,3] — this uses GEMM path since N>1
    let a = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let b = Array::from_slice(
        dev,
        &[
            1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![4, 3],
    );
    let c = ops::matmul::matmul(&registry, &a, &b, &queue).expect("matmul M=1 failed");
    let vals: Vec<f32> = c.to_vec_checked();
    // [1,2,3,4] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] = [1+4, 2+4, 3+4] = [5,6,7]
    assert!((vals[0] - 5.0).abs() < 1e-3, "C[0,0]={}", vals[0]);
    assert!((vals[1] - 6.0).abs() < 1e-3, "C[0,1]={}", vals[1]);
    assert!((vals[2] - 7.0).abs() < 1e-3, "C[0,2]={}", vals[2]);
}

#[test]
fn test_matmul_m1_large_n_uses_gemv() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Decode hot path: [1,K] @ [K,N] with N=128
    let k = 64;
    let n = 128;
    let a_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
    // Identity-like B: each column j has B[j,j]=1.0 for j<K, rest 0
    // Simpler: use known values
    let mut b_data = vec![0.0f32; k * n];
    for j in 0..k.min(n) {
        b_data[j * n + j] = 1.0;
    }
    let a = Array::from_slice(dev, &a_data, vec![1, k]);
    let b = Array::from_slice(dev, &b_data, vec![k, n]);
    let c = ops::matmul::matmul(&registry, &a, &b, &queue).expect("matmul M=1,N=128 failed");
    let vals: Vec<f32> = c.to_vec_checked();
    assert_eq!(c.shape(), &[1, n]);
    // First K columns should equal a_data, rest should be 0
    for j in 0..k {
        assert!(
            (vals[j] - a_data[j]).abs() < 1e-4,
            "C[0,{}]={}, expected {}",
            j,
            vals[j],
            a_data[j]
        );
    }
    for j in k..n {
        assert!(vals[j].abs() < 1e-4, "C[0,{}]={}, expected 0.0", j, vals[j]);
    }
}

#[test]
fn test_matmul_n1_uses_gemv() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // [M,K] @ [K,1] -> [M,1]
    let a = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Array::from_slice(dev, &[1.0f32, 0.0, 1.0], vec![3, 1]);
    let c = ops::matmul::matmul(&registry, &a, &b, &queue).expect("matmul N=1 failed");
    let vals: Vec<f32> = c.to_vec_checked();
    assert_eq!(c.shape(), &[2, 1]);
    // row 0: 1*1 + 2*0 + 3*1 = 4
    // row 1: 4*1 + 5*0 + 6*1 = 10
    assert!(
        (vals[0] - 4.0).abs() < 1e-4,
        "C[0,0]={}, expected 4.0",
        vals[0]
    );
    assert!(
        (vals[1] - 10.0).abs() < 1e-4,
        "C[1,0]={}, expected 10.0",
        vals[1]
    );
}

// ─── RP1-2: VJP backward tests ───

#[test]
fn test_vjp_softmax_backward() {
    use rmlx_core::vjp::*;
    // 1 row, 4 cols
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    // Compute softmax
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let output: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

    let grad_fn = SoftmaxGrad {
        output: output.clone(),
        rows: 1,
        cols: 4,
    };
    let grad_out = vec![1.0f32, 0.0, 0.0, 0.0]; // gradient w.r.t. first output
    let grads = grad_fn.backward(&grad_out).expect("backward failed");
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].len(), 4);
    // grad_input[0] = output[0] * (1 - output[0])
    let expected_0 = output[0] * (1.0 - output[0]);
    assert!(
        (grads[0][0] - expected_0).abs() < 1e-5,
        "softmax grad[0]: got {}, expected {}",
        grads[0][0],
        expected_0
    );
    // grad_input[1] = output[1] * (0 - output[0])
    let expected_1 = output[1] * (0.0 - output[0]);
    assert!(
        (grads[0][1] - expected_1).abs() < 1e-5,
        "softmax grad[1]: got {}, expected {}",
        grads[0][1],
        expected_1
    );
}

#[test]
fn test_vjp_rms_norm_backward() {
    use rmlx_core::vjp::*;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;
    let axis_size = 4;

    let grad_fn = RmsNormGrad {
        input: input.clone(),
        weight: weight.clone(),
        rows: 1,
        axis_size,
        eps,
    };

    // Compare with numerical gradient
    let rms_norm_fn = |x: &[f32]| -> Vec<f32> {
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
        x.iter()
            .zip(&weight)
            .map(|(v, w)| v * inv_rms * w)
            .collect()
    };

    let jacobian = numerical_gradient(rms_norm_fn, &input, 1e-3);
    let grad_out = vec![1.0f32, 0.0, 0.0, 0.0];
    let analytic = grad_fn.backward(&grad_out).expect("backward failed");

    // Compare: analytic grad should match sum of jacobian columns weighted by grad_out
    for i in 0..4 {
        let numerical: f32 = (0..4).map(|j| jacobian[i][j] * grad_out[j]).sum();
        assert!(
            (analytic[0][i] - numerical).abs() < 1e-3,
            "rms_norm grad[{i}]: analytic={}, numerical={}",
            analytic[0][i],
            numerical
        );
    }
}

// ─── RP1-3: LoRA tests ───

#[test]
fn test_lora_forward_shape() {
    use rmlx_core::lora::{LoraConfig, LoraLayer};
    let config = LoraConfig::new(4, 8.0).expect("LoraConfig::new failed");
    let layer = LoraLayer::new(16, 8, &config);
    let input = vec![1.0f32; 2 * 16]; // batch=2, in=16
    let base_output = vec![0.0f32; 2 * 8]; // batch=2, out=8
    let output = layer
        .forward(&base_output, &input, 2)
        .expect("forward failed");
    assert_eq!(
        output.len(),
        2 * 8,
        "output shape should be batch*out_features"
    );
}

#[test]
fn test_lora_zero_init_identity() {
    use rmlx_core::lora::{LoraConfig, LoraLayer};
    // B is zero-initialized, so LoRA contribution should be zero initially
    let config = LoraConfig::new(4, 8.0).expect("LoraConfig::new failed");
    let layer = LoraLayer::new(16, 8, &config);
    let input = vec![1.0f32; 16];
    let base_output = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let output = layer
        .forward(&base_output, &input, 1)
        .expect("forward failed");
    // Since B=0, output should equal base_output
    for (i, (&got, &expected)) in output.iter().zip(base_output.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-6,
            "lora zero init [{i}]: got {got}, expected {expected}"
        );
    }
}

#[test]
fn test_lora_gpu_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    use rmlx_core::lora::LoraLayer;
    let dev = registry.device().raw();

    // Small LoRA: in=4, out=3, rank=2, alpha=4
    let lora_a = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // (2, 4)
    let lora_b = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // (3, 2)
    let layer = LoraLayer::with_weights(4, 3, 2, 4.0, lora_a, lora_b).expect("with_weights failed");

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0]; // batch=1
    let base_data = vec![0.0f32, 0.0, 0.0]; // batch=1

    // CPU forward
    let cpu_out = layer
        .forward(&base_data, &input_data, 1)
        .expect("forward failed");

    // GPU forward
    let input_arr = Array::from_slice(dev, &input_data, vec![1, 4]);
    let base_arr = Array::from_slice(dev, &base_data, vec![1, 3]);
    let gpu_out = layer
        .forward_gpu(&base_arr, &input_arr, &registry, &queue)
        .expect("GPU LoRA failed");
    let gpu_vals: Vec<f32> = gpu_out.to_vec_checked();

    for (i, (&cpu, &gpu)) in cpu_out.iter().zip(gpu_vals.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-3,
            "LoRA CPU vs GPU [{i}]: cpu={cpu}, gpu={gpu}"
        );
    }
}

// ===== LaunchResult tests =====

#[test]
fn test_launch_result_binary_async() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let a = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
    let b = Array::from_slice(dev, &[10.0f32, 20.0, 30.0, 40.0], vec![4]);

    let launch =
        ops::binary::binary_op_async(&registry, &a, &b, ops::binary::BinaryOp::Add, &queue)
            .expect("binary_op_async failed");

    // Output should not be accessed until into_array()
    let output = launch.into_array();
    let vals: Vec<f32> = output.to_vec_checked();
    assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_launch_result_copy_async() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let data = vec![5.0f32, 6.0, 7.0, 8.0];
    let src = Array::from_slice(dev, &data, vec![4]);

    let launch = ops::copy::copy_async(&registry, &src, &queue).expect("copy_async failed");

    let output = launch.into_array();
    let vals: Vec<f32> = output.to_vec_checked();
    assert_eq!(vals, data);
}

#[test]
fn test_launch_result_reduce_async() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let arr = Array::from_slice(dev, &data, vec![4]);

    let launch = ops::reduce::reduce_all_async(&registry, &arr, ops::reduce::ReduceOp::Sum, &queue)
        .expect("reduce_all_async failed");

    let output = launch.into_array();
    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        (vals[0] - 10.0).abs() < 1e-3,
        "sum should be 10.0, got {}",
        vals[0]
    );
}

#[test]
fn test_launch_result_is_complete() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let a = Array::from_slice(dev, &[1.0f32; 4], vec![4]);
    let b = Array::from_slice(dev, &[2.0f32; 4], vec![4]);

    let launch =
        ops::binary::binary_op_async(&registry, &a, &b, ops::binary::BinaryOp::Mul, &queue)
            .expect("binary_op_async failed");

    // After into_array (which waits), is_complete should have been true
    let output = launch.into_array();
    let vals: Vec<f32> = output.to_vec_checked();
    assert_eq!(vals, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_launch_result_into_array_timeout_success() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let a = Array::from_slice(dev, &[3.0f32, 4.0], vec![2]);
    let b = Array::from_slice(dev, &[1.0f32, 1.0], vec![2]);

    let launch =
        ops::binary::binary_op_async(&registry, &a, &b, ops::binary::BinaryOp::Sub, &queue)
            .expect("binary_op_async failed");

    // Give generous timeout — GPU should finish well within
    let result = launch.into_array_timeout(std::time::Duration::from_secs(5));
    assert!(result.is_ok(), "should complete within timeout");
    let output = result.unwrap();
    let vals: Vec<f32> = output.to_vec_checked();
    assert_eq!(vals, vec![2.0, 3.0]);
}

// --- SiLU tests ---

/// CPU reference: silu(x) = x / (1 + exp(-x))
fn silu_ref(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn test_silu_f32_accuracy() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let input_data: Vec<f32> = vec![-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
    let expected: Vec<f32> = input_data.iter().map(|&x| silu_ref(x)).collect();

    let input = Array::from_slice(dev, &input_data, vec![input_data.len()]);
    let output = ops::silu::silu(&registry, &input, &queue).expect("silu f32 failed");
    let result: Vec<f32> = output.to_vec_checked();

    assert_eq!(result.len(), expected.len());
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        let abs_err = (got - exp).abs();
        let rtol = 1e-5;
        let atol = 1e-6;
        assert!(
            abs_err <= atol + rtol * exp.abs(),
            "silu f32 mismatch at [{}]: got={}, expected={}, err={}",
            i,
            got,
            exp,
            abs_err
        );
    }
}

#[test]
fn test_silu_f32_large_input() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Test with larger array to exercise threadgroup dispatch
    let n = 1024;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let expected: Vec<f32> = input_data.iter().map(|&x| silu_ref(x)).collect();

    let input = Array::from_slice(dev, &input_data, vec![n]);
    let output = ops::silu::silu(&registry, &input, &queue).expect("silu f32 large failed");
    let result: Vec<f32> = output.to_vec_checked();

    assert_eq!(result.len(), expected.len());
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        let abs_err = (got - exp).abs();
        let rtol = 1e-5;
        let atol = 1e-6;
        assert!(
            abs_err <= atol + rtol * exp.abs(),
            "silu f32 large mismatch at [{}]: input={}, got={}, expected={}, err={}",
            i,
            input_data[i],
            got,
            exp,
            abs_err
        );
    }
}

#[test]
fn test_silu_zero_and_symmetry() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // silu(0) = 0 exactly
    let input = Array::from_slice(dev, &[0.0f32], vec![1]);
    let output = ops::silu::silu(&registry, &input, &queue).expect("silu failed");
    let result: Vec<f32> = output.to_vec_checked();
    assert!(
        result[0].abs() < 1e-7,
        "silu(0) should be 0, got {}",
        result[0]
    );

    // silu(-x) != -silu(x) in general (not antisymmetric), but check known property:
    // silu(x) = x * sigmoid(x), and sigmoid(0)=0.5, so silu(0)=0
    // silu(large_positive) ≈ x, silu(large_negative) ≈ 0
    let input2 = Array::from_slice(dev, &[20.0f32, -20.0f32], vec![2]);
    let output2 = ops::silu::silu(&registry, &input2, &queue).expect("silu failed");
    let result2: Vec<f32> = output2.to_vec_checked();
    // silu(20) ≈ 20.0
    assert!(
        (result2[0] - 20.0).abs() < 1e-4,
        "silu(20) should be ~20, got {}",
        result2[0]
    );
    // silu(-20) ≈ 0.0
    assert!(
        result2[1].abs() < 1e-4,
        "silu(-20) should be ~0, got {}",
        result2[1]
    );
}
