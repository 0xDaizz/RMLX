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

// --- Negative tests ---

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
