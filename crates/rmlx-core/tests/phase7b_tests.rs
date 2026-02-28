//! Phase 7B tests for rmlx-core: VJP and LoRA.

use rmlx_core::lora::{LoraConfig, LoraLayer, LoraTrainer, TrainConfig};
use rmlx_core::vjp::{self, AddGrad, MatMulGrad, MulGrad, Operation, Tape};

// ─── VJP tests ───

#[test]
fn test_tape_record_backward() {
    // f(x) = x * x (elementwise), df/dx = 2x
    let mut tape = Tape::new();
    let x_val = vec![3.0f32];
    let x = tape.leaf(x_val.clone());
    let x2 = tape.leaf(x_val.clone()); // second reference to same value

    let output_val = vec![9.0f32]; // 3*3
    let grad_fn = MulGrad {
        lhs: vec![3.0],
        rhs: vec![3.0],
    };
    let out = tape.record(&[&x, &x2], output_val, Operation::Mul, Box::new(grad_fn));

    let grads = tape.backward(&out);
    // df/dx = rhs * grad_out + lhs * grad_out = 3*1 + 3*1 = 6?
    // Actually: x and x2 are separate leaves. grad for x = rhs = 3.0, grad for x2 = lhs = 3.0
    // But conceptually f(x) = x*x, so we want 2x = 6.
    // With two separate leaves each gets 3.0.
    assert!((grads[0][0] - 3.0).abs() < 1e-5, "grad_x = {}", grads[0][0]);
    assert!(
        (grads[1][0] - 3.0).abs() < 1e-5,
        "grad_x2 = {}",
        grads[1][0]
    );
    // Sum = 6.0 = 2*3 = 2x, correct for x^2.
    assert!(
        (grads[0][0] + grads[1][0] - 6.0).abs() < 1e-5,
        "total grad = 2x"
    );
}

#[test]
fn test_vjp_add() {
    // f(a, b) = a + b, df/da = 1, df/db = 1
    let mut tape = Tape::new();
    let a = tape.leaf(vec![2.0f32, 3.0]);
    let b = tape.leaf(vec![4.0f32, 5.0]);

    let output = vec![6.0f32, 8.0];
    let grad_fn = AddGrad { len: 2 };
    let out = tape.record(&[&a, &b], output, Operation::Add, Box::new(grad_fn));

    let grads = tape.backward(&out);
    // df/da = [1, 1], df/db = [1, 1]
    assert_eq!(grads[0], vec![1.0, 1.0]);
    assert_eq!(grads[1], vec![1.0, 1.0]);
}

#[test]
fn test_vjp_mul() {
    // f(a, b) = a * b (elementwise)
    // df/da = b, df/db = a
    let mut tape = Tape::new();
    let a = tape.leaf(vec![2.0f32, 3.0]);
    let b = tape.leaf(vec![4.0f32, 5.0]);

    let output = vec![8.0f32, 15.0];
    let grad_fn = MulGrad {
        lhs: vec![2.0, 3.0],
        rhs: vec![4.0, 5.0],
    };
    let out = tape.record(&[&a, &b], output, Operation::Mul, Box::new(grad_fn));

    let grads = tape.backward(&out);
    // df/da = b = [4, 5], df/db = a = [2, 3]
    assert_eq!(grads[0], vec![4.0, 5.0]);
    assert_eq!(grads[1], vec![2.0, 3.0]);
}

#[test]
fn test_numerical_gradient() {
    // f(x) = [x[0]^2, x[1]^2], sum = x[0]^2 + x[1]^2
    // Compare VJP for elementwise mul against numerical gradient
    let x = vec![3.0f32, 4.0f32];

    let jacobian = vjp::numerical_gradient(|x| vec![x[0] * x[0], x[1] * x[1]], &x, 1e-3);
    // jacobian[j][i] = d(output_i)/d(input_j)
    // d(x0^2)/d(x0) = 6.0, d(x1^2)/d(x0) = 0.0
    // d(x0^2)/d(x1) = 0.0, d(x1^2)/d(x1) = 8.0
    assert!(
        (jacobian[0][0] - 6.0).abs() < 1e-2,
        "d(x0^2)/d(x0) = {}",
        jacobian[0][0]
    );
    assert!(
        jacobian[0][1].abs() < 1e-2,
        "d(x1^2)/d(x0) = {}",
        jacobian[0][1]
    );
    assert!(
        jacobian[1][0].abs() < 1e-2,
        "d(x0^2)/d(x1) = {}",
        jacobian[1][0]
    );
    assert!(
        (jacobian[1][1] - 8.0).abs() < 1e-2,
        "d(x1^2)/d(x1) = {}",
        jacobian[1][1]
    );
}

#[test]
fn test_vjp_matmul() {
    // C = A @ B, A: 2x3, B: 3x2
    let mut tape = Tape::new();
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2

    let a = tape.leaf(a_data.clone());
    let b = tape.leaf(b_data.clone());

    // C = [[4,5],[10,11]]
    let output = vec![4.0f32, 5.0, 10.0, 11.0];
    let grad_fn = MatMulGrad {
        a: a_data,
        b: b_data,
        m: 2,
        k: 3,
        n: 2,
    };
    let out = tape.record(
        &[&a, &b],
        output,
        Operation::MatMul { m: 2, k: 3, n: 2 },
        Box::new(grad_fn),
    );

    let grads = tape.backward(&out);
    // With grad_output = [[1,1],[1,1]]
    // dA = grad @ B^T = [[1,1],[1,1]] @ [[1,0,1],[0,1,1]] = [[1,1,2],[1,1,2]]
    assert_eq!(grads[0].len(), 6);
    assert!((grads[0][0] - 1.0).abs() < 1e-5); // dA[0,0]
    assert!((grads[0][1] - 1.0).abs() < 1e-5); // dA[0,1]
    assert!((grads[0][2] - 2.0).abs() < 1e-5); // dA[0,2]
                                               // dB = A^T @ grad = [[1,4],[2,5],[3,6]]^T ... = [[5,5],[7,7],[9,9]]
    assert_eq!(grads[1].len(), 6);
}

// ─── LoRA tests ───

#[test]
fn test_lora_config_defaults() {
    let config = LoraConfig::default();
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 16.0).abs() < 1e-10);
    assert!((config.dropout - 0.0).abs() < 1e-10);
    assert_eq!(config.target_modules, vec!["q_proj", "v_proj"]);
    assert!((config.scaling() - 2.0).abs() < 1e-10); // 16/8
}

#[test]
fn test_lora_layer_shape() {
    let config = LoraConfig::new(4, 16.0);
    let layer = LoraLayer::new(8, 16, &config);
    assert_eq!(layer.in_features, 8);
    assert_eq!(layer.out_features, 16);
    assert_eq!(layer.rank, 4);
    assert_eq!(layer.lora_a.len(), 4 * 8); // rank * in_features
    assert_eq!(layer.lora_b.len(), 16 * 4); // out_features * rank
    assert_eq!(layer.num_params(), 4 * 8 + 16 * 4); // 96
}

#[test]
fn test_lora_forward_zero_init() {
    // B is zero-initialized, so LoRA contribution should be zero
    let config = LoraConfig::new(4, 16.0);
    let layer = LoraLayer::new(8, 16, &config);

    let input = vec![1.0f32; 8]; // batch=1, in=8
    let base_output = vec![5.0f32; 16]; // batch=1, out=16

    let output = layer.forward(&base_output, &input, 1);
    // Should equal base_output since B=0
    for (i, (&got, &expected)) in output.iter().zip(base_output.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "output[{i}] = {got}, expected {expected}"
        );
    }
}

#[test]
fn test_lora_scaling() {
    let config = LoraConfig::new(8, 16.0);
    assert!((config.scaling() - 2.0).abs() < 1e-10);

    let config2 = LoraConfig::new(4, 8.0);
    assert!((config2.scaling() - 2.0).abs() < 1e-10);

    let config3 = LoraConfig::new(16, 16.0);
    assert!((config3.scaling() - 1.0).abs() < 1e-10);
}

#[test]
fn test_lora_train_step_loss_decreases() {
    // Simple scenario: 1 token, out_features=4 (vocab), target=2
    let in_features = 4;
    let out_features = 4;
    let rank = 2;
    let alpha = 4.0;

    // Non-zero A and B so there's something to learn
    let lora_a = vec![0.1f32; rank * in_features];
    let lora_b = vec![0.1f32; out_features * rank];

    let mut layer = LoraLayer::with_weights(in_features, out_features, rank, alpha, lora_a, lora_b);

    let input = vec![1.0f32, 0.5, -0.5, 0.2];
    let base_output = vec![0.0f32; out_features]; // zero base
    let targets = vec![2usize]; // target class index

    let trainer = LoraTrainer::new(TrainConfig {
        learning_rate: 0.1,
        num_epochs: 1,
        batch_size: 1,
    });

    // Compute loss before
    let logits_before = layer.forward(&base_output, &input, 1);
    let loss_before = LoraTrainer::compute_loss(&logits_before, &targets, out_features);

    // One training step
    let step_loss = trainer.train_step(&mut layer, &input, &base_output, &targets);
    assert!(
        (step_loss - loss_before).abs() < 1e-5,
        "step_loss should equal loss_before"
    );

    // Compute loss after
    let logits_after = layer.forward(&base_output, &input, 1);
    let loss_after = LoraTrainer::compute_loss(&logits_after, &targets, out_features);

    assert!(
        loss_after < loss_before,
        "loss should decrease: before={loss_before}, after={loss_after}"
    );
}
