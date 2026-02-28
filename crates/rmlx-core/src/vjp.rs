//! Value and Jacobian Product (VJP) framework for reverse-mode autodiff.

/// Operations that can be recorded on the tape.
#[derive(Debug, Clone)]
pub enum Operation {
    Add,
    Mul,
    MatMul { m: usize, k: usize, n: usize },
    Softmax,
    RmsNorm,
    Rope,
    Gemv,
    Reduce,
    Custom(String),
}

/// Errors that can occur during VJP backward pass.
#[derive(Debug, Clone)]
pub enum VjpError {
    /// The operation does not have a backward implementation.
    UnsupportedBackward(String),
}

impl std::fmt::Display for VjpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VjpError::UnsupportedBackward(op) => {
                write!(f, "backward not implemented for {op}")
            }
        }
    }
}

impl std::error::Error for VjpError {}

/// Trait for computing backward gradients.
pub trait GradFn {
    /// Given the gradient of the output, return gradients for each input.
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError>;
}

/// A recorded node on the tape.
struct TapeEntry {
    /// Indices of input values in the tape's value store.
    input_indices: Vec<usize>,
    /// The operation performed.
    _operation: Operation,
    /// Gradient function for this operation.
    grad_fn: Box<dyn GradFn>,
    /// Number of output elements.
    output_len: usize,
}

/// A tape that records operations for reverse-mode autodiff.
pub struct Tape {
    entries: Vec<TapeEntry>,
    /// Stored values (flattened f32 slices).
    values: Vec<Vec<f32>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Register a leaf value (input) on the tape. Returns its index.
    pub fn leaf(&mut self, value: Vec<f32>) -> TapedValue {
        let idx = self.values.len();
        self.values.push(value);
        TapedValue { index: idx }
    }

    /// Record an operation. `inputs` are TapedValue references, `output` is the
    /// computed output, and `grad_fn` computes input gradients from output gradient.
    pub fn record(
        &mut self,
        inputs: &[&TapedValue],
        output: Vec<f32>,
        operation: Operation,
        grad_fn: Box<dyn GradFn>,
    ) -> TapedValue {
        let output_len = output.len();
        let input_indices: Vec<usize> = inputs.iter().map(|tv| tv.index).collect();
        let idx = self.values.len();
        self.values.push(output);
        self.entries.push(TapeEntry {
            input_indices,
            _operation: operation,
            grad_fn,
            output_len,
        });
        TapedValue { index: idx }
    }

    /// Run backward pass from the given output, returning gradients for all values.
    pub fn backward(&self, output: &TapedValue) -> Result<Vec<Vec<f32>>, VjpError> {
        let n = self.values.len();
        let mut grads: Vec<Vec<f32>> = self.values.iter().map(|v| vec![0.0; v.len()]).collect();

        // Seed gradient: d(output)/d(output) = 1.0
        let out_len = self.values[output.index].len();
        grads[output.index] = vec![1.0; out_len];

        // Walk entries in reverse (reverse topological order)
        // Entry i produced value at index (num_leaves + i)
        let num_leaves = n - self.entries.len();
        for (i, entry) in self.entries.iter().enumerate().rev() {
            let val_idx = num_leaves + i;
            let grad_output = grads[val_idx].clone();
            debug_assert_eq!(grad_output.len(), entry.output_len);

            let input_grads = entry.grad_fn.backward(&grad_output)?;
            debug_assert_eq!(input_grads.len(), entry.input_indices.len());

            for (j, &inp_idx) in entry.input_indices.iter().enumerate() {
                for (k, &g) in input_grads[j].iter().enumerate() {
                    grads[inp_idx][k] += g;
                }
            }
        }

        Ok(grads)
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

/// A value tracked on the tape, identified by its index.
#[derive(Debug, Clone)]
pub struct TapedValue {
    pub index: usize,
}

// ─── Built-in gradient functions ───

/// Gradient for elementwise addition: grad flows through unchanged to both inputs.
pub struct AddGrad {
    pub len: usize,
}

impl GradFn for AddGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        Ok(vec![grad_output.to_vec(), grad_output.to_vec()])
    }
}

/// Gradient for elementwise multiplication (product rule).
pub struct MulGrad {
    pub lhs: Vec<f32>,
    pub rhs: Vec<f32>,
}

impl GradFn for MulGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        let grad_lhs: Vec<f32> = grad_output
            .iter()
            .zip(&self.rhs)
            .map(|(g, r)| g * r)
            .collect();
        let grad_rhs: Vec<f32> = grad_output
            .iter()
            .zip(&self.lhs)
            .map(|(g, l)| g * l)
            .collect();
        Ok(vec![grad_lhs, grad_rhs])
    }
}

/// Gradient for matrix multiply C = A @ B.
/// A: (m, k), B: (k, n), C: (m, n)
/// dA = dC @ B^T, dB = A^T @ dC
pub struct MatMulGrad {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl GradFn for MatMulGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        let (m, k, n) = (self.m, self.k, self.n);
        // dA = grad_output @ B^T — (m,n) @ (n,k) -> (m,k)
        let mut grad_a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for p in 0..n {
                    sum += grad_output[i * n + p] * self.b[j * n + p];
                }
                grad_a[i * k + j] = sum;
            }
        }
        // dB = A^T @ grad_output — (k,m) @ (m,n) -> (k,n)
        let mut grad_b = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..m {
                    sum += self.a[p * k + i] * grad_output[p * n + j];
                }
                grad_b[i * n + j] = sum;
            }
        }
        Ok(vec![grad_a, grad_b])
    }
}

/// Gradient for row-wise softmax.
/// output = softmax(input), shape (rows, cols).
/// grad_input = output * (grad_output - sum(grad_output * output, axis=-1, keepdim))
pub struct SoftmaxGrad {
    /// Softmax output values (rows * cols).
    pub output: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl GradFn for SoftmaxGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        let (rows, cols) = (self.rows, self.cols);
        let mut grad_input = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            // dot = sum(grad_output[row] * output[row])
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += grad_output[base + c] * self.output[base + c];
            }
            // grad_input[i] = output[i] * (grad_output[i] - dot)
            for c in 0..cols {
                grad_input[base + c] = self.output[base + c] * (grad_output[base + c] - dot);
            }
        }
        Ok(vec![grad_input])
    }
}

/// Gradient for RMS normalization.
/// y = x * rsqrt(mean(x^2) + eps) * weight
/// Computed via chain rule matching MLX's vjp_rms kernel.
pub struct RmsNormGrad {
    /// Input values (rows * axis_size).
    pub input: Vec<f32>,
    /// Weight values (axis_size).
    pub weight: Vec<f32>,
    pub rows: usize,
    pub axis_size: usize,
    pub eps: f32,
}

impl GradFn for RmsNormGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        let (rows, d) = (self.rows, self.axis_size);
        let mut grad_input = vec![0.0f32; rows * d];

        for r in 0..rows {
            let base = r * d;
            // sum of x^2
            let mut sum_x2 = 0.0f32;
            for i in 0..d {
                let x = self.input[base + i];
                sum_x2 += x * x;
            }
            let normalizer = 1.0 / (sum_x2 / d as f32 + self.eps).sqrt();
            let normalizer3 = normalizer * normalizer * normalizer;

            // sum(g * w * x) for this row
            let mut sum_gwx = 0.0f32;
            for i in 0..d {
                sum_gwx += grad_output[base + i] * self.weight[i] * self.input[base + i];
            }
            let mean_gwx = sum_gwx / d as f32;

            // grad_x[i] = g[i] * w[i] * normalizer - x[i] * mean_gwx * normalizer^3
            for i in 0..d {
                grad_input[base + i] = grad_output[base + i] * self.weight[i] * normalizer
                    - self.input[base + i] * mean_gwx * normalizer3;
            }
        }
        // Two inputs: input and weight. Return grads for both to
        // satisfy tape invariant (one gradient per recorded input).
        // Weight gradient: d(loss)/d(w[i]) = sum_rows(g[r,i] * x[r,i] * normalizer[r])
        let mut grad_weight = vec![0.0f32; d];
        for r in 0..rows {
            let base = r * d;
            let mut sum_x2 = 0.0f32;
            for i in 0..d {
                let x = self.input[base + i];
                sum_x2 += x * x;
            }
            let normalizer = 1.0 / (sum_x2 / d as f32 + self.eps).sqrt();
            for i in 0..d {
                grad_weight[i] += grad_output[base + i] * self.input[base + i] * normalizer;
            }
        }
        Ok(vec![grad_input, grad_weight])
    }
}

/// Gradient for reduce-sum: broadcast grad to input shape.
pub struct ReduceSumGrad {
    /// Number of elements in the original input.
    pub input_len: usize,
}

impl GradFn for ReduceSumGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        // grad_output is scalar [1], broadcast to full input shape
        let g = grad_output[0];
        Ok(vec![vec![g; self.input_len]])
    }
}

/// Gradient for reduce-max: grad at argmax position, zero elsewhere.
pub struct ReduceMaxGrad {
    /// Original input values.
    pub input: Vec<f32>,
}

impl GradFn for ReduceMaxGrad {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        let g = grad_output[0];
        // Find argmax
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &v) in self.input.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        let mut grad = vec![0.0f32; self.input.len()];
        grad[max_idx] = g;
        Ok(vec![grad])
    }
}

/// Placeholder gradient for operations without backward implementation.
/// Returns `Err(VjpError::UnsupportedBackward)` instead of panicking.
pub struct PlaceholderGrad {
    pub op_name: String,
    pub input_lens: Vec<usize>,
}

impl GradFn for PlaceholderGrad {
    fn backward(&self, _grad_output: &[f32]) -> Result<Vec<Vec<f32>>, VjpError> {
        Err(VjpError::UnsupportedBackward(self.op_name.clone()))
    }
}

/// Numerical gradient approximation for testing VJP correctness.
/// `f` maps an input vector to an output vector.
/// Returns the Jacobian approximation at `x` using central differences.
pub fn numerical_gradient<F>(f: F, x: &[f32], eps: f32) -> Vec<Vec<f32>>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let base = f(x);
    let out_dim = base.len();
    let in_dim = x.len();

    // jacobian[i][j] = d(output_i) / d(input_j)
    // We return gradient per input element, summed over outputs (for scalar loss).
    // Actually, return full jacobian as vec of columns.
    let mut jacobian = vec![vec![0.0f32; out_dim]; in_dim];

    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for j in 0..in_dim {
        x_plus[j] = x[j] + eps;
        x_minus[j] = x[j] - eps;
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        for i in 0..out_dim {
            jacobian[j][i] = (f_plus[i] - f_minus[i]) / (2.0 * eps);
        }
        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }

    jacobian
}
