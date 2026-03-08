//! RMSNorm + GEMM Fusion Benchmark (J-5).
//!
//! Compares three configurations:
//!   1. **MLX ref**: MLX `rms_norm + matmul` (from `benchmarks/mlx_fusion_bench.py`)
//!   2. **RMLX baseline**: `rms_norm_into_cb` + `matmul_into_cb` (2 dispatches)
//!   3. **RMLX fused**: `matmul_norm_gemm_into_cb` (inv_rms + GEMM has_norm=true)
//!
//! MLX reference values are loaded from a JSON file via `MLX_FUSION_REF` env var:
//!
//!   # Step 1: Run MLX reference on node0
//!   python benchmarks/mlx_fusion_bench.py --out /tmp/mlx_fusion_ref.json
//!
//!   # Step 2: Run RMLX benchmark with MLX reference
//!   MLX_FUSION_REF=/tmp/mlx_fusion_ref.json cargo bench -p rmlx-core --bench fusion_bench
//!
//! Parameters: M=1/32/512, K=4096, N=4096/14336 (LLaMA-3 8B).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = micros.iter().sum();
        let mean = sum / n as f64;
        let variance: f64 = micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let p50 = percentile(&micros, 50.0);
        Stats { mean, std_dev, p50 }
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - (rank - lower as f64)) + sorted[upper] * (rank - lower as f64)
    }
}

fn tflops(m: usize, k: usize, n: usize, p50_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    flops / (p50_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// f16 random array generation (deterministic PRNG)
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut f16_bytes = Vec::with_capacity(numel * 2);
    let mut state = seed;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        let h = f32_to_f16_bits(val as f32);
        f16_bytes.extend_from_slice(&h.to_le_bytes());
    }
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// MLX reference loader — reads JSON from MLX_FUSION_REF env var (file path)
// ---------------------------------------------------------------------------
//
// Expected JSON format (from benchmarks/mlx_fusion_bench.py):
// {
//   "results": [
//     {"m": 1, "k": 4096, "n": 4096, "rms_mm_p50_us": 42.3, ...},
//     ...
//   ]
// }

/// Minimal JSON parser for MLX fusion reference file.
/// Extracts (m, k, n) -> rms_mm_p50_us from the "results" array.
fn load_mlx_refs_from_json(path: &str) -> HashMap<(usize, usize, usize), f64> {
    let mut refs = HashMap::new();
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("WARNING: Cannot read MLX_FUSION_REF file '{}': {}", path, e);
            return refs;
        }
    };

    // Simple field extraction without serde dependency.
    // Find each result object by scanning for "rms_mm_p50_us" entries.
    let mut pos = 0;
    let bytes = content.as_bytes();
    while pos < bytes.len() {
        // Find next "m": in results array
        let m_val = match find_json_int(&content, pos, "\"m\"") {
            Some((v, next)) => {
                pos = next;
                v
            }
            None => break,
        };
        let k_val = match find_json_int(&content, pos, "\"k\"") {
            Some((v, next)) => {
                pos = next;
                v
            }
            None => break,
        };
        let n_val = match find_json_int(&content, pos, "\"n\"") {
            Some((v, next)) => {
                pos = next;
                v
            }
            None => break,
        };
        let p50_val = match find_json_float(&content, pos, "\"rms_mm_p50_us\"") {
            Some((v, next)) => {
                pos = next;
                v
            }
            None => break,
        };
        refs.insert((m_val, k_val, n_val), p50_val);
    }

    refs
}

fn find_json_int(s: &str, from: usize, key: &str) -> Option<(usize, usize)> {
    let idx = s[from..].find(key)?;
    let after_key = from + idx + key.len();
    // Skip ':' and whitespace
    let colon = s[after_key..].find(':')?;
    let num_start = after_key + colon + 1;
    let trimmed = s[num_start..].trim_start();
    let num_offset = s.len() - trimmed.len();
    let _ = num_offset; // offset into original
    let num_end = trimmed.find(|c: char| !c.is_ascii_digit())?;
    let val: usize = trimmed[..num_end].parse().ok()?;
    let abs_end = s.len() - trimmed.len() + num_end;
    Some((val, abs_end))
}

fn find_json_float(s: &str, from: usize, key: &str) -> Option<(f64, usize)> {
    let idx = s[from..].find(key)?;
    let after_key = from + idx + key.len();
    let colon = s[after_key..].find(':')?;
    let num_start = after_key + colon + 1;
    let trimmed = s[num_start..].trim_start();
    let num_end = trimmed.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    let val: f64 = trimmed[..num_end].parse().ok()?;
    let abs_end = s.len() - trimmed.len() + num_end;
    Some((val, abs_end))
}

fn load_mlx_refs() -> HashMap<(usize, usize, usize), f64> {
    match std::env::var("MLX_FUSION_REF") {
        Ok(path) => load_mlx_refs_from_json(&path),
        Err(_) => HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

struct Scenario {
    label: &'static str,
    k: usize,
    n: usize,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            label: "LLaMA-3 attn proj",
            k: 4096,
            n: 4096,
        },
        Scenario {
            label: "LLaMA-3 FFN up/gate",
            k: 4096,
            n: 14336,
        },
    ]
}

// ---------------------------------------------------------------------------
// Benchmark: RMLX baseline (rms_norm + matmul, 2 dispatches)
// ---------------------------------------------------------------------------

fn bench_baseline(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    k: usize,
    n: usize,
    m: usize,
) -> Stats {
    let a = rand_array(device, &[m, k], 42);
    let b = rand_array(device, &[k, n], 44);
    let w = rand_array(device, &[k], 55);
    let eps: f32 = 1e-5;

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let normed = ops::rms_norm::rms_norm_into_cb(registry, &a, Some(&w), eps, cb).unwrap();
        let _ = ops::matmul::matmul_into_cb(registry, &normed, &b, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let normed = ops::rms_norm::rms_norm_into_cb(registry, &a, Some(&w), eps, cb).unwrap();
        let _ = ops::matmul::matmul_into_cb(registry, &normed, &b, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

// ---------------------------------------------------------------------------
// Benchmark: RMLX fused (matmul_norm_gemm_into_cb)
// ---------------------------------------------------------------------------

fn bench_fused(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    k: usize,
    n: usize,
    m: usize,
) -> Option<Stats> {
    let a = rand_array(device, &[m, k], 42);
    let b = rand_array(device, &[k, n], 44);
    let w = rand_array(device, &[k], 55);
    let eps: f32 = 1e-5;

    // Probe: skip gracefully if unsupported (non-MlxArch tile)
    {
        let cb = queue.new_command_buffer();
        match ops::matmul::matmul_norm_gemm_into_cb(registry, &a, &b, &w, eps, cb) {
            Ok(_) => {
                cb.commit();
                cb.wait_until_completed();
            }
            Err(_) => return None,
        }
    }

    for _ in 1..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let _ = ops::matmul::matmul_norm_gemm_into_cb(registry, &a, &b, &w, eps, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = ops::matmul::matmul_norm_gemm_into_cb(registry, &a, &b, &w, eps, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    Some(Stats::from_durations(&times))
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

struct Row {
    scenario: String,
    m: usize,
    k: usize,
    n: usize,
    mlx_p50: Option<f64>,
    baseline_p50: f64,
    fused_p50: Option<f64>,
}

fn print_summary(rows: &[Row]) {
    println!("\n{}", "=".repeat(115));
    println!("SUMMARY: MLX ref / RMLX baseline / RMLX fused  (p50 in microseconds)");
    println!("{}", "=".repeat(115));
    println!(
        "{:<22} {:>5}  {:>10} {:>8}  {:>10} {:>8}  {:>10} {:>8}  {:>8} {:>8}",
        "Scenario",
        "M",
        "MLX(us)",
        "TFLOPS",
        "base(us)",
        "TFLOPS",
        "fused(us)",
        "TFLOPS",
        "f/b",
        "f/mlx"
    );
    println!("{}", "-".repeat(115));

    for r in rows {
        let mlx_str = r
            .mlx_p50
            .map(|v| format!("{:10.1}", v))
            .unwrap_or_else(|| "       n/a".to_string());
        let mlx_tf = r
            .mlx_p50
            .map(|v| format!("{:8.2}", tflops(r.m, r.k, r.n, v)))
            .unwrap_or_else(|| "     n/a".to_string());

        let base_str = format!("{:10.1}", r.baseline_p50);
        let base_tf = format!("{:8.2}", tflops(r.m, r.k, r.n, r.baseline_p50));

        let (fused_str, fused_tf, fb_ratio, fmlx_ratio) = match r.fused_p50 {
            Some(fp) => {
                let fb = format!("{:8.2}x", r.baseline_p50 / fp);
                let fmlx = r
                    .mlx_p50
                    .map(|mp| format!("{:8.2}x", mp / fp))
                    .unwrap_or_else(|| "     n/a".to_string());
                (
                    format!("{:10.1}", fp),
                    format!("{:8.2}", tflops(r.m, r.k, r.n, fp)),
                    fb,
                    fmlx,
                )
            }
            None => (
                "   SKIPPED".to_string(),
                "     n/a".to_string(),
                "     n/a".to_string(),
                "     n/a".to_string(),
            ),
        };

        println!(
            "{:<22} {:>5}  {} {}  {} {}  {} {}  {} {}",
            r.scenario,
            r.m,
            mlx_str,
            mlx_tf,
            base_str,
            base_tf,
            fused_str,
            fused_tf,
            fb_ratio,
            fmlx_ratio,
        );
    }

    println!("{}", "-".repeat(115));
    println!("f/b  = fused speedup over RMLX baseline  (>1 means fused is faster)");
    println!("f/mlx = fused speedup over MLX reference  (>1 means RMLX fused beats MLX)");
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    let mlx_refs = load_mlx_refs();
    let has_mlx = !mlx_refs.is_empty();

    let m_values = [1, 32, 512];

    println!("=== RMSNorm + GEMM Fusion Benchmark (J-5, f16) ===");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    if has_mlx {
        println!(
            "MLX reference: loaded {} entries from JSON file.",
            mlx_refs.len()
        );
    } else {
        println!("No MLX reference found. To enable 3-way comparison:");
        println!("  1. python benchmarks/mlx_fusion_bench.py --out /tmp/mlx_fusion_ref.json");
        println!("  2. MLX_FUSION_REF=/tmp/mlx_fusion_ref.json cargo bench -p rmlx-core --bench fusion_bench");
    }
    println!("Fused path requires MlxArch tile (M>=33, N>=33); small M will be skipped.");
    println!();

    let mut rows = Vec::new();

    for scenario in &scenarios() {
        println!("[{}: K={}, N={}]", scenario.label, scenario.k, scenario.n);

        for &m in &m_values {
            let mlx_p50 = mlx_refs.get(&(m, scenario.k, scenario.n)).copied();

            if let Some(mp) = mlx_p50 {
                let tf = tflops(m, scenario.k, scenario.n, mp);
                println!("  M={:5}  MLX ref   p50={:8.1}us  TFLOPS={:.2}", m, mp, tf,);
            }

            let baseline = bench_baseline(&registry, &queue, device, scenario.k, scenario.n, m);
            let bt = tflops(m, scenario.k, scenario.n, baseline.p50);
            println!(
                "  M={:5}  baseline  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS={:.2}",
                m, baseline.p50, baseline.mean, baseline.std_dev, bt,
            );

            let fused = bench_fused(&registry, &queue, device, scenario.k, scenario.n, m);
            match &fused {
                Some(fs) => {
                    let ft = tflops(m, scenario.k, scenario.n, fs.p50);
                    let speedup = baseline.p50 / fs.p50;
                    println!(
                        "  M={:5}  fused     p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS={:.2}  speedup={:.2}x",
                        m, fs.p50, fs.mean, fs.std_dev, ft, speedup,
                    );
                }
                None => {
                    println!("  M={:5}  fused     SKIPPED (non-MlxArch tile)", m);
                }
            }

            rows.push(Row {
                scenario: scenario.label.to_string(),
                m,
                k: scenario.k,
                n: scenario.n,
                mlx_p50,
                baseline_p50: baseline.p50,
                fused_p50: fused.map(|s| s.p50),
            });
        }
        println!();
    }

    print_summary(&rows);
    println!("Done.");
}
