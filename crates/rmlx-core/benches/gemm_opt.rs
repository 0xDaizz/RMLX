//! ⚠️ NON-PRODUCTION PATH — old GEMM optimization parameter sweep. Direct kernel encoding,
//! development only. Bypasses matmul() dispatch.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Parameterized GEMM optimization benchmark — tests 11 kernel variants with
//! serpentine MMA ordering, threadgroup memory padding, and single/double buffering.
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench gemm_opt

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use std::ptr::NonNull;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _, MTLCommandQueue as _, MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLCommandEncoder as _};
use rmlx_metal::{MTLSize, MTLResourceOptions};
use rmlx_metal::types::{MtlBuffer};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// GEMM optimization configuration
// ---------------------------------------------------------------------------

struct GemmOptConfig {
    label: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    sg_rows: u32,
    sg_cols: u32,
    serpentine: bool,
    pad: u32,
    double_buffered: bool,
}

impl GemmOptConfig {
    fn n_sg(&self) -> u32 {
        self.sg_rows * self.sg_cols
    }
    fn n_threads(&self) -> u32 {
        self.n_sg() * 32
    }
    fn tm(&self) -> u32 {
        self.bm / (8 * self.sg_rows)
    }
    fn tn(&self) -> u32 {
        self.bn / (8 * self.sg_cols)
    }
}

const CONFIGS: &[GemmOptConfig] = &[
    // 1. Reference: current best config, no optimizations
    GemmOptConfig {
        label: "ref_bk32_2x4",
        bm: 64,
        bn: 64,
        bk: 32,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: false,
        pad: 0,
        double_buffered: true,
    },
    // 2. Serpentine MMA only
    GemmOptConfig {
        label: "serp_bk32_2x4",
        bm: 64,
        bn: 64,
        bk: 32,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: true,
        pad: 0,
        double_buffered: true,
    },
    // 3. TG memory padding only (PAD=8 halves = 16 bytes per row)
    GemmOptConfig {
        label: "pad_bk32_2x4",
        bm: 64,
        bn: 64,
        bk: 32,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: false,
        pad: 8,
        double_buffered: true,
    },
    // 4. Both serpentine + padding
    GemmOptConfig {
        label: "full_bk32_2x4",
        bm: 64,
        bn: 64,
        bk: 32,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: true,
        pad: 8,
        double_buffered: true,
    },
    // 5. bk16_2x2 + both optimizations
    GemmOptConfig {
        label: "full_bk16_2x2",
        bm: 64,
        bn: 64,
        bk: 16,
        sg_rows: 2,
        sg_cols: 2,
        serpentine: true,
        pad: 8,
        double_buffered: true,
    },
    // 6. MLX-like: bk16_1x2, 64 threads, acc[8][4]=32 MMA/SG + both optimizations
    GemmOptConfig {
        label: "full_bk16_1x2",
        bm: 64,
        bn: 64,
        bk: 16,
        sg_rows: 1,
        sg_cols: 2,
        serpentine: true,
        pad: 8,
        double_buffered: true,
    },
    // --- Occupancy-focused: single buffer to reduce TG memory ---
    // 7. MLX-like: single buffer, 2 SG, serpentine+pad (~5KB TG → ~6 TG/core)
    GemmOptConfig {
        label: "mlx_like",
        bm: 64,
        bn: 64,
        bk: 16,
        sg_rows: 1,
        sg_cols: 2,
        serpentine: true,
        pad: 8,
        double_buffered: false,
    },
    // 8. MLX-like without padding (~4KB TG)
    GemmOptConfig {
        label: "mlx_nopad",
        bm: 64,
        bn: 64,
        bk: 16,
        sg_rows: 1,
        sg_cols: 2,
        serpentine: true,
        pad: 0,
        double_buffered: false,
    },
    // 9. Single buffer, 4 SG, 2x2 layout (~6KB TG → ~5 TG/core)
    GemmOptConfig {
        label: "occ_2x2",
        bm: 64,
        bn: 64,
        bk: 16,
        sg_rows: 2,
        sg_cols: 2,
        serpentine: true,
        pad: 8,
        double_buffered: false,
    },
    // 10. Reference: double buffer bk32_2x4, NO optimizations → single buffer version
    GemmOptConfig {
        label: "ref_single",
        bm: 64,
        bn: 64,
        bk: 32,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: false,
        pad: 0,
        double_buffered: false,
    },
    // 11. 128x128 tile, single buffer, 8 SG (~10KB TG → ~3 TG/core)
    GemmOptConfig {
        label: "big_tile",
        bm: 128,
        bn: 128,
        bk: 16,
        sg_rows: 2,
        sg_cols: 4,
        serpentine: true,
        pad: 8,
        double_buffered: false,
    },
];

// ---------------------------------------------------------------------------
// Metal shader generator
// ---------------------------------------------------------------------------

fn generate_optimized_gemm_shader(cfg: &GemmOptConfig) -> String {
    let bm = cfg.bm;
    let bn = cfg.bn;
    let bk = cfg.bk;
    let n_sg = cfg.n_sg();
    let sg_rows = cfg.sg_rows;
    let sg_cols = cfg.sg_cols;
    let n_threads = cfg.n_threads();
    let tm = cfg.tm();
    let tn = cfg.tn();
    let pad = cfg.pad;
    let serpentine = cfg.serpentine;
    let double_buffered = cfg.double_buffered;
    let sw_bufs: u32 = if double_buffered { 2 } else { 1 };
    let label = cfg.label;

    let mut s = String::with_capacity(12288);

    // Header
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Constants
    s.push_str(&format!(
        "constant constexpr uint SW_BM = {bm};\n\
         constant constexpr uint SW_BN = {bn};\n\
         constant constexpr uint SW_BK = {bk};\n\
         constant constexpr uint SW_PAD = {pad};\n\
         constant constexpr uint SW_N_SG = {n_sg};\n\
         constant constexpr uint SW_SG_ROWS = {sg_rows};\n\
         constant constexpr uint SW_SG_COLS = {sg_cols};\n\
         constant constexpr uint SW_N_THREADS = {n_threads};\n\
         constant constexpr uint SW_TM = {tm};\n\
         constant constexpr uint SW_TN = {tn};\n\
         constant constexpr uint SW_BUFS = {sw_bufs};\n\
         constant constexpr uint SW_A_STRIDE = SW_BK + SW_PAD;\n\
         constant constexpr uint SW_B_STRIDE = SW_BN + SW_PAD;\n\n"
    ));

    // Uniform hint
    s.push_str(
        "#if __METAL_VERSION__ >= 310\n\
         template <typename T>\n\
         METAL_FUNC uniform<T> sw_as_uniform(T val) {\n\
             return make_uniform(val);\n\
         }\n\
         #else\n\
         template <typename T>\n\
         METAL_FUNC T sw_as_uniform(T val) {\n\
             return val;\n\
         }\n\
         #endif\n\n",
    );

    // Swizzle helper
    s.push_str(
        "inline uint2 sw_swizzle_tg(uint2 tid, uint swizzle_log) {\n\
             if (swizzle_log == 0) return tid;\n\
             return uint2(\n\
                 tid.x >> swizzle_log,\n\
                 (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))\n\
             );\n\
         }\n\n",
    );

    // Kernel function signature
    s.push_str(&format!(
        "kernel void gemm_opt_{label}(\n\
         \x20   device const half* A [[buffer(0)]],\n\
         \x20   device const half* B [[buffer(1)]],\n\
         \x20   device half* C       [[buffer(2)]],\n\
         \x20   constant uint& M     [[buffer(3)]],\n\
         \x20   constant uint& N     [[buffer(4)]],\n\
         \x20   constant uint& K     [[buffer(5)]],\n\
         \x20   constant uint& batch_stride_a [[buffer(6)]],\n\
         \x20   constant uint& batch_stride_b [[buffer(7)]],\n\
         \x20   constant uint& batch_stride_c [[buffer(8)]],\n\
         \x20   constant uint& swizzle_log    [[buffer(9)]],\n\
         \x20   uint3 group_id       [[threadgroup_position_in_grid]],\n\
         \x20   uint  tid_in_group   [[thread_index_in_threadgroup]],\n\
         \x20   uint  sgid           [[simdgroup_index_in_threadgroup]],\n\
         \x20   uint  lane_id        [[thread_index_in_simdgroup]])\n\
         {{\n"
    ));

    // Shared memory (with padded stride)
    s.push_str(
        "    threadgroup half As[SW_BUFS][SW_BM * SW_A_STRIDE];\n\
         \x20   threadgroup half Bs[SW_BUFS][SW_BK * SW_B_STRIDE];\n\n",
    );

    // Batch / swizzle setup
    s.push_str(
        "    const uint batch_idx = group_id.z;\n\
         \x20   uint2 swizzled = sw_swizzle_tg(uint2(group_id.x, group_id.y), swizzle_log);\n\
         \x20   const uint row_start = swizzled.y * sw_as_uniform(SW_BM);\n\
         \x20   const uint col_start = swizzled.x * sw_as_uniform(SW_BN);\n\n\
         \x20   device const half* A_batch = A + batch_idx * sw_as_uniform(batch_stride_a);\n\
         \x20   device const half* B_batch = B + batch_idx * sw_as_uniform(batch_stride_b);\n\
         \x20   device half*       C_batch = C + batch_idx * sw_as_uniform(batch_stride_c);\n\n",
    );

    // SG grid position
    s.push_str(
        "    const uint sg_row = sgid / SW_SG_COLS;\n\
         \x20   const uint sg_col = sgid % SW_SG_COLS;\n\n",
    );

    // Accumulators
    s.push_str(
        "    simdgroup_float8x8 acc[SW_TM][SW_TN];\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < SW_TM; i++)\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < SW_TN; j++)\n\
         \x20           acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\n",
    );

    // Uniform dimension copies
    s.push_str(
        "    const uint uK = sw_as_uniform(K);\n\
         \x20   const uint uM = sw_as_uniform(M);\n\
         \x20   const uint uN = sw_as_uniform(N);\n\
         \x20   const uint n_tiles = (uK + SW_BK - 1) / SW_BK;\n\n",
    );

    // Main compute loop — double-buffered or single-buffered
    if double_buffered {
        generate_optimized_double_buffered_loop(&mut s, tm, tn, serpentine);
    } else {
        generate_optimized_single_buffered_loop(&mut s, tm, tn, serpentine);
    }

    // Store path — separate scratch buffer (same as sweep2)
    s.push_str(&format!(
        "\n    // ── Store results ──\n\
         \x20   threadgroup float result_scratch[SW_N_SG * 64];\n\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20           simdgroup_store(acc[i][j], &result_scratch[sgid * 64], 8);\n\
         \x20           threadgroup_barrier(mem_flags::mem_threadgroup);\n\n\
         \x20           uint base_row = row_start + sg_row * {sg_sub_m} + i * 8;\n\
         \x20           uint base_col = col_start + sg_col * {sg_sub_n} + j * 8;\n\n\
         \x20           for (uint idx = lane_id; idx < 64; idx += 32) {{\n\
         \x20               uint lr = idx / 8;\n\
         \x20               uint lc = idx % 8;\n\
         \x20               uint gr = base_row + lr;\n\
         \x20               uint gc = base_col + lc;\n\
         \x20               if (gr < uM && gc < uN) {{\n\
         \x20                   C_batch[gr * uN + gc] = half(result_scratch[sgid * 64 + lr * 8 + lc]);\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20           threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20       }}\n\
         \x20   }}\n",
        sg_sub_m = tm * 8,
        sg_sub_n = tn * 8,
    ));

    // Close kernel
    s.push_str("}\n");

    s
}

fn generate_optimized_double_buffered_loop(s: &mut String, tm: u32, tn: u32, serpentine: bool) {
    // Prefetch first tile into buffer[0] (with padded stride)
    s.push_str(
        "    // ── Prefetch first tile into buffer[0] ──\n\
         \x20   for (uint idx = tid_in_group; idx < SW_BM * SW_BK / 4; idx += SW_N_THREADS) {\n\
         \x20       uint elem = idx * 4;\n\
         \x20       uint r = elem / SW_BK;\n\
         \x20       uint c = elem % SW_BK;\n\
         \x20       uint gr = row_start + r;\n\
         \x20       uint gc = c;\n\
         \x20       if (gr < uM && gc + 3 < uK) {\n\
         \x20           *reinterpret_cast<threadgroup half4*>(&As[0][r * SW_A_STRIDE + c]) =\n\
         \x20               *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc]);\n\
         \x20       } else {\n\
         \x20           for (uint d = 0; d < 4; d++) {\n\
         \x20               As[0][r * SW_A_STRIDE + c + d] = (gr < uM && gc + d < uK)\n\
         \x20                   ? A_batch[gr * uK + gc + d] : half(0);\n\
         \x20           }\n\
         \x20       }\n\
         \x20   }\n\
         \x20   for (uint idx = tid_in_group; idx < SW_BK * SW_BN / 4; idx += SW_N_THREADS) {\n\
         \x20       uint elem = idx * 4;\n\
         \x20       uint r = elem / SW_BN;\n\
         \x20       uint c = elem % SW_BN;\n\
         \x20       uint gr = r;\n\
         \x20       uint gc = col_start + c;\n\
         \x20       if (gr < uK && gc + 3 < uN) {\n\
         \x20           *reinterpret_cast<threadgroup half4*>(&Bs[0][r * SW_B_STRIDE + c]) =\n\
         \x20               *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
         \x20       } else {\n\
         \x20           for (uint d = 0; d < 4; d++) {\n\
         \x20               Bs[0][r * SW_B_STRIDE + c + d] = (gr < uK && gc + d < uN)\n\
         \x20                   ? B_batch[gr * uN + gc + d] : half(0);\n\
         \x20           }\n\
         \x20       }\n\
         \x20   }\n\
         \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\n",
    );

    // Main double-buffered loop
    s.push_str(
        "    // ── Main loop: double-buffered ──\n\
         \x20   for (uint tile = 0; tile < n_tiles; tile++) {\n\
         \x20       uint stage = tile & 1;\n\
         \x20       uint next_stage = 1 - stage;\n\
         \x20       uint next_kb = (tile + 1) * SW_BK;\n\n\
         \x20       // Prefetch next tile (vectorized half4) with padded stride\n\
         \x20       if (tile + 1 < n_tiles) {\n\
         \x20           for (uint idx = tid_in_group; idx < SW_BM * SW_BK / 4; idx += SW_N_THREADS) {\n\
         \x20               uint elem = idx * 4;\n\
         \x20               uint r = elem / SW_BK;\n\
         \x20               uint c = elem % SW_BK;\n\
         \x20               uint gr = row_start + r;\n\
         \x20               uint gc = next_kb + c;\n\
         \x20               if (gr < uM && gc + 3 < uK) {\n\
         \x20                   *reinterpret_cast<threadgroup half4*>(&As[next_stage][r * SW_A_STRIDE + c]) =\n\
         \x20                       *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc]);\n\
         \x20               } else {\n\
         \x20                   for (uint d = 0; d < 4; d++) {\n\
         \x20                       As[next_stage][r * SW_A_STRIDE + c + d] = (gr < uM && gc + d < uK)\n\
         \x20                           ? A_batch[gr * uK + gc + d] : half(0);\n\
         \x20                   }\n\
         \x20               }\n\
         \x20           }\n\
         \x20           for (uint idx = tid_in_group; idx < SW_BK * SW_BN / 4; idx += SW_N_THREADS) {\n\
         \x20               uint elem = idx * 4;\n\
         \x20               uint r = elem / SW_BN;\n\
         \x20               uint c = elem % SW_BN;\n\
         \x20               uint gr = next_kb + r;\n\
         \x20               uint gc = col_start + c;\n\
         \x20               if (gr < uK && gc + 3 < uN) {\n\
         \x20                   *reinterpret_cast<threadgroup half4*>(&Bs[next_stage][r * SW_B_STRIDE + c]) =\n\
         \x20                       *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
         \x20               } else {\n\
         \x20                   for (uint d = 0; d < 4; d++) {\n\
         \x20                       Bs[next_stage][r * SW_B_STRIDE + c + d] = (gr < uK && gc + d < uN)\n\
         \x20                           ? B_batch[gr * uN + gc + d] : half(0);\n\
         \x20                   }\n\
         \x20               }\n\
         \x20           }\n\
         \x20       }\n\n"
    );

    // Compute on current stage (with padded stride for simdgroup_load)
    let sg_sub_m = tm * 8;
    let sg_sub_n = tn * 8;

    // MMA loop — serpentine or standard
    let mma_body = if serpentine {
        "\x20           #pragma clang loop unroll(full)\n\
             \x20           for (uint i = 0; i < SW_TM; i++)\n\
             \x20               #pragma clang loop unroll(full)\n\
             \x20               for (uint jj = 0; jj < SW_TN; jj++) {{\n\
             \x20                   uint j = (i & 1u) ? (SW_TN - 1u - jj) : jj;\n\
             \x20                   simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);\n\
             \x20               }}\n"
            .to_string()
    } else {
        "\x20           #pragma clang loop unroll(full)\n\
             \x20           for (uint i = 0; i < SW_TM; i++)\n\
             \x20               #pragma clang loop unroll(full)\n\
             \x20               for (uint j = 0; j < SW_TN; j++)\n\
             \x20                   simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);\n"
            .to_string()
    };

    s.push_str(&format!(
        "        // Compute on buffer[stage]\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint kk = 0; kk < SW_BK; kk += 8) {{\n\
         \x20           simdgroup_half8x8 a_frag[SW_TM];\n\
         \x20           simdgroup_half8x8 b_frag[SW_TN];\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20               simdgroup_load(a_frag[i],\n\
         \x20                   &As[stage][(sg_row * {sg_sub_m} + i * 8) * SW_A_STRIDE + kk], SW_A_STRIDE);\n\
         \x20           }}\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20               simdgroup_load(b_frag[j],\n\
         \x20                   &Bs[stage][kk * SW_B_STRIDE + sg_col * {sg_sub_n} + j * 8], SW_B_STRIDE);\n\
         \x20           }}\n\n\
         {mma_body}\
         \x20       }}\n\n\
         \x20       if (tile + 1 < n_tiles) {{\n\
         \x20           threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20       }}\n\
         \x20   }}\n",
    ));
}

fn generate_optimized_single_buffered_loop(s: &mut String, tm: u32, tn: u32, serpentine: bool) {
    let sg_sub_m = tm * 8;
    let sg_sub_n = tn * 8;

    // MMA body — serpentine or standard (same as double-buffered but always stage 0)
    let mma_body = if serpentine {
        "\x20           #pragma clang loop unroll(full)\n\
             \x20           for (uint i = 0; i < SW_TM; i++)\n\
             \x20               #pragma clang loop unroll(full)\n\
             \x20               for (uint jj = 0; jj < SW_TN; jj++) {{\n\
             \x20                   uint j = (i & 1u) ? (SW_TN - 1u - jj) : jj;\n\
             \x20                   simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);\n\
             \x20               }}\n"
            .to_string()
    } else {
        "\x20           #pragma clang loop unroll(full)\n\
             \x20           for (uint i = 0; i < SW_TM; i++)\n\
             \x20               #pragma clang loop unroll(full)\n\
             \x20               for (uint j = 0; j < SW_TN; j++)\n\
             \x20                   simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);\n"
            .to_string()
    };

    s.push_str(&format!(
        "    // ── Main loop: single-buffered ──\n\
         \x20   for (uint tile = 0; tile < n_tiles; tile++) {{\n\
         \x20       uint kb = tile * SW_BK;\n\n\
         \x20       // Load tile into buffer[0]\n\
         \x20       for (uint idx = tid_in_group; idx < SW_BM * SW_BK / 4; idx += SW_N_THREADS) {{\n\
         \x20           uint elem = idx * 4;\n\
         \x20           uint r = elem / SW_BK;\n\
         \x20           uint c = elem % SW_BK;\n\
         \x20           uint gr = row_start + r;\n\
         \x20           uint gc = kb + c;\n\
         \x20           if (gr < uM && gc + 3 < uK) {{\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&As[0][r * SW_A_STRIDE + c]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc]);\n\
         \x20           }} else {{\n\
         \x20               for (uint d = 0; d < 4; d++) {{\n\
         \x20                   As[0][r * SW_A_STRIDE + c + d] = (gr < uM && gc + d < uK)\n\
         \x20                       ? A_batch[gr * uK + gc + d] : half(0);\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       for (uint idx = tid_in_group; idx < SW_BK * SW_BN / 4; idx += SW_N_THREADS) {{\n\
         \x20           uint elem = idx * 4;\n\
         \x20           uint r = elem / SW_BN;\n\
         \x20           uint c = elem % SW_BN;\n\
         \x20           uint gr = kb + r;\n\
         \x20           uint gc = col_start + c;\n\
         \x20           if (gr < uK && gc + 3 < uN) {{\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&Bs[0][r * SW_B_STRIDE + c]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
         \x20           }} else {{\n\
         \x20               for (uint d = 0; d < 4; d++) {{\n\
         \x20                   Bs[0][r * SW_B_STRIDE + c + d] = (gr < uK && gc + d < uN)\n\
         \x20                       ? B_batch[gr * uN + gc + d] : half(0);\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\n\
         \x20       // Compute on buffer[0]\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint kk = 0; kk < SW_BK; kk += 8) {{\n\
         \x20           simdgroup_half8x8 a_frag[SW_TM];\n\
         \x20           simdgroup_half8x8 b_frag[SW_TN];\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20               simdgroup_load(a_frag[i],\n\
         \x20                   &As[0][(sg_row * {sg_sub_m} + i * 8) * SW_A_STRIDE + kk], SW_A_STRIDE);\n\
         \x20           }}\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20               simdgroup_load(b_frag[j],\n\
         \x20                   &Bs[0][kk * SW_B_STRIDE + sg_col * {sg_sub_n} + j * 8], SW_B_STRIDE);\n\
         \x20           }}\n\n\
         {mma_body}\
         \x20       }}\n\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   }}\n",
    ));
}

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

fn rand_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
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

fn make_u32_buf(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, val: u32) -> MtlBuffer {
    let opts = MTLResourceOptions::StorageModeShared;
    unsafe { device.newBufferWithBytes_length_options(NonNull::new(&val as *const u32 as *const _ as *mut _).unwrap(), 4_usize, opts).unwrap() }
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn compute_swizzle_log(m: usize, bm: usize) -> u32 {
    let tiles_m = m.div_ceil(bm);
    if tiles_m > 3 {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_config(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    cfg: &GemmOptConfig,
    m: usize,
    k: usize,
    n: usize,
) {
    let kernel_name = format!("gemm_opt_{}", cfg.label);
    let pipeline = registry
        .get_pipeline(&kernel_name, DType::Float16)
        .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"));

    let a = rand_array(device, &[m, k], 42);
    let b = rand_array(device, &[k, n], 43);
    let c = Array::zeros(device, &[m, n], DType::Float16);

    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_buf = make_u32_buf(device, compute_swizzle_log(m, cfg.bm as usize));

    let grid_x = ceil_div(n, cfg.bn as usize) as u64;
    let grid_y = ceil_div(m, cfg.bm as usize) as u64;
    let grid = MTLSize { width: grid_x as usize, height: grid_y as usize, depth: 1_usize };
    let tg = MTLSize { width: cfg.n_threads() as usize, height: 1_usize, depth: 1_usize };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipeline);
        unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0_usize, 2_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&m_buf), 0_usize, 3_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&n_buf), 0_usize, 4_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&k_buf), 0_usize, 5_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsa_buf), 0_usize, 6_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsb_buf), 0_usize, 7_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsc_buf), 0_usize, 8_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&swizzle_buf), 0_usize, 9_usize) };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipeline);
        unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0_usize, 2_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&m_buf), 0_usize, 3_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&n_buf), 0_usize, 4_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&k_buf), 0_usize, 5_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsa_buf), 0_usize, 6_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsb_buf), 0_usize, 7_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&bsc_buf), 0_usize, 8_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(&swizzle_buf), 0_usize, 9_usize) };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let tflops_p50 = flops / (stats.p50 * 1e-6) / 1e12;
    let tflops_mean = flops / (stats.mean * 1e-6) / 1e12;

    println!(
        "  M={:5}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.2}  TFLOPS(mean)={:.2}",
        m, stats.p50, stats.mean, stats.std_dev, tflops_p50, tflops_mean,
    );
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    // Register all optimization kernels via JIT
    for cfg in CONFIGS {
        let source = generate_optimized_gemm_shader(cfg);
        let jit_name = format!("gemm_opt_{}", cfg.label);
        registry
            .register_jit_source(&jit_name, &source)
            .unwrap_or_else(|e| panic!("Failed to compile {}: {e}", cfg.label));
    }

    println!("=== GEMM Optimization Benchmark (f16) ===");
    println!(
        "Variants: {}, M values: 5, N values: [4096, 14336]",
        CONFIGS.len()
    );
    println!("Warmup: {}, Bench: {}", WARMUP_ITERS, BENCH_ITERS);
    println!();

    let m_values = [128, 256, 512, 1024, 2048];
    let kn_combos: &[(usize, usize)] = &[(4096, 4096), (4096, 14336)];

    for cfg in CONFIGS {
        let buf_label = if cfg.double_buffered {
            "double"
        } else {
            "single"
        };
        println!(
            "--- Config: {} (BM={} BN={} BK={} SG={}x{} thr={} serp={} pad={} buf={}) ---",
            cfg.label,
            cfg.bm,
            cfg.bn,
            cfg.bk,
            cfg.sg_rows,
            cfg.sg_cols,
            cfg.n_threads(),
            cfg.serpentine,
            cfg.pad,
            buf_label,
        );
        for &(k, n) in kn_combos {
            println!("K={}, N={}:", k, n);
            for &m in &m_values {
                bench_config(&registry, &queue, device, cfg, m, k, n);
            }
        }
        println!();
    }

    println!("Done.");
}
