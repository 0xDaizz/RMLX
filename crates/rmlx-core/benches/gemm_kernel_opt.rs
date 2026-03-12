//! GEMM kernel-level optimization benchmark — tests 6 variants measuring
//! independent contributions of 3 optimizations: direct store, wide load, aligned.
//!
//! All variants share BASE config: BM=64, BN=64, BK=32, SG=2x4, 256 threads,
//! double-buffered, no serpentine, no pad.
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench gemm_kernel_opt

use std::time::{Duration, Instant};

use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Kernel optimization configuration
// ---------------------------------------------------------------------------

struct GemmKernelOptConfig {
    label: &'static str,
    use_direct_store: bool,
    use_wide_load: bool,
    use_aligned: bool,
}

// Fixed base parameters for all variants
const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const SG_ROWS: u32 = 2;
const SG_COLS: u32 = 4;
const N_SG: u32 = SG_ROWS * SG_COLS;
const N_THREADS: u32 = N_SG * 32;
const TM: u32 = BM / (8 * SG_ROWS); // 4
const TN: u32 = BN / (8 * SG_COLS); // 2
const SG_SUB_M: u32 = TM * 8; // 32
const SG_SUB_N: u32 = TN * 8; // 16
const PAD: u32 = 0;
const SW_BUFS: u32 = 2;

const CONFIGS: &[GemmKernelOptConfig] = &[
    // 1. Reference: current bk32_2x4 kernel (identical to gemm_opt.rs ref_bk32_2x4)
    GemmKernelOptConfig {
        label: "ref",
        use_direct_store: false,
        use_wide_load: false,
        use_aligned: false,
    },
    // 2. Direct store only
    GemmKernelOptConfig {
        label: "direct_store",
        use_direct_store: true,
        use_wide_load: false,
        use_aligned: false,
    },
    // 3. Wide load only
    GemmKernelOptConfig {
        label: "wide_load",
        use_direct_store: false,
        use_wide_load: true,
        use_aligned: false,
    },
    // 4. Aligned only
    GemmKernelOptConfig {
        label: "aligned",
        use_direct_store: false,
        use_wide_load: false,
        use_aligned: true,
    },
    // 5. Direct store + wide load
    GemmKernelOptConfig {
        label: "ds_wl",
        use_direct_store: true,
        use_wide_load: true,
        use_aligned: false,
    },
    // 6. All three combined
    GemmKernelOptConfig {
        label: "full",
        use_direct_store: true,
        use_wide_load: true,
        use_aligned: true,
    },
    // 7. MLX architecture: BM=64, BN=64, BK=16, 2 SG (1×2), 64 threads,
    //    single buffer, wide loads, direct store, serpentine MMA
    GemmKernelOptConfig {
        label: "mlx_arch",
        use_direct_store: true,
        use_wide_load: true,
        use_aligned: false,
    },
];

// ---------------------------------------------------------------------------
// Metal shader generator
// ---------------------------------------------------------------------------

fn generate_kernel_opt_shader(cfg: &GemmKernelOptConfig) -> String {
    let label = cfg.label;
    let direct_store = cfg.use_direct_store;
    let wide_load = cfg.use_wide_load;
    let aligned = cfg.use_aligned;

    let mut s = String::with_capacity(16384);

    // Header
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Constants
    s.push_str(&format!(
        "constant constexpr uint SW_BM = {BM};\n\
         constant constexpr uint SW_BN = {BN};\n\
         constant constexpr uint SW_BK = {BK};\n\
         constant constexpr uint SW_PAD = {PAD};\n\
         constant constexpr uint SW_N_SG = {N_SG};\n\
         constant constexpr uint SW_SG_ROWS = {SG_ROWS};\n\
         constant constexpr uint SW_SG_COLS = {SG_COLS};\n\
         constant constexpr uint SW_N_THREADS = {N_THREADS};\n\
         constant constexpr uint SW_TM = {TM};\n\
         constant constexpr uint SW_TN = {TN};\n\
         constant constexpr uint SW_BUFS = {SW_BUFS};\n\
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
        "kernel void gemm_ko_{label}(\n\
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

    // Shared memory
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

    // Main compute loop — always double-buffered
    generate_double_buffered_loop(&mut s, wide_load, aligned);

    // Store path
    if direct_store {
        generate_direct_store(&mut s, aligned);
    } else {
        generate_scratch_store(&mut s, aligned);
    }

    // Close kernel
    s.push_str("}\n");

    s
}

/// Generate the load code for A matrix (either half4 or 2x half4, with or without bounds checks)
fn generate_a_load(
    s: &mut String,
    wide_load: bool,
    aligned: bool,
    stage_expr: &str,
    k_offset_expr: &str,
) {
    if wide_load {
        // 2x half4 loads (8 elements per iteration)
        if aligned {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BM * SW_BK) / 8; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 8;\n\
                 \x20           uint r = elem / SW_BK;\n\
                 \x20           uint c = elem % SW_BK;\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&A_batch[(row_start + r) * uK + {k_offset_expr} + c]);\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c + 4]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&A_batch[(row_start + r) * uK + {k_offset_expr} + c + 4]);\n\
                 \x20       }}\n"
            ));
        } else {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BM * SW_BK) / 8; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 8;\n\
                 \x20           uint r = elem / SW_BK;\n\
                 \x20           uint c = elem % SW_BK;\n\
                 \x20           uint gr = row_start + r;\n\
                 \x20           uint gc = {k_offset_expr} + c;\n\
                 \x20           if (gr < uM && gc + 7 < uK) {{\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc]);\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c + 4]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc + 4]);\n\
                 \x20           }} else {{\n\
                 \x20               for (uint d = 0; d < 8; d++) {{\n\
                 \x20                   As[{stage_expr}][r * SW_A_STRIDE + c + d] = (gr < uM && gc + d < uK)\n\
                 \x20                       ? A_batch[gr * uK + gc + d] : half(0);\n\
                 \x20               }}\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    } else {
        // half4 loads (reference)
        if aligned {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BM * SW_BK) / 4; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 4;\n\
                 \x20           uint r = elem / SW_BK;\n\
                 \x20           uint c = elem % SW_BK;\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&A_batch[(row_start + r) * uK + {k_offset_expr} + c]);\n\
                 \x20       }}\n"
            ));
        } else {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BM * SW_BK) / 4; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 4;\n\
                 \x20           uint r = elem / SW_BK;\n\
                 \x20           uint c = elem % SW_BK;\n\
                 \x20           uint gr = row_start + r;\n\
                 \x20           uint gc = {k_offset_expr} + c;\n\
                 \x20           if (gr < uM && gc + 3 < uK) {{\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&As[{stage_expr}][r * SW_A_STRIDE + c]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + gc]);\n\
                 \x20           }} else {{\n\
                 \x20               for (uint d = 0; d < 4; d++) {{\n\
                 \x20                   As[{stage_expr}][r * SW_A_STRIDE + c + d] = (gr < uM && gc + d < uK)\n\
                 \x20                       ? A_batch[gr * uK + gc + d] : half(0);\n\
                 \x20               }}\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    }
}

/// Generate the load code for B matrix (either half4 or half8, with or without bounds checks)
fn generate_b_load(
    s: &mut String,
    wide_load: bool,
    aligned: bool,
    stage_expr: &str,
    k_offset_expr: &str,
) {
    if wide_load {
        // 2x half4 loads (8 elements per iteration)
        if aligned {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BK * SW_BN) / 8; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 8;\n\
                 \x20           uint r = elem / SW_BN;\n\
                 \x20           uint c = elem % SW_BN;\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&B_batch[({k_offset_expr} + r) * uN + col_start + c]);\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c + 4]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&B_batch[({k_offset_expr} + r) * uN + col_start + c + 4]);\n\
                 \x20       }}\n"
            ));
        } else {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BK * SW_BN) / 8; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 8;\n\
                 \x20           uint r = elem / SW_BN;\n\
                 \x20           uint c = elem % SW_BN;\n\
                 \x20           uint gr = {k_offset_expr} + r;\n\
                 \x20           uint gc = col_start + c;\n\
                 \x20           if (gr < uK && gc + 7 < uN) {{\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c + 4]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc + 4]);\n\
                 \x20           }} else {{\n\
                 \x20               for (uint d = 0; d < 8; d++) {{\n\
                 \x20                   Bs[{stage_expr}][r * SW_B_STRIDE + c + d] = (gr < uK && gc + d < uN)\n\
                 \x20                       ? B_batch[gr * uN + gc + d] : half(0);\n\
                 \x20               }}\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    } else {
        // half4 loads (reference)
        if aligned {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BK * SW_BN) / 4; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 4;\n\
                 \x20           uint r = elem / SW_BN;\n\
                 \x20           uint c = elem % SW_BN;\n\
                 \x20           *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c]) =\n\
                 \x20               *reinterpret_cast<device const half4*>(&B_batch[({k_offset_expr} + r) * uN + col_start + c]);\n\
                 \x20       }}\n"
            ));
        } else {
            s.push_str(&format!(
                "        for (uint idx = tid_in_group; idx < (SW_BK * SW_BN) / 4; idx += SW_N_THREADS) {{\n\
                 \x20           uint elem = idx * 4;\n\
                 \x20           uint r = elem / SW_BN;\n\
                 \x20           uint c = elem % SW_BN;\n\
                 \x20           uint gr = {k_offset_expr} + r;\n\
                 \x20           uint gc = col_start + c;\n\
                 \x20           if (gr < uK && gc + 3 < uN) {{\n\
                 \x20               *reinterpret_cast<threadgroup half4*>(&Bs[{stage_expr}][r * SW_B_STRIDE + c]) =\n\
                 \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
                 \x20           }} else {{\n\
                 \x20               for (uint d = 0; d < 4; d++) {{\n\
                 \x20                   Bs[{stage_expr}][r * SW_B_STRIDE + c + d] = (gr < uK && gc + d < uN)\n\
                 \x20                       ? B_batch[gr * uN + gc + d] : half(0);\n\
                 \x20               }}\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    }
}

fn generate_double_buffered_loop(s: &mut String, wide_load: bool, aligned: bool) {
    // Prefetch first tile into buffer[0]
    s.push_str("    // ── Prefetch first tile into buffer[0] ──\n");
    generate_a_load(s, wide_load, aligned, "0", "0");
    generate_b_load(s, wide_load, aligned, "0", "0");
    s.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");

    // Main double-buffered loop
    s.push_str(
        "    // ── Main loop: double-buffered ──\n\
         \x20   for (uint tile = 0; tile < n_tiles; tile++) {\n\
         \x20       uint stage = tile & 1;\n\
         \x20       uint next_stage = 1 - stage;\n\
         \x20       uint next_kb = (tile + 1) * SW_BK;\n\n\
         \x20       // Prefetch next tile\n\
         \x20       if (tile + 1 < n_tiles) {\n",
    );
    generate_a_load(s, wide_load, aligned, "next_stage", "next_kb");
    generate_b_load(s, wide_load, aligned, "next_stage", "next_kb");
    s.push_str("        }\n\n");

    // Compute on current stage
    s.push_str(&format!(
        "        // Compute on buffer[stage]\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint kk = 0; kk < SW_BK; kk += 8) {{\n\
         \x20           simdgroup_half8x8 a_frag[SW_TM];\n\
         \x20           simdgroup_half8x8 b_frag[SW_TN];\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20               simdgroup_load(a_frag[i],\n\
         \x20                   &As[stage][(sg_row * {SG_SUB_M} + i * 8) * SW_A_STRIDE + kk], SW_A_STRIDE);\n\
         \x20           }}\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20               simdgroup_load(b_frag[j],\n\
         \x20                   &Bs[stage][kk * SW_B_STRIDE + sg_col * {SG_SUB_N} + j * 8], SW_B_STRIDE);\n\
         \x20           }}\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < SW_TM; i++)\n\
         \x20               #pragma clang loop unroll(full)\n\
         \x20               for (uint j = 0; j < SW_TN; j++)\n\
         \x20                   simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);\n\
         \x20       }}\n\n\
         \x20       if (tile + 1 < n_tiles) {{\n\
         \x20           threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20       }}\n\
         \x20   }}\n",
    ));
}

fn generate_direct_store(s: &mut String, aligned: bool) {
    s.push_str(&format!(
        "\n    // ── Store results: direct store from simdgroup registers ──\n\
         \x20   // MLX-style lane-to-coordinate mapping for simdgroup_float8x8\n\
         \x20   const uint qid = lane_id / 4;\n\
         \x20   const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);\n\
         \x20   const uint fn = (qid & 2u) * 2u + (lane_id % 2u) * 2u;\n\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20           uint base_row = row_start + sg_row * {SG_SUB_M} + i * 8;\n\
         \x20           uint base_col = col_start + sg_col * {SG_SUB_N} + j * 8;\n\n\
         \x20           uint gr = base_row + fm;\n\
         \x20           uint gc0 = base_col + fn;\n\
         \x20           uint gc1 = gc0 + 1;\n\n\
         \x20           auto elems = acc[i][j].thread_elements();\n\n"
    ));

    if aligned {
        s.push_str(
            "            C_batch[gr * uN + gc0] = half(elems[0]);\n\
             \x20           C_batch[gr * uN + gc1] = half(elems[1]);\n",
        );
    } else {
        s.push_str(
            "            if (gr < uM && gc0 < uN) {\n\
             \x20               C_batch[gr * uN + gc0] = half(elems[0]);\n\
             \x20           }\n\
             \x20           if (gr < uM && gc1 < uN) {\n\
             \x20               C_batch[gr * uN + gc1] = half(elems[1]);\n\
             \x20           }\n",
        );
    }

    s.push_str(
        "        }\n\
         \x20   }\n",
    );
}

fn generate_scratch_store(s: &mut String, aligned: bool) {
    s.push_str(&format!(
        "\n    // ── Store results: scratch buffer path ──\n\
         \x20   threadgroup float result_scratch[SW_N_SG * 64];\n\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < SW_TM; i++) {{\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < SW_TN; j++) {{\n\
         \x20           simdgroup_store(acc[i][j], &result_scratch[sgid * 64], 8);\n\
         \x20           threadgroup_barrier(mem_flags::mem_threadgroup);\n\n\
         \x20           uint base_row = row_start + sg_row * {SG_SUB_M} + i * 8;\n\
         \x20           uint base_col = col_start + sg_col * {SG_SUB_N} + j * 8;\n\n"
    ));

    if aligned {
        s.push_str(
            "            for (uint idx = lane_id; idx < 64; idx += 32) {\n\
             \x20               uint lr = idx / 8;\n\
             \x20               uint lc = idx % 8;\n\
             \x20               C_batch[(base_row + lr) * uN + (base_col + lc)] =\n\
             \x20                   half(result_scratch[sgid * 64 + lr * 8 + lc]);\n\
             \x20           }\n",
        );
    } else {
        s.push_str(
            "            for (uint idx = lane_id; idx < 64; idx += 32) {\n\
             \x20               uint lr = idx / 8;\n\
             \x20               uint lc = idx % 8;\n\
             \x20               uint gr = base_row + lr;\n\
             \x20               uint gc = base_col + lc;\n\
             \x20               if (gr < uM && gc < uN) {\n\
             \x20                   C_batch[gr * uN + gc] = half(result_scratch[sgid * 64 + lr * 8 + lc]);\n\
             \x20               }\n\
             \x20           }\n"
        );
    }

    s.push_str(
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20       }\n\
         \x20   }\n",
    );
}

// ---------------------------------------------------------------------------
// MLX-architecture kernel generator (variant 7)
// BM=64, BN=64, BK=16, WM=1, WN=2 (2 SG), 64 threads, single buffer,
// wide loads (4×half4 = 16 elements/thread), direct store, serpentine MMA
// ---------------------------------------------------------------------------

fn generate_mlx_arch_shader() -> String {
    let mut s = String::with_capacity(16384);

    // Header
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Constants
    s.push_str(
        "constant constexpr uint MLX_BM = 64;\n\
         constant constexpr uint MLX_BN = 64;\n\
         constant constexpr uint MLX_BK = 16;\n\
         constant constexpr uint MLX_N_SG = 2;\n\
         constant constexpr uint MLX_N_THREADS = 64;\n\
         constant constexpr uint MLX_TM = 8;   // BM / 8 = 64/8\n\
         constant constexpr uint MLX_TN = 4;   // (BN/2) / 8 = 32/8\n\n",
    );

    // Uniform hint
    s.push_str(
        "#if __METAL_VERSION__ >= 310\n\
         template <typename T>\n\
         METAL_FUNC uniform<T> mlx_as_uniform(T val) {\n\
             return make_uniform(val);\n\
         }\n\
         #else\n\
         template <typename T>\n\
         METAL_FUNC T mlx_as_uniform(T val) {\n\
             return val;\n\
         }\n\
         #endif\n\n",
    );

    // Swizzle helper
    s.push_str(
        "inline uint2 mlx_swizzle_tg(uint2 tid, uint swizzle_log) {\n\
             if (swizzle_log == 0) return tid;\n\
             return uint2(\n\
                 tid.x >> swizzle_log,\n\
                 (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))\n\
             );\n\
         }\n\n",
    );

    // Kernel function signature — same buffer layout as other variants
    s.push_str(
        "kernel void gemm_ko_mlx_arch(\n\
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
         {\n",
    );

    // Threadgroup memory — single buffer (no double buffering)
    s.push_str(
        "    threadgroup half As[MLX_BM * MLX_BK];  // 64×16 = 1024 halves = 2KB\n\
         \x20   threadgroup half Bs[MLX_BK * MLX_BN];  // 16×64 = 1024 halves = 2KB\n\n",
    );

    // Batch / swizzle setup
    s.push_str(
        "    const uint batch_idx = group_id.z;\n\
         \x20   uint2 swizzled = mlx_swizzle_tg(uint2(group_id.x, group_id.y), swizzle_log);\n\
         \x20   const uint row_start = swizzled.y * mlx_as_uniform(MLX_BM);\n\
         \x20   const uint col_start = swizzled.x * mlx_as_uniform(MLX_BN);\n\n\
         \x20   device const half* A_batch = A + batch_idx * mlx_as_uniform(batch_stride_a);\n\
         \x20   device const half* B_batch = B + batch_idx * mlx_as_uniform(batch_stride_b);\n\
         \x20   device half*       C_batch = C + batch_idx * mlx_as_uniform(batch_stride_c);\n\n",
    );

    // SG position: 1×2 grid (WM=1, WN=2)
    s.push_str(
        "    // SG grid: 1×2 — sg_row always 0, sg_col = sgid (0 or 1)\n\
         \x20   const uint base_m = 0;        // WM=1, single row of SG\n\
         \x20   const uint base_n = sgid * 32; // each SG covers 32 cols\n\n",
    );

    // Accumulators: TM=8, TN=4
    s.push_str(
        "    simdgroup_float8x8 acc[MLX_TM][MLX_TN];\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < MLX_TM; i++)\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < MLX_TN; j++)\n\
         \x20           acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\n",
    );

    // Uniform dimension copies
    s.push_str(
        "    const uint uK = mlx_as_uniform(K);\n\
         \x20   const uint uM = mlx_as_uniform(M);\n\
         \x20   const uint uN = mlx_as_uniform(N);\n\
         \x20   const uint n_tiles = (uK + MLX_BK - 1) / MLX_BK;\n\n",
    );

    // ── Main loop: single-buffered ──
    // barrier → load A+B → barrier → MMA → advance
    s.push_str(
        "    // ── Main loop: single-buffered ──\n\
         \x20   for (uint tile = 0; tile < n_tiles; tile++) {\n\
         \x20       uint kb = tile * MLX_BK;\n\n",
    );

    // Load A: each of 64 threads loads one row of BK=16 elements (4×half4)
    s.push_str(
        "        // Load A tile: 64 threads × 16 elements = 64×16 = BM×BK\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20       {\n\
         \x20           uint a_row = tid_in_group;  // 0..63\n\
         \x20           uint gr = row_start + a_row;\n\
         \x20           if (gr < uM && kb + 15 < uK) {\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&As[a_row * 16]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + kb]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&As[a_row * 16 + 4]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + kb + 4]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&As[a_row * 16 + 8]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + kb + 8]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&As[a_row * 16 + 12]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&A_batch[gr * uK + kb + 12]);\n\
         \x20           } else {\n\
         \x20               for (uint d = 0; d < 16; d++) {\n\
         \x20                   As[a_row * 16 + d] = (gr < uM && kb + d < uK)\n\
         \x20                       ? A_batch[gr * uK + kb + d] : half(0);\n\
         \x20               }\n\
         \x20           }\n\
         \x20       }\n\n",
    );

    // Load B: 16 rows × 64 cols = 1024 elements, 64 threads, each loads 16 elements
    // Map: bi = tid_in_group / 4, bj = (tid_in_group % 4) * 16
    s.push_str(
        "        // Load B tile: 64 threads × 16 elements = 16×64 = BK×BN\n\
         \x20       {\n\
         \x20           uint bi = tid_in_group >> 2;         // 0..15 (row in B tile)\n\
         \x20           uint bj = (tid_in_group & 3u) << 4;  // 0, 16, 32, 48\n\
         \x20           uint gr = kb + bi;\n\
         \x20           uint gc = col_start + bj;\n\
         \x20           if (gr < uK && gc + 15 < uN) {\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&Bs[bi * 64 + bj]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&Bs[bi * 64 + bj + 4]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc + 4]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&Bs[bi * 64 + bj + 8]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc + 8]);\n\
         \x20               *reinterpret_cast<threadgroup half4*>(&Bs[bi * 64 + bj + 12]) =\n\
         \x20                   *reinterpret_cast<device const half4*>(&B_batch[gr * uN + gc + 12]);\n\
         \x20           } else {\n\
         \x20               for (uint d = 0; d < 16; d++) {\n\
         \x20                   Bs[bi * 64 + bj + d] = (gr < uK && gc + d < uN)\n\
         \x20                       ? B_batch[gr * uN + gc + d] : half(0);\n\
         \x20               }\n\
         \x20           }\n\
         \x20       }\n\n",
    );

    // Second barrier before compute
    s.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");

    // MMA compute with serpentine — BK/8 = 16/8 = 2 inner iterations
    s.push_str(
        "        // MMA compute (serpentine)\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint kk = 0; kk < 2; kk++) {\n\
         \x20           simdgroup_half8x8 a_frag[MLX_TM];\n\
         \x20           simdgroup_half8x8 b_frag[MLX_TN];\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < MLX_TM; i++) {\n\
         \x20               simdgroup_load(a_frag[i],\n\
         \x20                   &As[(base_m + i * 8) * 16 + kk * 8], 16);\n\
         \x20           }\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint j = 0; j < MLX_TN; j++) {\n\
         \x20               simdgroup_load(b_frag[j],\n\
         \x20                   &Bs[kk * 8 * 64 + (base_n + j * 8)], 64);\n\
         \x20           }\n\n\
         \x20           #pragma clang loop unroll(full)\n\
         \x20           for (uint i = 0; i < MLX_TM; i++) {\n\
         \x20               #pragma clang loop unroll(full)\n\
         \x20               for (uint j = 0; j < MLX_TN; j++) {\n\
         \x20                   uint n_serp = (i % 2) ? (3 - j) : j;\n\
         \x20                   simdgroup_multiply_accumulate(\n\
         \x20                       acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);\n\
         \x20               }\n\
         \x20           }\n\
         \x20       }\n",
    );

    // Close main loop
    s.push_str("    }\n\n");

    // Direct store — MLX lane mapping
    s.push_str(
        "    // ── Store results: direct store from simdgroup registers ──\n\
         \x20   const uint qid = lane_id / 4;\n\
         \x20   const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);\n\
         \x20   const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;\n\n\
         \x20   #pragma clang loop unroll(full)\n\
         \x20   for (uint i = 0; i < MLX_TM; i++) {\n\
         \x20       #pragma clang loop unroll(full)\n\
         \x20       for (uint j = 0; j < MLX_TN; j++) {\n\
         \x20           uint gr = row_start + base_m + i * 8 + fm;\n\
         \x20           uint gc0 = col_start + base_n + j * 8 + fn_val;\n\
         \x20           uint gc1 = gc0 + 1;\n\n\
         \x20           auto elems = acc[i][j].thread_elements();\n\n\
         \x20           if (gr < uM && gc0 < uN) {\n\
         \x20               C_batch[gr * uN + gc0] = half(elems[0]);\n\
         \x20           }\n\
         \x20           if (gr < uM && gc1 < uN) {\n\
         \x20               C_batch[gr * uN + gc1] = half(elems[1]);\n\
         \x20           }\n\
         \x20       }\n\
         \x20   }\n",
    );

    // Close kernel
    s.push_str("}\n");

    s
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

fn make_u32_buf(device: &metal::Device, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
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
    queue: &metal::CommandQueue,
    device: &metal::Device,
    cfg: &GemmKernelOptConfig,
    m: usize,
    k: usize,
    n: usize,
) {
    let kernel_name = format!("gemm_ko_{}", cfg.label);
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
    let swizzle_buf = make_u32_buf(device, compute_swizzle_log(m, BM as usize));

    let grid_x = ceil_div(n, BN as usize) as u64;
    let grid_y = ceil_div(m, BM as usize) as u64;
    let grid = MTLSize::new(grid_x, grid_y, 1);
    // mlx_arch uses 64 threads; all other variants use N_THREADS (256)
    let n_threads = if cfg.label == "mlx_arch" {
        64u64
    } else {
        N_THREADS as u64
    };
    let tg = MTLSize::new(n_threads, 1, 1);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(c.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa_buf), 0);
        enc.set_buffer(7, Some(&bsb_buf), 0);
        enc.set_buffer(8, Some(&bsc_buf), 0);
        enc.set_buffer(9, Some(&swizzle_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(c.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa_buf), 0);
        enc.set_buffer(7, Some(&bsb_buf), 0);
        enc.set_buffer(8, Some(&bsc_buf), 0);
        enc.set_buffer(9, Some(&swizzle_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
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
    let queue = device.new_command_queue();

    // Register all optimization kernels via JIT
    for cfg in CONFIGS {
        let jit_name = format!("gemm_ko_{}", cfg.label);
        if cfg.label == "mlx_arch" {
            // MLX-architecture variant uses its own dedicated shader generator
            let source = generate_mlx_arch_shader();
            registry
                .register_jit_source(&jit_name, &source)
                .unwrap_or_else(|e| panic!("Failed to compile {}: {e}", cfg.label));
        } else {
            let source = generate_kernel_opt_shader(cfg);
            registry
                .register_jit_source(&jit_name, &source)
                .unwrap_or_else(|e| panic!("Failed to compile {}: {e}", cfg.label));
        }
    }

    // ── Correctness verification ──
    // Run each variant on M=128,K=128,N=128 and compare against ref
    println!("=== Correctness Verification (M=128, K=128, N=128) ===");
    {
        let vm = 128usize;
        let vk = 128usize;
        let vn = 128usize;
        let va = rand_array(device, &[vm, vk], 42);
        let vb = rand_array(device, &[vk, vn], 43);

        // Run ref kernel to get ground truth
        let ref_pipeline = registry
            .get_pipeline("gemm_ko_ref", DType::Float16)
            .expect("ref pipeline");
        let c_ref = Array::zeros(device, &[vm, vn], DType::Float16);
        let m_buf = make_u32_buf(device, vm as u32);
        let n_buf = make_u32_buf(device, vn as u32);
        let k_buf = make_u32_buf(device, vk as u32);
        let bsa_buf = make_u32_buf(device, (vm * vk) as u32);
        let bsb_buf = make_u32_buf(device, (vk * vn) as u32);
        let bsc_buf = make_u32_buf(device, (vm * vn) as u32);
        let sw_buf = make_u32_buf(device, 0u32);
        let grid = MTLSize::new(
            ceil_div(vn, BN as usize) as u64,
            ceil_div(vm, BM as usize) as u64,
            1,
        );
        let tg = MTLSize::new(N_THREADS as u64, 1, 1);

        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ref_pipeline);
        enc.set_buffer(0, Some(va.metal_buffer()), 0);
        enc.set_buffer(1, Some(vb.metal_buffer()), 0);
        enc.set_buffer(2, Some(c_ref.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa_buf), 0);
        enc.set_buffer(7, Some(&bsb_buf), 0);
        enc.set_buffer(8, Some(&bsc_buf), 0);
        enc.set_buffer(9, Some(&sw_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let ref_ptr = c_ref.metal_buffer().contents() as *const u16;
        let ref_vals: Vec<u16> = unsafe { std::slice::from_raw_parts(ref_ptr, vm * vn) }.to_vec();

        // Test each non-ref variant
        for cfg in &CONFIGS[1..] {
            let kernel_name = format!("gemm_ko_{}", cfg.label);
            let pipeline = registry
                .get_pipeline(&kernel_name, DType::Float16)
                .unwrap_or_else(|_| panic!("{} pipeline", cfg.label));
            let c_test = Array::zeros(device, &[vm, vn], DType::Float16);

            // mlx_arch uses 64 threads; all other variants use N_THREADS (256)
            let verify_tg = if cfg.label == "mlx_arch" {
                MTLSize::new(64, 1, 1)
            } else {
                tg
            };

            let cb = queue.new_command_buffer_with_unretained_references();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(va.metal_buffer()), 0);
            enc.set_buffer(1, Some(vb.metal_buffer()), 0);
            enc.set_buffer(2, Some(c_test.metal_buffer()), 0);
            enc.set_buffer(3, Some(&m_buf), 0);
            enc.set_buffer(4, Some(&n_buf), 0);
            enc.set_buffer(5, Some(&k_buf), 0);
            enc.set_buffer(6, Some(&bsa_buf), 0);
            enc.set_buffer(7, Some(&bsb_buf), 0);
            enc.set_buffer(8, Some(&bsc_buf), 0);
            enc.set_buffer(9, Some(&sw_buf), 0);
            enc.dispatch_thread_groups(grid, verify_tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let test_ptr = c_test.metal_buffer().contents() as *const u16;
            let test_vals: Vec<u16> =
                unsafe { std::slice::from_raw_parts(test_ptr, vm * vn) }.to_vec();

            let mut mismatches = 0u32;
            let mut first_mm: Option<(usize, u16, u16)> = None;
            for i in 0..ref_vals.len() {
                if ref_vals[i] != test_vals[i] {
                    mismatches += 1;
                    if first_mm.is_none() {
                        first_mm = Some((i, ref_vals[i], test_vals[i]));
                    }
                }
            }

            if mismatches == 0 {
                println!("  {} : PASS (exact match)", cfg.label);
            } else {
                let (idx, rv, tv) = first_mm.unwrap();
                let row = idx / vn;
                let col = idx % vn;
                println!(
                    "  {} : FAIL — {} mismatches / {} (first: [{},{}] ref=0x{:04x} test=0x{:04x})",
                    cfg.label,
                    mismatches,
                    ref_vals.len(),
                    row,
                    col,
                    rv,
                    tv,
                );
            }
        }
    }
    println!();

    println!("=== GEMM Kernel Optimization Benchmark (f16) ===");
    println!(
        "Base: BM={} BN={} BK={} SG={}x{} thr={} double-buffered PAD={}",
        BM, BN, BK, SG_ROWS, SG_COLS, N_THREADS, PAD,
    );
    println!(
        "Variants: {}, Warmup: {}, Bench: {}",
        CONFIGS.len(),
        WARMUP_ITERS,
        BENCH_ITERS
    );
    println!();

    let m_values = [128, 256, 512, 1024, 2048];
    let kn_combos: &[(usize, usize)] = &[(4096, 4096), (4096, 14336)];

    for cfg in CONFIGS {
        let opts: Vec<&str> = [
            if cfg.use_direct_store {
                Some("direct_store")
            } else {
                None
            },
            if cfg.use_wide_load {
                Some("wide_load")
            } else {
                None
            },
            if cfg.use_aligned {
                Some("aligned")
            } else {
                None
            },
        ]
        .iter()
        .filter_map(|x| *x)
        .collect();

        let opts_str = if opts.is_empty() {
            "none (reference)".to_string()
        } else {
            opts.join(" + ")
        };

        println!("--- Config: {} [{}] ---", cfg.label, opts_str);
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
