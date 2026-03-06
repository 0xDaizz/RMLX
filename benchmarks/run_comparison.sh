#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# RMLX vs MLX Comparison Benchmark Runner
#
# This script runs three benchmark suites back-to-back:
#   1. RMLX (Rust)         — cargo bench
#   2. Custom MLX (~/mlx)  — Python with custom-built MLX
#   3. Vanilla MLX          — Python with stock MLX from ~/mlx-vanilla
#
# Prerequisites on M3-Ultra-80c:
#   - Custom MLX built and installed:
#       cd ~/mlx && pip install -e .
#     OR built in-tree so that ~/mlx/build/lib.* exists.
#
#   - Vanilla MLX installed separately:
#       cd ~/mlx-vanilla && pip install -e .
#     OR: pip install mlx --target ~/mlx-vanilla/site-packages
#
# PYTHONPATH strategy:
#   We use PYTHONPATH to switch between the two MLX installations.
#   - Custom MLX:  whatever is on the default sys.path (pip install -e ~/mlx)
#   - Vanilla MLX: PYTHONPATH=~/mlx-vanilla/build/lib.* (or site-packages)
#
# Adjust the paths below to match your actual build layout.
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_PY="$SCRIPT_DIR/mlx_comparison_bench.py"

# --- Configurable paths (edit these for your setup) -----------------------
CUSTOM_MLX_DIR="$HOME/mlx"
VANILLA_MLX_DIR="$HOME/mlx-vanilla"

# Python interpreter — change if you use a venv
PYTHON="${PYTHON:-python3}"

# Benchmark parameters
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-100}"

# Cargo / Rust
export PATH="$HOME/.cargo/bin:$PATH"

# ==========================================================================
# Helper
# ==========================================================================
separator() {
    echo ""
    echo "########################################################################"
    echo "# $1"
    echo "########################################################################"
    echo ""
}

# ==========================================================================
# 1. RMLX (Rust) benchmark
# ==========================================================================
separator "RMLX — cargo bench (Rust / Metal)"

cd "$PROJECT_DIR"
cargo bench -p rmlx-core --bench ops_bench 2>&1 || {
    echo "[WARN] cargo bench failed or not yet set up — skipping RMLX."
}

# ==========================================================================
# 2. Custom MLX benchmark
# ==========================================================================
separator "Custom MLX ($CUSTOM_MLX_DIR)"

# Ensure custom MLX is the one resolved (default sys.path, no PYTHONPATH override).
# If you installed with `pip install -e ~/mlx`, it's already on the default path.
unset PYTHONPATH 2>/dev/null || true

$PYTHON "$BENCH_PY" --warmup "$WARMUP" --iters "$ITERS"

# ==========================================================================
# 3. Vanilla MLX benchmark
# ==========================================================================
separator "Vanilla MLX ($VANILLA_MLX_DIR)"

# Strategy: use PYTHONPATH to make the vanilla build take priority.
# Adjust the glob below to match your vanilla MLX's built location.
#
# Option A — editable install in a separate venv:
#   Activate the venv instead:
#     source ~/mlx-vanilla-venv/bin/activate
#
# Option B — PYTHONPATH to the build directory:
#   The built .so lives under build/lib.<platform>-<pyver>/
#   e.g. ~/mlx-vanilla/build/lib.macosx-14.0-arm64-cpython-312/
VANILLA_BUILD="$(ls -d "$VANILLA_MLX_DIR"/build/lib.* 2>/dev/null | head -1 || true)"

if [ -n "$VANILLA_BUILD" ]; then
    echo "Using PYTHONPATH=$VANILLA_BUILD"
    PYTHONPATH="$VANILLA_BUILD" $PYTHON "$BENCH_PY" --warmup "$WARMUP" --iters "$ITERS"
elif [ -d "$VANILLA_MLX_DIR/site-packages" ]; then
    echo "Using PYTHONPATH=$VANILLA_MLX_DIR/site-packages"
    PYTHONPATH="$VANILLA_MLX_DIR/site-packages" $PYTHON "$BENCH_PY" --warmup "$WARMUP" --iters "$ITERS"
elif [ -f "$VANILLA_MLX_DIR/setup.py" ] || [ -f "$VANILLA_MLX_DIR/pyproject.toml" ]; then
    echo "[INFO] Vanilla MLX source found but no build directory detected."
    echo "       Build it first:  cd $VANILLA_MLX_DIR && pip install -e ."
    echo "       Or use a separate venv — edit this script accordingly."
    echo "       Skipping vanilla MLX benchmark."
else
    echo "[WARN] Vanilla MLX not found at $VANILLA_MLX_DIR — skipping."
fi

# ==========================================================================
separator "All benchmarks complete"
