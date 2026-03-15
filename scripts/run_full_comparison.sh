#!/usr/bin/env bash
# run_full_comparison.sh — Comprehensive RMLX vs MLX benchmark suite.
# Runs single-node, TP=2, EP=2 for both decode and prefill paths.
# Collects all results into a timestamped directory and prints a summary table.
set -euo pipefail

# ─── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Comprehensive RMLX vs MLX benchmark comparison suite.
Runs single-node, TP=2, EP=2 for both decode and prefill paths.
Syncs code, builds on both machines, runs all benchmarks, collects results.

Options:
    --node0 HOST       SSH hostname for rank 0 (default: \$RMLX_NODE0 or node0)
    --node1 HOST       SSH hostname for rank 1 (default: \$RMLX_NODE1 or node1)
    --node0-ip IP      RDMA IP for rank 0 (default: \$RMLX_NODE0_IP or 10.0.0.1)
    --node1-ip IP      RDMA IP for rank 1 (default: \$RMLX_NODE1_IP or 10.0.0.2)
    --remote-dir DIR   Remote project directory (default: \$RMLX_REMOTE_DIR or \$HOME/rmlx)
    --decode-only      Skip prefill benchmarks
    --ep-only          Only EP benchmarks (single-node + 2-node)
    --tp-only          Only TP benchmarks (single-node + 2-node)
    --mlx-only         Only MLX benchmarks
    --rmlx-only        Only RMLX benchmarks
    --skip-sync        Skip code sync and build (use existing binaries)
    --help, -h         Show this help message

Environment variables (override defaults, overridden by CLI args):
    RMLX_NODE0, RMLX_NODE1, RMLX_NODE0_IP, RMLX_NODE1_IP, RMLX_REMOTE_DIR
    RMLX_COORDINATOR_PORT (default: 18520)

Benchmark matrix:
    ┌──────────────────┬──────────────────┬──────────────────┐
    │                  │  RMLX (Rust)     │  MLX (Python)    │
    ├──────────────────┼──────────────────┼──────────────────┤
    │ Single-node dec  │ distributed_bench│ mlx_tp_bench.py  │
    │ TP=2 decode      │ distributed_bench│ mlx_tp_bench.py  │
    │ EP=2 decode      │ ep_bench         │ (simulated)      │
    │ Single-node pref │ e2e_prefill_bench│ mlx_prefill_bench│
    └──────────────────┴──────────────────┴──────────────────┘
USAGE
    exit 0
}

# ─── Parse args ────────────────────────────────────────────────────────
_NODE0=""
_NODE1=""
_NODE0_IP=""
_NODE1_IP=""
_REMOTE_DIR=""
DECODE_ONLY=false
EP_ONLY=false
TP_ONLY=false
MLX_ONLY=false
RMLX_ONLY=false
SKIP_SYNC=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)       _NODE0="$2"; shift 2 ;;
        --node1)       _NODE1="$2"; shift 2 ;;
        --node0-ip)    _NODE0_IP="$2"; shift 2 ;;
        --node1-ip)    _NODE1_IP="$2"; shift 2 ;;
        --remote-dir)  _REMOTE_DIR="$2"; shift 2 ;;
        --decode-only) DECODE_ONLY=true; shift ;;
        --ep-only)     EP_ONLY=true; shift ;;
        --tp-only)     TP_ONLY=true; shift ;;
        --mlx-only)    MLX_ONLY=true; shift ;;
        --rmlx-only)   RMLX_ONLY=true; shift ;;
        --skip-sync)   SKIP_SYNC=true; shift ;;
        --help|-h)     usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Apply priority: CLI > env > default
NODE0="${_NODE0:-${RMLX_NODE0:-node0}}"
NODE1="${_NODE1:-${RMLX_NODE1:-node1}}"
NODE0_IP="${_NODE0_IP:-${RMLX_NODE0_IP:-10.0.0.1}}"
NODE1_IP="${_NODE1_IP:-${RMLX_NODE1_IP:-10.0.0.2}}"
REMOTE_DIR="${_REMOTE_DIR:-${RMLX_REMOTE_DIR:-\$HOME/rmlx}}"

# ─── Configuration ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COORDINATOR_PORT="${RMLX_COORDINATOR_PORT:-18520}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="$PROJECT_DIR/bench_results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# RDMA environment for distributed benchmarks
COMMON_ENV="RMLX_RDMA_DEVICE=rdma_en5 \
RMLX_WORLD_SIZE=2 \
RMLX_COORDINATOR=$NODE0_IP \
RMLX_COORDINATOR_PORT=$COORDINATOR_PORT \
RMLX_IBV_DEVICES=$REMOTE_DIR/config/devices_tb5.json"

# ─── Filter helpers ──────────────────────────────────────────────────────
should_run_rmlx() { [ "$MLX_ONLY" = false ]; }
should_run_mlx()  { [ "$RMLX_ONLY" = false ]; }
should_run_tp()   { [ "$EP_ONLY" = false ]; }
should_run_ep()   { [ "$TP_ONLY" = false ]; }
should_run_prefill() { [ "$DECODE_ONLY" = false ]; }

# ─── Helpers ────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

ssh_cmd() {
    local host="$1"
    shift
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
}

# Track results for summary table
declare -a SUMMARY_LABELS=()
declare -a SUMMARY_FILES=()

record_result() {
    local label="$1"
    local file="$2"
    SUMMARY_LABELS+=("$label")
    SUMMARY_FILES+=("$file")
}

# ─── Phase 1: Sync & Build ──────────────────────────────────────────────
sync_and_build() {
    if [ "$SKIP_SYNC" = true ]; then
        log "Skipping sync and build (--skip-sync)"
        return
    fi

    log "=== Phase 1: Sync & Build ==="

    # Check connectivity
    log "Checking connectivity to $NODE0 and $NODE1..."
    if ! ssh_cmd "$NODE0" true 2>/dev/null; then
        echo "ERROR: Cannot SSH to $NODE0" >&2
        exit 1
    fi
    if ! ssh_cmd "$NODE1" true 2>/dev/null; then
        echo "ERROR: Cannot SSH to $NODE1" >&2
        exit 1
    fi
    log "Both hosts reachable."

    # Sync code
    log "Syncing code to $NODE0..."
    rsync -az --delete \
        --exclude target \
        --exclude .git \
        --exclude bench_results \
        "$PROJECT_DIR/" "$NODE0:$REMOTE_DIR/"

    log "Syncing code to $NODE1..."
    rsync -az --delete \
        --exclude target \
        --exclude .git \
        --exclude bench_results \
        "$PROJECT_DIR/" "$NODE1:$REMOTE_DIR/"
    log "Code synced."

    # Build on both machines (parallel)
    # Build all bench targets we might need
    log "Building on both machines (parallel)..."
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR cargo build --release -p rmlx-nn \
        --bench distributed_bench --bench ep_bench --bench e2e_prefill_bench \
        --features distributed 2>&1" &
    PID1=$!
    ssh_cmd "$NODE1" "env -C $REMOTE_DIR cargo build --release -p rmlx-nn \
        --bench distributed_bench --bench ep_bench --bench e2e_prefill_bench \
        --features distributed 2>&1" &
    PID2=$!

    wait $PID1 || { echo "ERROR: Build failed on $NODE0" >&2; exit 1; }
    wait $PID2 || { echo "ERROR: Build failed on $NODE1" >&2; exit 1; }
    log "Build complete on both machines."
}

# ─── Phase 2a: RMLX Single-Node Decode ──────────────────────────────────
run_rmlx_single_decode() {
    should_run_rmlx || return 0
    should_run_tp || return 0

    local outfile="$RESULTS_DIR/rmlx_single_decode.txt"
    log "=== RMLX Single-Node Decode (distributed_bench baseline) ==="
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        cargo bench -p rmlx-nn --bench distributed_bench --features distributed 2>&1" \
        | tee "$outfile"
    record_result "RMLX single-node decode" "$outfile"
    log "Done: RMLX single-node decode"
}

# ─── Phase 2b: MLX Single-Node Decode ───────────────────────────────────
run_mlx_single_decode() {
    should_run_mlx || return 0
    should_run_tp || return 0

    local outfile="$RESULTS_DIR/mlx_single_decode.txt"
    log "=== MLX Single-Node Decode (mlx_tp_bench.py) ==="
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        python3 benchmarks/mlx_tp_bench.py 2>&1" \
        | tee "$outfile"
    record_result "MLX single-node decode" "$outfile"
    log "Done: MLX single-node decode"
}

# ─── Phase 3: RMLX 2-Node TP Decode ─────────────────────────────────────
run_rmlx_tp2_decode() {
    should_run_rmlx || return 0
    should_run_tp || return 0

    local outfile_r0="$RESULTS_DIR/rmlx_tp2_decode_rank0.txt"
    local outfile_r1="$RESULTS_DIR/rmlx_tp2_decode_rank1.txt"
    log "=== RMLX 2-Node TP Decode (distributed_bench) ==="
    log "  Rank 0 (coordinator): $NODE0 ($NODE0_IP)"
    log "  Rank 1:               $NODE1 ($NODE1_IP)"

    # Start rank 1 first
    ssh_cmd "$NODE1" "env -C $REMOTE_DIR \
        RMLX_RANK=1 $COMMON_ENV \
        cargo bench -p rmlx-nn --bench distributed_bench --features distributed 2>&1" \
        > "$outfile_r1" 2>&1 &
    RANK1_PID=$!

    sleep 2

    # Start rank 0 (coordinator)
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        RMLX_RANK=0 $COMMON_ENV \
        cargo bench -p rmlx-nn --bench distributed_bench --features distributed 2>&1" \
        | tee "$outfile_r0" &
    RANK0_PID=$!

    wait $RANK0_PID || log "WARNING: TP decode rank 0 exited with error"
    wait $RANK1_PID || log "WARNING: TP decode rank 1 exited with error"

    record_result "RMLX TP=2 decode (rank0)" "$outfile_r0"
    log "Done: RMLX 2-node TP decode"
}

# ─── Phase 4: MLX EP Decode ──────────────────────────────────────────────
run_mlx_ep_decode() {
    should_run_mlx || return 0
    should_run_ep || return 0

    # Check if mlx_ep_bench.py exists; some setups may not have it
    if ! ssh_cmd "$NODE0" "test -f $REMOTE_DIR/benchmarks/mlx_ep_bench.py" 2>/dev/null; then
        log "SKIP: MLX EP decode — benchmarks/mlx_ep_bench.py not found"
        return 0
    fi

    local outfile="$RESULTS_DIR/mlx_ep_decode.txt"
    log "=== MLX EP Decode (mlx_ep_bench.py) ==="
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        python3 benchmarks/mlx_ep_bench.py 2>&1" \
        | tee "$outfile"
    record_result "MLX EP decode" "$outfile"
    log "Done: MLX EP decode"
}

# ─── Phase 5: RMLX 2-Node EP Decode ─────────────────────────────────────
run_rmlx_ep2_decode() {
    should_run_rmlx || return 0
    should_run_ep || return 0

    local outfile_r0="$RESULTS_DIR/rmlx_ep2_decode_rank0.txt"
    local outfile_r1="$RESULTS_DIR/rmlx_ep2_decode_rank1.txt"
    log "=== RMLX 2-Node EP Decode (ep_bench) ==="
    log "  Rank 0 (coordinator): $NODE0 ($NODE0_IP)"
    log "  Rank 1:               $NODE1 ($NODE1_IP)"

    # Start rank 1 first
    ssh_cmd "$NODE1" "env -C $REMOTE_DIR \
        RMLX_RANK=1 $COMMON_ENV \
        cargo bench -p rmlx-nn --bench ep_bench --features distributed 2>&1" \
        > "$outfile_r1" 2>&1 &
    RANK1_PID=$!

    sleep 2

    # Start rank 0 (coordinator)
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        RMLX_RANK=0 $COMMON_ENV \
        cargo bench -p rmlx-nn --bench ep_bench --features distributed 2>&1" \
        | tee "$outfile_r0" &
    RANK0_PID=$!

    wait $RANK0_PID || log "WARNING: EP decode rank 0 exited with error"
    wait $RANK1_PID || log "WARNING: EP decode rank 1 exited with error"

    record_result "RMLX EP=2 decode (rank0)" "$outfile_r0"
    log "Done: RMLX 2-node EP decode"
}

# ─── Phase 6: RMLX Single-Node Prefill ──────────────────────────────────
run_rmlx_single_prefill() {
    should_run_rmlx || return 0
    should_run_prefill || return 0

    local outfile="$RESULTS_DIR/rmlx_single_prefill.txt"
    log "=== RMLX Single-Node Prefill (e2e_prefill_bench) ==="
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        cargo bench -p rmlx-nn --bench e2e_prefill_bench --features distributed 2>&1" \
        | tee "$outfile"
    record_result "RMLX single-node prefill" "$outfile"
    log "Done: RMLX single-node prefill"
}

# ─── Phase 7: MLX Single-Node Prefill ───────────────────────────────────
run_mlx_single_prefill() {
    should_run_mlx || return 0
    should_run_prefill || return 0

    local outfile="$RESULTS_DIR/mlx_single_prefill.txt"
    log "=== MLX Single-Node Prefill (mlx_prefill_bench.py) ==="
    ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
        python3 benchmarks/mlx_prefill_bench.py 2>&1" \
        | tee "$outfile"
    record_result "MLX single-node prefill" "$outfile"
    log "Done: MLX single-node prefill"
}

# ─── Summary ─────────────────────────────────────────────────────────────
print_summary() {
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║              FULL COMPARISON BENCHMARK SUMMARY                 ║"
    log "╠══════════════════════════════════════════════════════════════════╣"
    log "║  Timestamp: $TIMESTAMP                                  ║"
    log "║  Node 0:    $NODE0 ($NODE0_IP)                          ║"
    log "║  Node 1:    $NODE1 ($NODE1_IP)                          ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log ""

    echo ""
    echo "┌─────────────────────────────────┬────────────────────────────────┐"
    echo "│ Benchmark                       │ Result File                    │"
    echo "├─────────────────────────────────┼────────────────────────────────┤"

    for i in "${!SUMMARY_LABELS[@]}"; do
        local label="${SUMMARY_LABELS[$i]}"
        local file="${SUMMARY_FILES[$i]}"
        local basename
        basename=$(basename "$file")
        printf "│ %-31s │ %-30s │\n" "$label" "$basename"
    done

    echo "└─────────────────────────────────┴────────────────────────────────┘"
    echo ""

    # Print condensed results from each file
    for i in "${!SUMMARY_LABELS[@]}"; do
        local label="${SUMMARY_LABELS[$i]}"
        local file="${SUMMARY_FILES[$i]}"
        echo ""
        echo "━━━ ${label} ━━━"
        if [ -f "$file" ]; then
            # Extract lines with timing info (mean=, us, ms, TFLOPS, latency patterns)
            grep -iE '(mean=|median|p50|p95|TFLOPS|tflops|latency|throughput|us/layer|ms/layer|SUMMARY|speedup|benchmark)' "$file" 2>/dev/null \
                || echo "  (no timing lines matched — see raw file)"
        else
            echo "  (file not found)"
        fi
    done

    echo ""
    log "All results saved to: $RESULTS_DIR/"
    log "Done."
}

# ─── Main ────────────────────────────────────────────────────────────────
main() {
    log "Starting full RMLX vs MLX comparison benchmark"
    log "  Filters: decode_only=$DECODE_ONLY ep_only=$EP_ONLY tp_only=$TP_ONLY"
    log "           mlx_only=$MLX_ONLY rmlx_only=$RMLX_ONLY skip_sync=$SKIP_SYNC"
    log "  Results: $RESULTS_DIR"
    echo ""

    # Phase 1: Sync & Build
    sync_and_build

    # Phase 2: Single-node decode (sequential — GPU bench must not overlap)
    run_rmlx_single_decode
    run_mlx_single_decode

    # Phase 3: 2-node TP decode
    run_rmlx_tp2_decode

    # Phase 4-5: EP benchmarks
    run_mlx_ep_decode
    run_rmlx_ep2_decode

    # Phase 6-7: Prefill benchmarks
    run_rmlx_single_prefill
    run_mlx_single_prefill

    # Summary
    print_summary
}

main
