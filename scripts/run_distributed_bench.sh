#!/usr/bin/env bash
# run_distributed_bench.sh — Launch distributed TP/EP benchmark on two nodes.
# Syncs code, builds on both machines, runs single-node baseline then 2-node benchmark.
set -euo pipefail

# ─── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Launch distributed TP/EP benchmark on two nodes. Syncs code, builds on
both machines, runs single-node baseline then 2-node distributed benchmark.

Options:
    --node0 HOST       SSH hostname for rank 0 (default: \$RMLX_NODE0 or node0)
    --node1 HOST       SSH hostname for rank 1 (default: \$RMLX_NODE1 or node1)
    --node0-ip IP      RDMA IP for rank 0 (default: \$RMLX_NODE0_IP or 10.0.0.1)
    --node1-ip IP      RDMA IP for rank 1 (default: \$RMLX_NODE1_IP or 10.0.0.2)
    --remote-dir DIR   Remote project directory (default: \$RMLX_REMOTE_DIR or \$HOME/rmlx)
    --local-only       Single-node baseline only (no SSH needed)
    --ep               EP benchmark instead of TP
    --help, -h         Show this help message

Environment variables (override defaults, overridden by CLI args):
    RMLX_NODE0, RMLX_NODE1, RMLX_NODE0_IP, RMLX_NODE1_IP, RMLX_REMOTE_DIR
    RMLX_COORDINATOR_PORT (default: 18520)
USAGE
    exit 0
}

# ─── Parse args ────────────────────────────────────────────────────────
# Priority: CLI args > env vars > defaults
_NODE0=""
_NODE1=""
_NODE0_IP=""
_NODE1_IP=""
_REMOTE_DIR=""
LOCAL_ONLY=false
RUN_EP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)      _NODE0="$2"; shift 2 ;;
        --node1)      _NODE1="$2"; shift 2 ;;
        --node0-ip)   _NODE0_IP="$2"; shift 2 ;;
        --node1-ip)   _NODE1_IP="$2"; shift 2 ;;
        --remote-dir) _REMOTE_DIR="$2"; shift 2 ;;
        --local-only) LOCAL_ONLY=true; shift ;;
        --ep)         RUN_EP=true; shift ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Apply priority: CLI > env > default
NODE0="${_NODE0:-${RMLX_NODE0:-node0}}"
NODE1="${_NODE1:-${RMLX_NODE1:-node1}}"
NODE0_IP="${_NODE0_IP:-${RMLX_NODE0_IP:-10.0.0.1}}"
NODE1_IP="${_NODE1_IP:-${RMLX_NODE1_IP:-10.0.0.2}}"
REMOTE_DIR="${_REMOTE_DIR:-${RMLX_REMOTE_DIR:-\$HOME/rmlx}}"

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COORDINATOR_PORT="${RMLX_COORDINATOR_PORT:-18520}"
BENCH_NAME="distributed_bench"
RESULTS_DIR="$PROJECT_DIR/bench_results"

if [ "$RUN_EP" = true ]; then
    BENCH_NAME="ep_bench"
fi

# ─── Helpers ────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

ssh_cmd() {
    local host="$1"
    shift
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
}

# ─── Local-only mode ────────────────────────────────────────────────────────
if [ "$LOCAL_ONLY" = true ]; then
    log "Running single-node baseline benchmark locally..."
    mkdir -p "$RESULTS_DIR"
    cd "$PROJECT_DIR"
    cargo bench -p rmlx-nn --bench "$BENCH_NAME" --features distributed 2>&1 \
        | tee "$RESULTS_DIR/local_baseline.txt"
    log "Done. Results in $RESULTS_DIR/local_baseline.txt"
    exit 0
fi

# ─── Check connectivity ────────────────────────────────────────────────────
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

# ─── Sync code to both machines ────────────────────────────────────────────
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

# ─── Build on both machines (parallel) ─────────────────────────────────────
log "Building on both machines (parallel)..."
ssh_cmd "$NODE0" "env -C $REMOTE_DIR cargo build --release -p rmlx-nn --bench $BENCH_NAME --features distributed 2>&1" &
PID1=$!
ssh_cmd "$NODE1" "env -C $REMOTE_DIR cargo build --release -p rmlx-nn --bench $BENCH_NAME --features distributed 2>&1" &
PID2=$!

wait $PID1 || { echo "ERROR: Build failed on $NODE0" >&2; exit 1; }
wait $PID2 || { echo "ERROR: Build failed on $NODE1" >&2; exit 1; }
log "Build complete on both machines."

# ─── Create results directory ──────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ─── Run single-node baseline on node0 first ──────────────────────────
log "Running single-node baseline on $NODE0..."
ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
    cargo bench -p rmlx-nn --bench $BENCH_NAME --features distributed 2>&1" \
    | tee "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_baseline_node0.txt"

# ─── Run distributed benchmark on both machines ───────────────────────────
log "Starting distributed benchmark (2-node)..."
log "  Rank 0 (coordinator): $NODE0 ($NODE0_IP)"
log "  Rank 1:               $NODE1 ($NODE1_IP)"

# Environment variables for distributed init (rmlx_distributed::init)
# Uses RMLX_COORDINATOR for coordinator-mediated QP exchange
COMMON_ENV="RMLX_RDMA_DEVICE=rdma_en5 \
RMLX_WORLD_SIZE=2 \
RMLX_COORDINATOR=$NODE0_IP \
RMLX_COORDINATOR_PORT=$COORDINATOR_PORT \
RMLX_IBV_DEVICES=$REMOTE_DIR/config/devices_tb5.json"

# Start rank 1 first (it will connect to rank 0's coordinator)
log "Starting rank 1 on $NODE1..."
ssh_cmd "$NODE1" "env -C $REMOTE_DIR \
    RMLX_RANK=1 $COMMON_ENV \
    cargo bench -p rmlx-nn --bench $BENCH_NAME --features distributed 2>&1" \
    > "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_tp2_rank1.txt" 2>&1 &
RANK1_PID=$!

# Give rank 1 a moment to start
sleep 2

# Start rank 0 (coordinator)
log "Starting rank 0 on $NODE0..."
ssh_cmd "$NODE0" "env -C $REMOTE_DIR \
    RMLX_RANK=0 $COMMON_ENV \
    cargo bench -p rmlx-nn --bench $BENCH_NAME --features distributed 2>&1" \
    | tee "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_tp2_rank0.txt" &
RANK0_PID=$!

# Wait for both to complete
wait $RANK0_PID
RANK0_STATUS=$?
wait $RANK1_PID
RANK1_STATUS=$?

if [ $RANK0_STATUS -ne 0 ]; then
    log "WARNING: Rank 0 exited with status $RANK0_STATUS"
fi
if [ $RANK1_STATUS -ne 0 ]; then
    log "WARNING: Rank 1 exited with status $RANK1_STATUS"
fi

# ─── Collect results ───────────────────────────────────────────────────────
log "=== Results ==="
echo ""
echo "--- Baseline (single-node, node0) ---"
cat "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_baseline_node0.txt"
echo ""
echo "--- 2-node Rank 0 (node0) ---"
cat "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_tp2_rank0.txt"
echo ""
echo "--- 2-node Rank 1 (node1) ---"
cat "$RESULTS_DIR/${TIMESTAMP}_${BENCH_NAME}_tp2_rank1.txt"
echo ""
log "Results saved to $RESULTS_DIR/"
log "Done."
