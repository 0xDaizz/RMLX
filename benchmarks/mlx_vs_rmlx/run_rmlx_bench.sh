#!/usr/bin/env bash
# RMLX comprehensive RDMA benchmark — 2-node launcher.
# Builds the comprehensive_bench binary on both nodes, then runs a 2-node
# RDMA benchmark using the establish() pattern (same as test suite).
set -euo pipefail

# ─── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

RMLX comprehensive RDMA benchmark — 2-node launcher. Builds the benchmark
binary on both nodes, then runs a 2-node RDMA benchmark.

Options:
    --node0 HOST       SSH hostname for rank 0 (default: \$RMLX_NODE0 or node0)
    --node1 HOST       SSH hostname for rank 1 (default: \$RMLX_NODE1 or node1)
    --node0-ip IP      RDMA IP for rank 0 (default: \$RMLX_NODE0_IP or 10.0.0.1)
    --node1-ip IP      RDMA IP for rank 1 (default: \$RMLX_NODE1_IP or 10.0.0.2)
    --remote-dir DIR   Remote project directory (default: \$RMLX_REMOTE_DIR or \$HOME/rmlx)
    --skip-build       Skip cargo build step
    --help, -h         Show this help message

Environment variables (override defaults, overridden by CLI args):
    RMLX_NODE0, RMLX_NODE1, RMLX_NODE0_IP, RMLX_NODE1_IP, RMLX_REMOTE_DIR
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
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)      _NODE0="$2"; shift 2 ;;
        --node1)      _NODE1="$2"; shift 2 ;;
        --node0-ip)   _NODE0_IP="$2"; shift 2 ;;
        --node1-ip)   _NODE1_IP="$2"; shift 2 ;;
        --remote-dir) _REMOTE_DIR="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Apply priority: CLI > env > default
NODE1="${_NODE0:-${RMLX_NODE0:-node0}}"
NODE2="${_NODE1:-${RMLX_NODE1:-node1}}"
NODE1_IP="${_NODE0_IP:-${RMLX_NODE0_IP:-10.0.0.1}}"
NODE2_IP="${_NODE1_IP:-${RMLX_NODE1_IP:-10.0.0.2}}"
RMLX_ROOT="${_REMOTE_DIR:-${RMLX_REMOTE_DIR:-\$HOME/rmlx}}"

# ─── Constants ─────────────────────────────────────────────────────────
RDMA_DEVICE="rdma_en5"
PORT="${RMLX_TEST_PORT:-30100}"
BIN="target/release/examples/comprehensive_bench"
TIMEOUT=180

echo "=== RMLX Comprehensive RDMA Benchmark ==="
echo "  Nodes: $NODE1 ↔ $NODE2"
echo "  Device: $RDMA_DEVICE"
echo "  Port: $PORT"
echo ""

# Build (unless --skip-build)
if ! $SKIP_BUILD; then
    echo "[1/3] Building (release)..."
    ssh "$NODE1" "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1
    ssh "$NODE2" "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1
    echo ""
fi

# Clean
echo "[2/3] Cleanup..."
ssh "$NODE1" "env -C /tmp pkill -f comprehensive_bench 2>/dev/null" || true
ssh "$NODE2" "env -C /tmp pkill -f comprehensive_bench 2>/dev/null" || true
sleep 2
echo ""

# Run
echo "[3/3] Running benchmark..."
ENV="RMLX_RDMA_DEVICE=$RDMA_DEVICE RMLX_TEST_PORT=$PORT"

# Rank 1 (client) first
ssh "$NODE2" "env -C $RMLX_ROOT \
    $ENV RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP \
    timeout $TIMEOUT ./$BIN" > /tmp/rmlx_bench_rank1.json 2>/tmp/rmlx_bench_rank1.log &
pid2=$!

sleep 2

# Rank 0 (server)
ssh "$NODE1" "env -C $RMLX_ROOT \
    $ENV RMLX_RANK=0 RMLX_PEER_HOST=$NODE2_IP \
    timeout $TIMEOUT ./$BIN" > /tmp/rmlx_bench_rank0.json 2>/tmp/rmlx_bench_rank0.log &
pid1=$!

wait $pid1 || true; r0=$?
wait $pid2 || true; r1=$?

echo ""
echo "Exit codes: rank0=$r0, rank1=$r1"

if [[ $r0 -eq 0 ]]; then
    echo ""
    echo "=== Results (rank 0) ==="
    cat /tmp/rmlx_bench_rank0.log
    echo ""
    echo "JSON: /tmp/rmlx_bench_rank0.json"
else
    echo "=== FAILED ==="
    cat /tmp/rmlx_bench_rank0.log
    cat /tmp/rmlx_bench_rank1.log
fi
