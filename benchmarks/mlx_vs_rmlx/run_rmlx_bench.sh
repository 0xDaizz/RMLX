#!/usr/bin/env bash
# RMLX comprehensive RDMA benchmark — 2-node launcher.
#
# Usage:
#   ./run_rmlx_bench.sh [--skip-build]
#
# Uses establish() pattern (same as test suite).
# Both nodes must have code synced and built.

set -euo pipefail

NODE1="hwStudio1"
NODE2="hwStudio2"
NODE1_IP="10.254.0.5"
NODE2_IP="10.254.0.6"
RDMA_DEVICE="rdma_en5"
PORT="${RMLX_TEST_PORT:-30100}"
RMLX_ROOT="/Users/hw/rmlx"
BIN="target/release/examples/comprehensive_bench"
TIMEOUT=180

echo "=== RMLX Comprehensive RDMA Benchmark ==="
echo "  Nodes: $NODE1 ↔ $NODE2"
echo "  Device: $RDMA_DEVICE"
echo "  Port: $PORT"
echo ""

# Build (unless --skip-build)
if [[ "${1:-}" != "--skip-build" ]]; then
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
