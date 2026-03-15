#!/usr/bin/env bash
set -euo pipefail

# 2-Node RDMA Integration Test Runner
# Usage: ./scripts/test_rdma_2node.sh [--skip-setup]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NODE1="hwStudio1"
NODE2="hwStudio2"
RDMA_DEVICE="rdma_en5"
TEST_PORT="${RMLX_TEST_PORT:-18515}"

echo "=== 2-Node RDMA Integration Tests ==="
echo ""

# Step 1: Network setup (unless --skip-setup)
if [[ "${1:-}" != "--skip-setup" ]]; then
    echo "[1/4] RDMA network setup..."
    python3 "$SCRIPT_DIR/rdma_setup.py" \
        --node "$NODE1:en5:10.254.0.5" \
        --node "$NODE2:en5:10.254.0.6" \
        --netmask 30
    echo ""
fi

# Step 2: Build test binary on both nodes
echo "[2/4] Building test binary..."
ssh "$NODE1" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" | tail -3
ssh "$NODE2" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" | tail -3
echo ""

# Step 3: Run single-node tests on node1
echo "[3/4] Single-node tests (hwStudio1)..."
ssh "$NODE1" "env -C /Users/hw/rmlx \
    RMLX_TEST_RDMA=1 \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    cargo test -p rmlx-distributed --test rdma_2node_integration -- \
    --include-ignored \
    --test-threads=1 \
    test_register_nocopy test_allreduce test_f16 2>&1" | tail -15
echo ""

# Step 4: Run 2-node test (both nodes simultaneously)
echo "[4/4] 2-node nocopy send test..."
echo "  Starting server on $NODE1 (port $TEST_PORT)..."

# Node1 acts as rank 0 (server)
ssh "$NODE1" "env -C /Users/hw/rmlx \
    RMLX_TEST_RDMA=1 \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    RMLX_TEST_2NODE=1 \
    RMLX_RANK=0 \
    RMLX_WORLD_SIZE=2 \
    RMLX_TEST_PORT=$TEST_PORT \
    RMLX_PEER_HOST=10.254.0.6 \
    timeout 30 cargo test -p rmlx-distributed --test rdma_2node_integration -- \
    --include-ignored test_nocopy_send_page_aligned --test-threads=1 2>&1" &
PID1=$!

sleep 1  # Let server start listening

echo "  Starting client on $NODE2 (port $TEST_PORT)..."
# Node2 acts as rank 1 (client)
ssh "$NODE2" "env -C /Users/hw/rmlx \
    RMLX_TEST_RDMA=1 \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    RMLX_TEST_2NODE=1 \
    RMLX_RANK=1 \
    RMLX_WORLD_SIZE=2 \
    RMLX_TEST_PORT=$TEST_PORT \
    RMLX_PEER_HOST=10.254.0.5 \
    timeout 30 cargo test -p rmlx-distributed --test rdma_2node_integration -- \
    --include-ignored test_nocopy_send_page_aligned --test-threads=1 2>&1" &
PID2=$!

# Wait for both
FAIL=0
wait $PID1 || { echo "  ✗ $NODE1 failed"; FAIL=1; }
wait $PID2 || { echo "  ✗ $NODE2 failed"; FAIL=1; }

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "=== All tests passed ==="
else
    echo "=== Some tests failed ==="
    exit 1
fi
