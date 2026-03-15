#!/usr/bin/env bash
set -euo pipefail

# 2-Node RDMA Integration Test Runner
# Builds test binary once, then runs it directly — no cargo overhead during tests.
#
# Usage:
#   ./scripts/test_rdma_2node.sh              # Full run (setup + build + test)
#   ./scripts/test_rdma_2node.sh --skip-setup # Skip rdma_setup.py
#   ./scripts/test_rdma_2node.sh --2node-only # Only 2-node tests (skip single-node)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NODE1="hwStudio1"
NODE2="hwStudio2"
# RDMA IPs (for rdma_setup.py only)
NODE1_RDMA_IP="10.254.0.5"
NODE2_RDMA_IP="10.254.0.6"
# TCP exchange IPs — using RDMA direct IPs (TB5, verified working for TCP)
NODE1_IP="10.254.0.5"
NODE2_IP="10.254.0.6"
RDMA_DEVICE="rdma_en5"
BASE_PORT="${RMLX_TEST_PORT:-18515}"
TIMEOUT=12  # establish(10s max) + test logic(~1s) + margin

GREEN='\033[32m'; RED='\033[31m'; BOLD='\033[1m'; NC='\033[0m'
PASS=0; FAIL=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }

# ── Parse args ──
SKIP_SETUP=false
ONLY_2NODE=false
for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --2node-only) ONLY_2NODE=true; SKIP_SETUP=true ;;
    esac
done

# ── Step 1: RDMA Setup ──
if ! $SKIP_SETUP; then
    echo -e "${BOLD}[1/5] RDMA Network Setup${NC}"
    python3 "$SCRIPT_DIR/rdma_setup.py" \
        --node "$NODE1:en5:$NODE1_RDMA_IP" \
        --node "$NODE2:en5:$NODE2_RDMA_IP" \
        --netmask 30
    echo ""
fi

# ── Step 2: Build test binary on both nodes ──
echo -e "${BOLD}[2/5] Building test binary${NC}"

# Build on node1 and capture binary path
BIN_PATH=$(ssh "$NODE1" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" \
    | grep -o 'target/debug/deps/rdma_2node_integration-[a-f0-9]*' | head -1)

if [[ -z "$BIN_PATH" ]]; then
    # Fallback: find binary by glob
    BIN_PATH=$(ssh "$NODE1" "env -C /Users/hw/rmlx find target/debug/deps -name 'rdma_2node_integration-*' -type f -perm +111 2>/dev/null | head -1")
fi

if [[ -z "$BIN_PATH" ]]; then
    echo -e "${RED}ERROR: Could not find test binary${NC}"
    exit 1
fi

# Verify binary exists on both nodes
ssh "$NODE1" "env -C /Users/hw/rmlx test -x ./$BIN_PATH" || { echo -e "${RED}ERROR: Binary not found on $NODE1${NC}"; exit 1; }
ssh "$NODE2" "env -C /Users/hw/rmlx test -x ./$BIN_PATH" || { echo -e "${RED}ERROR: Binary not found on $NODE2${NC}"; exit 1; }

echo "  Binary: $BIN_PATH"

# Build on node2
ssh "$NODE2" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" | tail -2
echo ""

# ── Step 2.5: RDMA cleanup ──
echo -e "${BOLD}[2.5/5] RDMA resource cleanup${NC}"
# Kill any lingering test processes on both nodes
for node in "$NODE1" "$NODE2"; do
    ssh "$node" "env -C /tmp bash -c 'pkill -f rdma_2node_integration 2>/dev/null; pkill -f tp_2node_e2e 2>/dev/null; sleep 1; echo cleaned'" 2>/dev/null || true
done
sleep 2  # Let kernel release RDMA resources
echo "  Done"
echo ""

# ── Step 3: Single-node tests ──
if ! $ONLY_2NODE; then
    echo -e "${BOLD}[3/5] Single-node tests (${NODE1})${NC}"
    if ssh "$NODE1" "env -C /Users/hw/rmlx \
        RMLX_TEST_RDMA=1 RMLX_RDMA_DEVICE=$RDMA_DEVICE \
        timeout $TIMEOUT cargo test -p rmlx-distributed --test rdma_2node_integration -- \
        --include-ignored --test-threads=1 \
        test_register_nocopy test_allreduce test_f16 2>&1" | tail -5; then
        pass "single-node (9 tests)"
    else
        fail "single-node"
    fi
    echo ""
fi

# ── Step 4: 2-node tests (binary direct execution) ──
echo -e "${BOLD}[4/5] 2-node tests${NC}"

# Single unified test: nocopy send + allreduce suite over one RDMA connection.
TESTS=(
    "test_2node_full_suite:0"
)

for entry in "${TESTS[@]}"; do
    test_name="${entry%%:*}"
    port_offset="${entry##*:}"
    port=$((BASE_PORT + port_offset * 2))

    echo -e "  ── ${BOLD}$test_name${NC} (port $port)"

    ENV_COMMON="RMLX_TEST_RDMA=1 RMLX_RDMA_DEVICE=$RDMA_DEVICE RMLX_TEST_2NODE=1 RMLX_WORLD_SIZE=2 RMLX_TEST_PORT=$port"

    # Start rank 1 (client) FIRST — it will retry connecting until server is ready
    ssh "$NODE2" "env -C /Users/hw/rmlx \
        $ENV_COMMON RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP \
        timeout $TIMEOUT ./$BIN_PATH --ignored --exact $test_name 2>&1" &
    pid2=$!

    sleep 1  # Let client start connect-retry loop

    # Start rank 0 (server) — accepts client connection immediately
    ssh "$NODE1" "env -C /Users/hw/rmlx \
        $ENV_COMMON RMLX_RANK=0 RMLX_PEER_HOST=$NODE2_IP \
        timeout $TIMEOUT ./$BIN_PATH --ignored --exact $test_name 2>&1" &
    pid1=$!

    ok=true
    wait $pid1 || { echo "    ${NODE1} rank=0 failed"; ok=false; }
    wait $pid2 || { echo "    ${NODE2} rank=1 failed"; ok=false; }

    if $ok; then pass "$test_name"; else fail "$test_name"; fi
done

echo ""

# ── Step 5: TP E2E 2-node tests ──
echo -e "${BOLD}[5/5] TP E2E 2-node tests${NC}"

# Find TP test binary
TP_BIN=$(ssh "$NODE1" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test tp_2node_e2e --no-run 2>&1" \
    | grep -o 'target/debug/deps/tp_2node_e2e-[a-f0-9]*' | head -1)

if [[ -z "$TP_BIN" ]]; then
    TP_BIN=$(ssh "$NODE1" "env -C /Users/hw/rmlx find target/debug/deps -name 'tp_2node_e2e-*' -type f -perm +111 2>/dev/null | head -1")
fi

if [[ -z "$TP_BIN" ]]; then
    echo -e "  ${RED}ERROR: TP test binary not found${NC}"
else
    ssh "$NODE2" "env -C /Users/hw/rmlx test -x ./$TP_BIN" || {
        ssh "$NODE2" "env -C /Users/hw/rmlx cargo test -p rmlx-distributed --test tp_2node_e2e --no-run 2>&1" | tail -2
    }

    echo -e "  TP Binary: $TP_BIN"

    # Run TP test (single test, different port to avoid collision)
    TP_PORT=$((BASE_PORT + 10))
    echo -e "  ── ${BOLD}test_tp_2node_full_suite${NC} (port $TP_PORT)"

    ENV_COMMON="RMLX_TEST_RDMA=1 RMLX_RDMA_DEVICE=$RDMA_DEVICE RMLX_TEST_2NODE=1 RMLX_WORLD_SIZE=2 RMLX_TEST_PORT=$TP_PORT"

    # Client first (rank 1)
    ssh "$NODE2" "env -C /Users/hw/rmlx \
        $ENV_COMMON RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP \
        timeout $TIMEOUT ./$TP_BIN --ignored --exact test_tp_2node_full_suite 2>&1" &
    pid2=$!
    sleep 1
    # Server (rank 0)
    ssh "$NODE1" "env -C /Users/hw/rmlx \
        $ENV_COMMON RMLX_RANK=0 RMLX_PEER_HOST=$NODE2_IP \
        timeout $TIMEOUT ./$TP_BIN --ignored --exact test_tp_2node_full_suite 2>&1" &
    pid1=$!

    ok=true
    wait $pid1 || { echo "    ${NODE1} rank=0 failed"; ok=false; }
    wait $pid2 || { echo "    ${NODE2} rank=1 failed"; ok=false; }

    if $ok; then pass "test_tp_2node_full_suite"; else fail "test_tp_2node_full_suite"; fi
fi

echo ""

# ── Summary ──
echo -e "${BOLD}Summary${NC}: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
[[ $FAIL -eq 0 ]] && echo -e "${GREEN}All tests passed!${NC}" || { echo -e "${RED}Some tests failed${NC}"; exit 1; }
