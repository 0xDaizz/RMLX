#!/usr/bin/env bash
# 2-Node RDMA Integration Test Runner
# Builds test binary once, then runs it directly — no cargo overhead during tests.
# Runs single-node RDMA tests first, then 2-node tests, then TP E2E 2-node tests.
set -euo pipefail

# ─── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

2-Node RDMA integration test runner. Builds test binary once, then runs
it directly without cargo overhead during tests.

Options:
    --node0 HOST       SSH hostname for rank 0 (default: \$RMLX_NODE0 or node0)
    --node1 HOST       SSH hostname for rank 1 (default: \$RMLX_NODE1 or node1)
    --node0-ip IP      RDMA IP for rank 0 (default: \$RMLX_NODE0_IP or 10.0.0.1)
    --node1-ip IP      RDMA IP for rank 1 (default: \$RMLX_NODE1_IP or 10.0.0.2)
    --remote-dir DIR   Remote project directory (default: \$RMLX_REMOTE_DIR or \$HOME/rmlx)
    --skip-setup       Skip rdma_setup.py
    --2node-only       Only 2-node tests (skip single-node)
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
SKIP_SETUP=false
ONLY_2NODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)      _NODE0="$2"; shift 2 ;;
        --node1)      _NODE1="$2"; shift 2 ;;
        --node0-ip)   _NODE0_IP="$2"; shift 2 ;;
        --node1-ip)   _NODE1_IP="$2"; shift 2 ;;
        --remote-dir) _REMOTE_DIR="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        --2node-only) ONLY_2NODE=true; SKIP_SETUP=true; shift ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Apply priority: CLI > env > default
NODE1="${_NODE0:-${RMLX_NODE0:-node0}}"
NODE2="${_NODE1:-${RMLX_NODE1:-node1}}"
NODE1_RDMA_IP="${_NODE0_IP:-${RMLX_NODE0_IP:-10.0.0.1}}"
NODE2_RDMA_IP="${_NODE1_IP:-${RMLX_NODE1_IP:-10.0.0.2}}"
# TCP exchange IPs — using RDMA direct IPs (TB5, verified working for TCP)
NODE1_IP="$NODE1_RDMA_IP"
NODE2_IP="$NODE2_RDMA_IP"
REMOTE_DIR="${_REMOTE_DIR:-${RMLX_REMOTE_DIR:-\$HOME/rmlx}}"

# ─── Constants ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RDMA_DEVICE="rdma_en5"
BASE_PORT="${RMLX_TEST_PORT:-18515}"
TIMEOUT=12  # establish(10s max) + test logic(~1s) + margin

GREEN='\033[32m'; RED='\033[31m'; BOLD='\033[1m'; NC='\033[0m'
PASS=0; FAIL=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }

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
BIN_PATH=$(ssh "$NODE1" "env -C $REMOTE_DIR cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" \
    | grep -o 'target/debug/deps/rdma_2node_integration-[a-f0-9]*' | head -1)

if [[ -z "$BIN_PATH" ]]; then
    # Fallback: find binary by glob
    BIN_PATH=$(ssh "$NODE1" "env -C $REMOTE_DIR find target/debug/deps -name 'rdma_2node_integration-*' -type f -perm +111 2>/dev/null | head -1")
fi

if [[ -z "$BIN_PATH" ]]; then
    echo -e "${RED}ERROR: Could not find test binary${NC}"
    exit 1
fi

# Verify binary exists on both nodes
ssh "$NODE1" "env -C $REMOTE_DIR test -x ./$BIN_PATH" || { echo -e "${RED}ERROR: Binary not found on $NODE1${NC}"; exit 1; }
ssh "$NODE2" "env -C $REMOTE_DIR test -x ./$BIN_PATH" || { echo -e "${RED}ERROR: Binary not found on $NODE2${NC}"; exit 1; }

echo "  Binary: $BIN_PATH"

# Build on node2
ssh "$NODE2" "env -C $REMOTE_DIR cargo test -p rmlx-distributed --test rdma_2node_integration --no-run 2>&1" | tail -2
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
    if ssh "$NODE1" "env -C $REMOTE_DIR \
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
    ssh "$NODE2" "env -C $REMOTE_DIR \
        $ENV_COMMON RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP \
        timeout $TIMEOUT ./$BIN_PATH --ignored --exact $test_name 2>&1" &
    pid2=$!

    sleep 1  # Let client start connect-retry loop

    # Start rank 0 (server) — accepts client connection immediately
    ssh "$NODE1" "env -C $REMOTE_DIR \
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
TP_BIN=$(ssh "$NODE1" "env -C $REMOTE_DIR cargo test -p rmlx-distributed --test tp_2node_e2e --no-run 2>&1" \
    | grep -o 'target/debug/deps/tp_2node_e2e-[a-f0-9]*' | head -1)

if [[ -z "$TP_BIN" ]]; then
    TP_BIN=$(ssh "$NODE1" "env -C $REMOTE_DIR find target/debug/deps -name 'tp_2node_e2e-*' -type f -perm +111 2>/dev/null | head -1")
fi

if [[ -z "$TP_BIN" ]]; then
    echo -e "  ${RED}ERROR: TP test binary not found${NC}"
else
    ssh "$NODE2" "env -C $REMOTE_DIR test -x ./$TP_BIN" || {
        ssh "$NODE2" "env -C $REMOTE_DIR cargo test -p rmlx-distributed --test tp_2node_e2e --no-run 2>&1" | tail -2
    }

    echo -e "  TP Binary: $TP_BIN"

    # Run TP test (single test, different port to avoid collision)
    TP_PORT=$((BASE_PORT + 10))
    echo -e "  ── ${BOLD}test_tp_2node_full_suite${NC} (port $TP_PORT)"

    ENV_COMMON="RMLX_TEST_RDMA=1 RMLX_RDMA_DEVICE=$RDMA_DEVICE RMLX_TEST_2NODE=1 RMLX_WORLD_SIZE=2 RMLX_TEST_PORT=$TP_PORT"

    # Client first (rank 1)
    ssh "$NODE2" "env -C $REMOTE_DIR \
        $ENV_COMMON RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP \
        timeout $TIMEOUT ./$TP_BIN --ignored --exact test_tp_2node_full_suite 2>&1" &
    pid2=$!
    sleep 1
    # Server (rank 0)
    ssh "$NODE1" "env -C $REMOTE_DIR \
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
