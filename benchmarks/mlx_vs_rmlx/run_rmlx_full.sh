#!/usr/bin/env bash
# Full RMLX benchmark: reboot → RDMA setup → build → bench (single shot, no retries).
# Reboots both nodes for a clean RDMA state, configures network, then runs
# the comprehensive benchmark binary.
set -uo pipefail

# ─── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Full RMLX benchmark: reboot both nodes, RDMA setup, build, and run
comprehensive benchmark (single shot, no retries).

Options:
    --node0 HOST       SSH hostname for rank 0 (default: \$RMLX_NODE0 or node0)
    --node1 HOST       SSH hostname for rank 1 (default: \$RMLX_NODE1 or node1)
    --node0-ip IP      RDMA IP for rank 0 (default: \$RMLX_NODE0_IP or 10.0.0.1)
    --node1-ip IP      RDMA IP for rank 1 (default: \$RMLX_NODE1_IP or 10.0.0.2)
    --remote-dir DIR   Remote project directory (default: \$RMLX_REMOTE_DIR or \$HOME/rmlx)
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node0)      _NODE0="$2"; shift 2 ;;
        --node1)      _NODE1="$2"; shift 2 ;;
        --node0-ip)   _NODE0_IP="$2"; shift 2 ;;
        --node1-ip)   _NODE1_IP="$2"; shift 2 ;;
        --remote-dir) _REMOTE_DIR="$2"; shift 2 ;;
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
BIN="target/release/examples/comprehensive_bench"
RDMA_DEVICE="rdma_en5"
PORT=30600

echo "=== Step 1: Reboot both nodes ==="
ssh $NODE2 "env -C /tmp sudo bash -c 'nvram auto-boot=true; sync; sync; shutdown -r +0'" 2>&1 || true
ssh $NODE1 "env -C /tmp sudo bash -c 'nvram auto-boot=true; sync; sync; shutdown -r +0'" 2>&1 || true
echo "Waiting 150s for reboot..."
sleep 150

echo "=== Step 2: Verify nodes up ==="
ssh -o ConnectTimeout=15 $NODE1 "env -C /tmp uptime" || { echo "NODE1 down!"; exit 1; }
ssh -o ConnectTimeout=15 $NODE2 "env -C /tmp uptime" || { echo "NODE2 down!"; exit 1; }

echo "=== Step 3: RDMA setup ==="
ssh $NODE1 "env -C /tmp sudo ifconfig en5 $NODE1_IP netmask 255.255.255.252"
ssh $NODE2 "env -C /tmp sudo ifconfig en5 $NODE2_IP netmask 255.255.255.252"
sleep 1
# Verify TCP connectivity
ssh $NODE2 "env -C /tmp bash -c 'echo OK | timeout 3 nc $NODE1_IP 30599 &'" &
sleep 1
ssh $NODE1 "env -C /tmp bash -c 'timeout 3 nc -l 30599'" && echo "TCP OK" || echo "TCP FAIL"
wait

echo "=== Step 4: Build ==="
ssh $NODE1 "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1
ssh $NODE2 "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1

echo "=== Step 5: Run benchmark (single shot) ==="
ssh $NODE2 "env -C $RMLX_ROOT \
    RMLX_RANK=1 RMLX_PEER_HOST=$NODE1_IP RMLX_TEST_PORT=$PORT \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    timeout 180 ./$BIN" > /tmp/rmlx_bench_rank1.json 2>/tmp/rmlx_bench_rank1.log &
pid2=$!
sleep 2
ssh $NODE1 "env -C $RMLX_ROOT \
    RMLX_RANK=0 RMLX_PEER_HOST=$NODE2_IP RMLX_TEST_PORT=$PORT \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    timeout 180 ./$BIN" > /tmp/rmlx_bench_rank0.json 2>/tmp/rmlx_bench_rank0.log &
pid1=$!

wait $pid1 || true; r0=$?
wait $pid2 || true; r1=$?

echo ""
echo "Exit: rank0=$r0, rank1=$r1"
echo ""
echo "=== Rank 0 Log ==="
cat /tmp/rmlx_bench_rank0.log
echo ""
echo "=== Done ==="
