#!/usr/bin/env bash
# Builds and launches the RMLX comprehensive RDMA benchmark on two nodes.
#
# Usage:
#   ./run_rmlx_bench.sh <NODE0_IP> <NODE1_HOST> [DEVICE_FILE]
#
# Example:
#   ./run_rmlx_bench.sh 192.168.1.10 hwStudio2 /path/to/devices.json
#
# Environment overrides:
#   NODE0_IP          - Coordinator IP (rank 0)
#   NODE1_HOST        - SSH host for rank 1
#   RMLX_IBV_DEVICES  - Path to device file (same on both nodes)
#   RMLX_COORDINATOR_PORT - Coordinator port (default: 18520)

set -euo pipefail

RMLX_ROOT="/Users/hw/rmlx"
BINARY="target/release/examples/comprehensive_bench"

NODE0_IP="${1:-${NODE0_IP:?'NODE0_IP required as arg1 or env var'}}"
NODE1_HOST="${2:-${NODE1_HOST:?'NODE1_HOST required as arg2 or env var'}}"
DEVICE_FILE="${3:-${RMLX_IBV_DEVICES:?'RMLX_IBV_DEVICES required as arg3 or env var'}}"
PORT="${RMLX_COORDINATOR_PORT:-18520}"

echo "=== RMLX Comprehensive RDMA Benchmark ==="
echo "  Node 0 (coordinator): ${NODE0_IP}"
echo "  Node 1 (SSH):         ${NODE1_HOST}"
echo "  Device file:          ${DEVICE_FILE}"
echo "  Coordinator port:     ${PORT}"
echo ""

# Build
echo "[build] Building comprehensive_bench (release)..."
cargo build --release -p rmlx-distributed --example comprehensive_bench \
    --manifest-path "${RMLX_ROOT}/Cargo.toml"
echo "[build] Done."

# Sync binary to remote node
echo "[sync] Syncing binary to ${NODE1_HOST}..."
rsync -az "${RMLX_ROOT}/${BINARY}" "${NODE1_HOST}:${RMLX_ROOT}/${BINARY}"
echo "[sync] Done."

echo ""
echo "[run] Launching rank 1 on ${NODE1_HOST}..."
ssh "${NODE1_HOST}" "env -C ${RMLX_ROOT} \
    RMLX_RANK=1 \
    RMLX_WORLD_SIZE=2 \
    RMLX_COORDINATOR=${NODE0_IP} \
    RMLX_COORDINATOR_PORT=${PORT} \
    RMLX_IBV_DEVICES=${DEVICE_FILE} \
    ./${BINARY}" \
    > /tmp/rmlx_bench_rank1.json 2>/tmp/rmlx_bench_rank1.log &
RANK1_PID=$!

echo "[run] Launching rank 0 locally..."
env -C "${RMLX_ROOT}" \
    RMLX_RANK=0 \
    RMLX_WORLD_SIZE=2 \
    RMLX_COORDINATOR="${NODE0_IP}" \
    RMLX_COORDINATOR_PORT="${PORT}" \
    RMLX_IBV_DEVICES="${DEVICE_FILE}" \
    "./${BINARY}" \
    > /tmp/rmlx_bench_rank0.json 2>/tmp/rmlx_bench_rank0.log &
RANK0_PID=$!

echo "[run] Waiting for both ranks to complete..."

FAIL=0
wait "${RANK0_PID}" || { echo "[ERROR] Rank 0 failed (exit=$?)"; FAIL=1; }
wait "${RANK1_PID}" || { echo "[ERROR] Rank 1 failed (exit=$?)"; FAIL=1; }

echo ""
if [ "${FAIL}" -eq 0 ]; then
    echo "=== Benchmark Complete ==="
else
    echo "=== Benchmark FAILED ==="
    echo "  Rank 0 log: /tmp/rmlx_bench_rank0.log"
    echo "  Rank 1 log: /tmp/rmlx_bench_rank1.log"
fi

echo "  Rank 0 results: /tmp/rmlx_bench_rank0.json"
echo "  Rank 1 results: /tmp/rmlx_bench_rank1.json"
echo "  Rank 0 log:     /tmp/rmlx_bench_rank0.log"
echo "  Rank 1 log:     /tmp/rmlx_bench_rank1.log"

exit "${FAIL}"
