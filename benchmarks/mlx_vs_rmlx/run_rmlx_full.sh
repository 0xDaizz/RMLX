#!/usr/bin/env bash
# Full RMLX benchmark: reboot → setup → bench (single shot, no retries)
set -uo pipefail

NODE1="hwStudio1"
NODE2="hwStudio2"
RMLX_ROOT="/Users/hw/rmlx"
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
ssh $NODE1 "env -C /tmp sudo ifconfig en5 10.254.0.5 netmask 255.255.255.252"
ssh $NODE2 "env -C /tmp sudo ifconfig en5 10.254.0.6 netmask 255.255.255.252"
sleep 1
# Verify TCP connectivity
ssh $NODE2 "env -C /tmp bash -c 'echo OK | timeout 3 nc 10.254.0.5 30599 &'" &
sleep 1
ssh $NODE1 "env -C /tmp bash -c 'timeout 3 nc -l 30599'" && echo "TCP OK" || echo "TCP FAIL"
wait

echo "=== Step 4: Build ==="
ssh $NODE1 "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1
ssh $NODE2 "env -C $RMLX_ROOT cargo build --release --example comprehensive_bench -p rmlx-distributed 2>&1" | tail -1

echo "=== Step 5: Run benchmark (single shot) ==="
ssh $NODE2 "env -C $RMLX_ROOT \
    RMLX_RANK=1 RMLX_PEER_HOST=10.254.0.5 RMLX_TEST_PORT=$PORT \
    RMLX_RDMA_DEVICE=$RDMA_DEVICE \
    timeout 180 ./$BIN" > /tmp/rmlx_bench_rank1.json 2>/tmp/rmlx_bench_rank1.log &
pid2=$!
sleep 2
ssh $NODE1 "env -C $RMLX_ROOT \
    RMLX_RANK=0 RMLX_PEER_HOST=10.254.0.6 RMLX_TEST_PORT=$PORT \
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
