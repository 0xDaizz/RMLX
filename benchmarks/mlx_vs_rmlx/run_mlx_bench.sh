#!/usr/bin/env bash
# Launch MLX JACCL distributed benchmark on 2 nodes.
#
# Usage:
#   ./run_mlx_bench.sh                    # default: all categories
#   ./run_mlx_bench.sh --categories send_recv allreduce
#   ./run_mlx_bench.sh --warmup 5 --iters 20
#
# Prerequisites:
#   - MLX installed with JACCL backend on both nodes
#   - hosts.json in this directory (or set HOSTFILE env var)
#   - SSH key-based auth between nodes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/bench_mlx.py"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hosts.json}"
OUTPUT="${OUTPUT:-/tmp/mlx_bench_results.json}"

# ── Create hosts.json if not present ─────────────────────────────────────
if [ ! -f "$HOSTFILE" ]; then
    echo "No hosts.json found at ${HOSTFILE}."
    echo "Creating template — edit with your actual hostnames before running."
    cat > "$HOSTFILE" <<'HOSTS_EOF'
{
  "hosts": [
    {"hostname": "node1-hostname", "port": 22},
    {"hostname": "node2-hostname", "port": 22}
  ]
}
HOSTS_EOF
    echo "Template written to ${HOSTFILE}. Edit and re-run."
    exit 1
fi

echo "============================================"
echo " MLX JACCL Distributed Benchmark"
echo "============================================"
echo "  Script:   ${BENCH_SCRIPT}"
echo "  Hostfile: ${HOSTFILE}"
echo "  Output:   ${OUTPUT}"
echo "  Args:     $*"
echo "============================================"
echo ""

# ── Launch via mlx.launch ────────────────────────────────────────────────
mlx.launch \
    --backend jaccl \
    --hostfile "$HOSTFILE" \
    -- \
    python3 "$BENCH_SCRIPT" \
    --output "$OUTPUT" \
    "$@"

echo ""
echo "Done. Results at: ${OUTPUT}"

# ── Pretty-print summary if jq is available ──────────────────────────────
if command -v jq &>/dev/null && [ -f "$OUTPUT" ]; then
    echo ""
    echo "Framework: $(jq -r '.framework' "$OUTPUT")"
    echo "Backend:   $(jq -r '.backend' "$OUTPUT")"
    echo "Nodes:     $(jq -r '.node_count' "$OUTPUT")"
    echo "Timestamp: $(jq -r '.timestamp' "$OUTPUT")"
fi
