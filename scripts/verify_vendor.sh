#!/bin/bash
# verify_vendor.sh — Verify vendored shader integrity
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$ROOT_DIR/shaders/mlx_compat"

echo "=== Vendor Verification ==="
ERRORS=0

# Check all expected files exist
for f in bf16.h bf16_math.h complex.h defines.h utils.h \
         copy.h binary.h binary_ops.h reduce_utils.h atomic.h softmax.h \
         reduction/ops.h reduction/reduce_all.h reduction/reduce_col.h reduction/reduce_row.h \
         rms_norm.metal rope.metal gemv.metal gemv_masked.metal \
         steel/utils.h steel/defines.h; do
    if [[ ! -f "$DEST_DIR/$f" ]]; then
        echo "MISSING: $f"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $f"
    fi
done

# Check manifest exists
if [[ ! -f "$DEST_DIR/vendor_manifest.toml" ]]; then
    echo "MISSING: vendor_manifest.toml"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: vendor_manifest.toml"
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "All vendored files present. ($((21)) files verified)"
else
    echo "FAIL: $ERRORS files missing"
    exit 1
fi
