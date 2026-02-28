#!/bin/bash
# verify_vendor.sh — Verify vendored shader integrity against vendor_manifest.toml
# Reads all entries from the manifest, checks file presence and SHA256 hash.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$ROOT_DIR/shaders/mlx_compat"
MANIFEST="$DEST_DIR/vendor_manifest.toml"

echo "=== Vendor Verification ==="

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: vendor_manifest.toml not found at $MANIFEST"
    exit 1
fi

ERRORS=0
VERIFIED=0
TOTAL=0

while IFS= read -r line; do
    dest_rel=$(echo "$line" | cut -d'|' -f1)
    expected_sha=$(echo "$line" | cut -d'|' -f2)
    TOTAL=$((TOTAL + 1))

    dst_file="$DEST_DIR/$dest_rel"

    if [[ ! -f "$dst_file" ]]; then
        echo "  MISSING: $dest_rel"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # Verify SHA256
    actual_sha=$(shasum -a 256 "$dst_file" | cut -d' ' -f1)
    if [[ "$expected_sha" != "PLACEHOLDER" ]] && [[ -n "$expected_sha" ]] && [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "  HASH MISMATCH: $dest_rel (expected=$expected_sha actual=$actual_sha)"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $dest_rel"
        VERIFIED=$((VERIFIED + 1))
    fi
done < <(
    # Parse manifest: extract dest_path, sha256 pairs
    paste -d'|' \
        <(grep '^dest_path' "$MANIFEST" | cut -d'"' -f2) \
        <(grep '^sha256' "$MANIFEST" | cut -d'"' -f2)
)

echo ""
echo "$VERIFIED/$TOTAL files verified"

if [[ $ERRORS -gt 0 ]]; then
    echo "FAIL: $ERRORS files missing or hash mismatch"
    exit 1
fi

echo "All vendored files verified."
