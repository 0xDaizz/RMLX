#!/bin/bash
# vendor_mlx_shaders.sh — Idempotent MLX shader vendoring script
# Usage: ./scripts/vendor_mlx_shaders.sh [--dry-run]
# Downloads and verifies MLX Metal shaders from ml-explore/mlx

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$ROOT_DIR/shaders/mlx_compat"
MANIFEST="$DEST_DIR/vendor_manifest.toml"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No files will be modified"
fi

echo "=== MLX Shader Vendoring ==="
echo "Destination: $DEST_DIR"
echo "Manifest: $MANIFEST"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: vendor_manifest.toml not found at $MANIFEST"
    exit 1
fi

# In production: clone/download from upstream, verify hashes, copy files
# For now: verify existing stub files match manifest structure

echo ""
echo "Checking manifest entries..."

MISSING=0
while IFS= read -r dest; do
    dest=$(echo "$dest" | tr -d '"' | xargs)
    if [[ -z "$dest" ]]; then continue; fi
    if [[ ! -f "$DEST_DIR/$dest" ]]; then
        echo "  MISSING: $dest"
        MISSING=$((MISSING + 1))
    else
        echo "  OK: $dest"
    fi
done < <(grep '^dest_path' "$MANIFEST" | cut -d'=' -f2)

echo ""
if [[ $MISSING -gt 0 ]]; then
    echo "WARNING: $MISSING files listed in manifest are missing"
    exit 1
else
    echo "All manifest entries present."
fi

echo "Vendoring complete (stub mode — replace with actual upstream fetch)"
