#!/bin/bash
# vendor_mlx_shaders.sh — Idempotent MLX shader vendoring script
# Usage: ./scripts/vendor_mlx_shaders.sh [--dry-run] [--mlx-dir /path/to/mlx] [--force]
# Copies Metal shaders from a local mlx checkout, fixes include paths,
# and verifies SHA256 against vendor_manifest.toml.
#
# By default runs in strict mode: SHA mismatches are errors and cause exit 1.
# Use --force to downgrade SHA mismatches to warnings and continue.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$ROOT_DIR/shaders/mlx_compat"
MANIFEST="$DEST_DIR/vendor_manifest.toml"
MLX_DIR="${MLX_DIR:-$HOME/mlx}"
MLX_KERNELS="$MLX_DIR/mlx/backend/metal/kernels"

DRY_RUN=false
STRICT=true
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --mlx-dir=*) MLX_DIR="${arg#--mlx-dir=}"; MLX_KERNELS="$MLX_DIR/mlx/backend/metal/kernels" ;;
        --force)    STRICT=false ;;
        --strict)   STRICT=true ;;
    esac
done

if $DRY_RUN; then
    echo "[DRY-RUN] No files will be modified"
fi

echo "=== MLX Shader Vendoring ==="
echo "Source:      $MLX_KERNELS"
echo "Destination: $DEST_DIR"
echo "Manifest:    $MANIFEST"
echo "Strict mode: $STRICT"

if [[ ! -d "$MLX_KERNELS" ]]; then
    echo "ERROR: MLX kernel directory not found at $MLX_KERNELS"
    echo "Set MLX_DIR or pass --mlx-dir=/path/to/mlx"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: vendor_manifest.toml not found at $MANIFEST"
    exit 1
fi

# Get upstream commit SHA
UPSTREAM_SHA=$(cd "$MLX_DIR" && git rev-parse HEAD)
echo "Upstream commit: $UPSTREAM_SHA"
echo ""

fix_includes() {
    # Rewrite MLX-style includes to local relative includes
    sed 's|#include "mlx/backend/metal/kernels/|#include "|g' "$1"
}

# Parse manifest for dest_path entries
ERRORS=0
VENDORED=0

while IFS= read -r line; do
    # Extract source_path and dest_path from manifest
    source_rel=$(echo "$line" | cut -d'|' -f1)
    dest_rel=$(echo "$line" | cut -d'|' -f2)
    expected_sha=$(echo "$line" | cut -d'|' -f3)

    src_file="$MLX_KERNELS/${source_rel#mlx/backend/metal/kernels/}"
    dst_file="$DEST_DIR/$dest_rel"

    if [[ ! -f "$src_file" ]]; then
        echo "  MISSING upstream: $src_file"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY-RUN] Would vendor: $dest_rel"
    else
        mkdir -p "$(dirname "$dst_file")"
        fix_includes "$src_file" > "$dst_file"

        # Verify SHA256
        actual_sha=$(shasum -a 256 "$dst_file" | cut -d' ' -f1)
        if [[ "$expected_sha" != "PLACEHOLDER" ]] && [[ -n "$expected_sha" ]] && [[ "$actual_sha" != "$expected_sha" ]]; then
            echo "  ERROR: SHA mismatch for $dest_rel (expected=$expected_sha actual=$actual_sha)"
            ERRORS=$((ERRORS + 1))
        else
            echo "  OK: $dest_rel ($actual_sha)"
        fi
    fi
    VENDORED=$((VENDORED + 1))
done < <(
    # Parse manifest: extract source_path, dest_path, sha256 triples
    paste -d'|' \
        <(grep '^source_path' "$MANIFEST" | cut -d'"' -f2) \
        <(grep '^dest_path' "$MANIFEST" | cut -d'"' -f2) \
        <(grep '^sha256' "$MANIFEST" | cut -d'"' -f2)
)

echo ""
echo "Vendored $VENDORED files ($ERRORS errors)"

if [[ $ERRORS -gt 0 ]]; then
    if $STRICT; then
        echo "ERROR: $ERRORS integrity errors detected (strict mode). Use --force to override."
        exit 1
    else
        echo "WARN: $ERRORS integrity errors detected but --force was specified, continuing."
    fi
fi

echo "Done."
