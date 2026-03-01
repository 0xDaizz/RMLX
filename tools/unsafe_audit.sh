#!/usr/bin/env bash
set -euo pipefail

PASS=0
FAIL=0

echo "============================================"
echo "  rmlx unsafe audit"
echo "============================================"
echo ""

# Helper: count unsafe blocks in non-test Rust source files
count_unsafe_non_test() {
    local crate_dir="$1"
    local count=0
    while IFS= read -r file; do
        # Skip test files
        [[ "$file" == */tests/* ]] && continue
        # Use awk to skip #[cfg(test)] mod blocks, then count unsafe {
        local n
        n=$(awk '
            /^#\[cfg\(test\)\]/ { skip=1; depth=0; next }
            skip && /{/ { depth++; next }
            skip && /}/ { depth--; if(depth<=0) skip=0; next }
            skip { next }
            /unsafe\s*\{/ || /unsafe\s+fn / || /unsafe\s+impl / { count++ }
            END { print count+0 }
        ' "$file")
        count=$((count + n))
    done < <(find "$crate_dir/src" -name '*.rs' -type f 2>/dev/null)
    echo "$count"
}

# Check 1: rmlx-nn unsafe count
NN_ALLOWLIST=2
nn_count=$(count_unsafe_non_test "crates/rmlx-nn")
if [ "$nn_count" -le "$NN_ALLOWLIST" ]; then
    echo "PASS: rmlx-nn unsafe blocks: $nn_count (allowlist: $NN_ALLOWLIST)"
    PASS=$((PASS + 1))
else
    echo "FAIL: rmlx-nn unsafe blocks: $nn_count (allowlist: $NN_ALLOWLIST)"
    FAIL=$((FAIL + 1))
fi

# Check 2: rmlx-python unsafe count
PY_ALLOWLIST=0
py_count=$(count_unsafe_non_test "crates/rmlx-python")
if [ "$py_count" -le "$PY_ALLOWLIST" ]; then
    echo "PASS: rmlx-python unsafe blocks: $py_count (allowlist: $PY_ALLOWLIST)"
    PASS=$((PASS + 1))
else
    echo "FAIL: rmlx-python unsafe blocks: $py_count (allowlist: $PY_ALLOWLIST)"
    FAIL=$((FAIL + 1))
fi

# Check 3: no mem::forget in non-test code (excluding comments)
forget_count=$(grep -r "mem::forget" --include='*.rs' crates/ \
    | grep -v '_ko\.' \
    | grep -v '/tests/' \
    | grep -v '#\[cfg(test)\]' \
    | grep -v '^\s*//' \
    | grep -v '^\s*///' \
    | grep -v 'doc\b' \
    | grep -c 'mem::forget' || true)
if [ "$forget_count" -eq 0 ]; then
    echo "PASS: No mem::forget in non-test code"
    PASS=$((PASS + 1))
else
    echo "FAIL: Found $forget_count mem::forget in non-test code"
    grep -rn "mem::forget" --include='*.rs' crates/ | grep -v '/tests/' | grep -v '^\s*//'
    FAIL=$((FAIL + 1))
fi

# Check 4: no lock().unwrap() in progress.rs non-test code
if [ -f "crates/rmlx-rdma/src/progress.rs" ]; then
    lock_unwrap=$(awk '
        /^#\[cfg\(test\)\]/ { skip=1; depth=0; next }
        skip && /{/ { depth++; next }
        skip && /}/ { depth--; if(depth<=0) skip=0; next }
        skip { next }
        /lock\(\)\.unwrap\(\)/ { count++ }
        END { print count+0 }
    ' crates/rmlx-rdma/src/progress.rs)
    if [ "$lock_unwrap" -eq 0 ]; then
        echo "PASS: No lock().unwrap() in progress.rs non-test code"
        PASS=$((PASS + 1))
    else
        echo "FAIL: Found $lock_unwrap lock().unwrap() in progress.rs non-test code"
        FAIL=$((FAIL + 1))
    fi
else
    echo "PASS: progress.rs not found (skip)"
    PASS=$((PASS + 1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS/$((PASS + FAIL)) passed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo "  STATUS: FAILED"
    exit 1
else
    echo "  STATUS: ALL CLEAR"
    exit 0
fi
