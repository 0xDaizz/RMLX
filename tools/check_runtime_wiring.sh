#!/usr/bin/env bash
# Checks that key runtime functions are called from non-test production code.
set -euo pipefail

FAIL=0
check() {
    local pattern="$1"
    local desc="$2"
    # Search non-test .rs files, exclude test modules and test files
    local count
    count=$(grep -rn "$pattern" crates/ --include="*.rs" \
        | grep -v "#\[test\]" \
        | grep -v "tests/" \
        | grep -v "test_" \
        | grep -v "// " \
        | grep -v "fn.*test" \
        | wc -l | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        echo "FAIL: $desc — no non-test call found for '$pattern'"
        FAIL=1
    else
        echo "OK:   $desc ($count call sites)"
    fi
}

check "dispatch_async" "Async dispatch"
check "combine_async_start\|combine_async_finish" "Async combine"
check "DescriptorProxy" "GPU descriptor proxy"
check "pre_post_recv_credits" "Recv credit pre-posting"
check "begin_transfer_async\|begin_compute_with_event" "Pipeline async transfer"

exit $FAIL
