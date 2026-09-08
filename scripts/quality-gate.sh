#!/bin/sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/.." && pwd)
MEMORY_AVAILABLE_KIB_MIN=$((5 * 1024 * 1024))

run_cargo() {
    test "$#" -gt 0
    test -x "$REPOSITORY_ROOT/scripts/cargo-low-memory.sh"
    memory_available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ -z "$memory_available_kib" ] || [ "$memory_available_kib" -lt "$MEMORY_AVAILABLE_KIB_MIN" ]; then
        echo 'refusing quality-gate compilation: at least 5 GiB of available memory is required' >&2
        return 1
    fi
    printf 'quality-gate memory check: %s MiB available\n' "$((memory_available_kib / 1024))"
    if [ "${CI:-false}" = "true" ]; then
        CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CMAKE_BUILD_PARALLEL_LEVEL=1 \
            cargo "$@"
    else
        "$REPOSITORY_ROOT/scripts/cargo-low-memory.sh" "$@"
    fi
}

cd "$REPOSITORY_ROOT"

test -s Cargo.toml
test -s Cargo.lock
cargo fmt --all -- --check
"$REPOSITORY_ROOT/scripts/verify-repository.sh"
# Constant assertions are intentional executable documentation of safety limits.
run_cargo clippy --locked --all-targets -- -D warnings -A clippy::assertions-on-constants
run_cargo test --locked -- --test-threads=1

printf 'quality gate passed\n'
