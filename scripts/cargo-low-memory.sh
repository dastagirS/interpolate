#!/bin/sh
set -eu

MEMORY_AVAILABLE_KIB_MIN=$((5 * 1024 * 1024))
MEMORY_PHYSICAL_BYTES_MAX=$((6 * 1024 * 1024 * 1024))
MEMORY_SWAP_BYTES_MAX=$((2 * 1024 * 1024 * 1024))
MEMORY_AVAILABLE_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <cargo arguments...>" >&2
    exit 2
fi
if [ -z "$MEMORY_AVAILABLE_KIB" ] || [ "$MEMORY_AVAILABLE_KIB" -lt "$MEMORY_AVAILABLE_KIB_MIN" ]; then
    echo "refusing Cargo operation: at least 5 GiB of available memory is required" >&2
    exit 1
fi

export CARGO_BUILD_JOBS=1
export CARGO_INCREMENTAL=0
export CMAKE_BUILD_PARALLEL_LEVEL=1

printf 'available memory: %s MiB; physical-memory cap: 6144 MiB; swap cap: 2048 MiB; build jobs: 1\n' \
    "$((MEMORY_AVAILABLE_KIB / 1024))"

# A cgroup limits resident memory without restricting Vulkan's large virtual
# address reservations. A virtual-memory ulimit can prevent GPUI from creating
# its GPU context even when physical memory is plentiful.
if systemd-run --user --scope --quiet \
    -p "MemoryMax=$MEMORY_PHYSICAL_BYTES_MAX" \
    -p "MemorySwapMax=$MEMORY_SWAP_BYTES_MAX" \
    true >/dev/null 2>&1; then
    exec systemd-run --user --scope --quiet \
        -p "MemoryMax=$MEMORY_PHYSICAL_BYTES_MAX" \
        -p "MemorySwapMax=$MEMORY_SWAP_BYTES_MAX" \
        cargo "$@"
fi

echo "warning: user cgroups unavailable; using a 6 GiB virtual-memory fallback" >&2
ulimit -v $((6 * 1024 * 1024))
exec cargo "$@"
