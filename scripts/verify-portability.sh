#!/bin/sh
set -eu

PACKAGE_ROOT=${1:-}
PACKAGE_FILE_COUNT_MAX=100000
UNSUPPORTED_ISA_PATTERN='x86-64-v[234]'
MAX_GLIBC_VERSION='GLIBC_2.35'

[ -n "$PACKAGE_ROOT" ]
[ -d "$PACKAGE_ROOT" ]
command -v file >/dev/null 2>&1
a=$(command -v readelf)
test -n "$a"
APPLICATION="$PACKAGE_ROOT/interpolate"
RUNTIME_PYTHON="$PACKAGE_ROOT/runtime/python/bin/python3.real"
[ -x "$APPLICATION" ]
[ -x "$RUNTIME_PYTHON" ]

check_elf() {
    elf_path=$1
    [ -f "$elf_path" ]
    if ! file -b "$elf_path" | grep -q 'ELF'; then
        return 0
    fi
    isa_requirements=$(
        readelf -n "$elf_path" 2>/dev/null |
            awk '/x86 ISA needed:/ {print; exit}'
    )
    if printf '%s\n' "$isa_requirements" | grep -Eq "$UNSUPPORTED_ISA_PATTERN"; then
        echo "unsupported x86 ISA in $elf_path: $isa_requirements" >&2
        return 1
    fi
    glibc_versions=$(
        readelf --version-info "$elf_path" 2>/dev/null |
            grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' |
            sort -Vu || true
    )
    for glibc_version in $glibc_versions; do
        if [ "$(printf '%s\n' "$glibc_version" "$MAX_GLIBC_VERSION" | sort -V | tail -n 1)" != "$MAX_GLIBC_VERSION" ]; then
            echo "glibc requirement $glibc_version exceeds $MAX_GLIBC_VERSION in $elf_path" >&2
            return 1
        fi
    done
}

check_elf "$APPLICATION"
check_elf "$RUNTIME_PYTHON"

package_file_count=0
find "$PACKAGE_ROOT" -type f -print |
while IFS= read -r file_path; do
    package_file_count=$((package_file_count + 1))
    if [ "$package_file_count" -gt "$PACKAGE_FILE_COUNT_MAX" ]; then
        echo "refusing to inspect more than $PACKAGE_FILE_COUNT_MAX package files" >&2
        exit 1
    fi
    check_elf "$file_path"
done

if ldd "$APPLICATION" | grep -q 'not found'; then
    echo 'portable package contains unresolved application libraries' >&2
    exit 1
fi

RUNTIME_LIBRARY_PATH="$PACKAGE_ROOT/runtime/python/lib:$PACKAGE_ROOT/runtime/python/lib/python3.12/site-packages/torch/lib:$PACKAGE_ROOT/runtime/python/lib/python3.12/site-packages/vapoursynth"
if ! LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$RUNTIME_PYTHON" | grep -q 'not found'; then
    :
else
    echo 'portable package contains unresolved Python runtime libraries' >&2
    exit 1
fi

if ! LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$RUNTIME_PYTHON" -c 'import torch, vapoursynth, vsrife'; then
    echo 'portable package Python inference imports failed' >&2
    exit 1
fi

printf 'portable package checks passed: %s\n' "$PACKAGE_ROOT"
