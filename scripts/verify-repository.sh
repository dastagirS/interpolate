#!/bin/sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/.." && pwd)
MODEL_DIRECTORY="$REPOSITORY_ROOT/models/rife-v4.25"
PARAMETER_SHA256="6ba231fb00e4ae82b120f938d9b2df91db32fbf322bd110f29450efaf61848d6"
WEIGHTS_SHA256="10de487a095e61cb2971c39e3b5e17005a70fba6201c77fb96e063f4423b583f"
SCRIPT_COUNT_MAX=32

cd "$REPOSITORY_ROOT"

test -s "$MODEL_DIRECTORY/flownet.param"
test -s "$MODEL_DIRECTORY/flownet.bin"
printf '%s  %s\n%s  %s\n' \
    "$PARAMETER_SHA256" "$MODEL_DIRECTORY/flownet.param" \
    "$WEIGHTS_SHA256" "$MODEL_DIRECTORY/flownet.bin" \
    | sha256sum --check --strict

test -s Cargo.lock
test -s LICENSE
test -s legal/SOURCE-PROVENANCE.md
test -s .github/CODEOWNERS
grep -Fq '/.github/workflows/ @dastagirS' .github/CODEOWNERS
grep -Fq '/scripts/quality-gate.sh @dastagirS' .github/CODEOWNERS
grep -Fq "$PARAMETER_SHA256" legal/SOURCE-PROVENANCE.md
grep -Fq "$WEIGHTS_SHA256" legal/SOURCE-PROVENANCE.md

if git ls-files --error-unmatch plan.md >/dev/null 2>&1; then
    echo 'plan.md must remain local and untracked' >&2
    exit 1
fi

forbidden_paths=$(git ls-files | grep -E '(^|/)(target|dist)/|\.partial\.|(^|/)core(\.[0-9]+)?$|(^|/)sample\.(mp4|mkv|mov)$' || true)
if [ -n "$forbidden_paths" ]; then
    echo 'generated or local-only files are tracked:' >&2
    printf '%s\n' "$forbidden_paths" >&2
    exit 1
fi

git diff --check

script_count=0
for script_path in scripts/*.sh; do
    script_count=$((script_count + 1))
    if [ "$script_count" -gt "$SCRIPT_COUNT_MAX" ]; then
        echo "refusing to inspect more than $SCRIPT_COUNT_MAX shell scripts" >&2
        exit 1
    fi
    test -s "$script_path"
    sh -n "$script_path"
done
test "$script_count" -gt 0

printf 'repository integrity checks passed\n'
