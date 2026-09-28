#!/usr/bin/env bash
set -u -o pipefail

EXPECTED_BINARY_SHA256='4b4bf265019393af4da1b8c473a48fd47ca4c919036cc490224a79aa02248eeb'
EXPECTED_MODEL_SHA256='6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2'
FAILURE_COUNT=0

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ -x "$SCRIPT_DIRECTORY/runtime/python/bin/python3" ]]; then
    PACKAGE_ROOT=$SCRIPT_DIRECTORY
elif [[ -x "$SCRIPT_DIRECTORY/../runtime/python/bin/python3" ]]; then
    PACKAGE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/.." && pwd)
else
    printf 'cannot locate bundled Python runtime relative to %s\n' "$SCRIPT_DIRECTORY" >&2
    exit 2
fi

PYTHON_EXECUTABLE="$PACKAGE_ROOT/runtime/python/bin/python3"
BINARY="$PACKAGE_ROOT/interpolate"
MODEL="$PACKAGE_ROOT/models/rife-v4.25/flownet_v4.25.pkl"

run_check() {
    local name=$1
    shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        printf '[PASS] %s\n' "$name"
    else
        printf '[FAIL] %s\n' "$name"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
}

printf 'package root: %s\n' "$PACKAGE_ROOT"
printf 'timestamp: %s\n' "$(date --iso-8601=seconds)"

run_check 'portable executable checksum' bash -c '
    actual=$(sha256sum "$1" | awk "{print \$1}")
    printf "actual: %s\n" "$actual"
    [[ "$actual" == "$2" ]]
' bash "$BINARY" "$EXPECTED_BINARY_SHA256"

run_check 'NVIDIA driver' nvidia-smi
run_check 'bundled Python version' "$PYTHON_EXECUTABLE" --version
run_check 'PyTorch CUDA and RIFE model load' env INTERPOLATE_PACKAGE_ROOT="$PACKAGE_ROOT" "$PYTHON_EXECUTABLE" -c '
from pathlib import Path
import hashlib
import os
import traceback

try:
    import torch
    import vapoursynth
    import vsrife

    print("torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports CUDA unavailable")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("PyTorch reports no CUDA devices")

    device = torch.device("cuda", 0)
    print("CUDA device:", torch.cuda.get_device_name(device))
    print("CUDA memory before model load:", torch.cuda.mem_get_info(device))

    package_root = Path(os.environ["INTERPOLATE_PACKAGE_ROOT"])
    model_path = (package_root / "models/rife-v4.25/flownet_v4.25.pkl").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print("model:", model_path)
    print("model bytes:", model_path.stat().st_size)
    print("model SHA-256:", model_hash)

    vsrife.model_dir = str(model_path.parent)
    from vsrife.IFNet_HDv3_v4_25 import Head, IFNet

    print("loading RIFE model...")
    network, encoder = vsrife.init_module(
        model_path.name,
        IFNet,
        1.0,
        False,
        device,
        torch.float16,
        Head,
    )
    network.eval()
    if encoder is not None:
        encoder.eval()
    print("CUDA memory after model load:", torch.cuda.mem_get_info(device))
    print("model loaded successfully")
except BaseException:
    traceback.print_exc()
    raise
'

run_check 'bundled model checksum' bash -c '
    actual=$(sha256sum "$1" | awk "{print \$1}")
    printf "actual: %s\n" "$actual"
    [[ "$actual" == "$2" ]]
' bash "$MODEL" "$EXPECTED_MODEL_SHA256"

printf '\nFailures: %d\n' "$FAILURE_COUNT"
if (( FAILURE_COUNT != 0 )); then
    printf 'Save all output and send it for diagnosis.\n' >&2
    exit 1
fi
printf 'All CUDA startup checks passed.\n'
