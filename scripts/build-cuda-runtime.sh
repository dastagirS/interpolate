#!/bin/sh
set -eu

RUNTIME_ROOT=${INTERPOLATE_CUDA_RUNTIME_ROOT:-runtime/python}
SYSTEM_PYTHON=${INTERPOLATE_SYSTEM_PYTHON:-python3}
PYTORCH_REQUIREMENT=${INTERPOLATE_PYTORCH_REQUIREMENT:-'torch>=2.10,<2.15'}
VSRIFE_REQUIREMENT=${INTERPOLATE_VSRIFE_REQUIREMENT:-'vsrife==5.7.0'}
VAPOURSYNTH_REQUIREMENT=${INTERPOLATE_VAPOURSYNTH_REQUIREMENT:-vapoursynth}

[ -n "$RUNTIME_ROOT" ]
[ -n "$SYSTEM_PYTHON" ]
[ -x "$(command -v "$SYSTEM_PYTHON")" ]
[ ! -e "$RUNTIME_ROOT" ] || rm -rf "$RUNTIME_ROOT"
mkdir -p "$(dirname "$RUNTIME_ROOT")"
"$SYSTEM_PYTHON" -m venv --copies "$RUNTIME_ROOT"

PYTHON_VERSION=$("$SYSTEM_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_STDLIB=$("$SYSTEM_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')
PYTHON_LIBRARY=$("$SYSTEM_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PYTHON_LIBRARY_NAME=$("$SYSTEM_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("INSTSONAME") or "")')
[ -n "$PYTHON_VERSION" ]
[ -d "$PYTHON_STDLIB" ]
[ -d "$PYTHON_LIBRARY" ]
[ -n "$PYTHON_LIBRARY_NAME" ]

mkdir -p "$RUNTIME_ROOT/lib"
cp -a "$PYTHON_STDLIB" "$RUNTIME_ROOT/lib/python$PYTHON_VERSION"
cp -a "$PYTHON_LIBRARY/$PYTHON_LIBRARY_NAME" "$RUNTIME_ROOT/lib/$PYTHON_LIBRARY_NAME"
mv "$RUNTIME_ROOT/bin/python3" "$RUNTIME_ROOT/bin/python3.real"
cat > "$RUNTIME_ROOT/bin/python3" <<PYTHON_LAUNCHER
#!/bin/sh
set -eu
LAUNCHER_DIRECTORY=\$(CDPATH= cd -- "\$(dirname -- "\$0")/.." && pwd)
export PYTHONHOME="\$LAUNCHER_DIRECTORY"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="\$LAUNCHER_DIRECTORY/lib:\$LAUNCHER_DIRECTORY/lib/python$PYTHON_VERSION/site-packages/torch/lib:\$LAUNCHER_DIRECTORY/lib/python$PYTHON_VERSION/site-packages/vapoursynth\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$LAUNCHER_DIRECTORY/bin/python3.real" "\$@"
PYTHON_LAUNCHER
chmod 0755 "$RUNTIME_ROOT/bin/python3"

"$RUNTIME_ROOT/bin/python3" -m pip install --disable-pip-version-check --no-cache-dir \
    "$PYTORCH_REQUIREMENT" \
    "$VAPOURSYNTH_REQUIREMENT" \
    "$VSRIFE_REQUIREMENT"
"$RUNTIME_ROOT/bin/python3" -c 'import torch, vapoursynth, vsrife; assert isinstance(torch.cuda.is_available(), bool)'
test -x "$RUNTIME_ROOT/bin/python3"
test -d "$RUNTIME_ROOT/lib/python$PYTHON_VERSION"
printf 'bundled CUDA Python runtime prepared at %s\n' "$RUNTIME_ROOT"
