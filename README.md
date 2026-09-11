# Interpolate

A native Linux desktop application for GPU video frame interpolation. The interface is written in Rust with GPUI, while RIFE 4.25 inference runs in a focused C++ backend through ncnn and Vulkan.

> Interpolate is under active development. Keep the original source video until you have verified the generated output.

## Architecture

The executable enters through a small `main` module. The deep desktop application module owns GPUI state, rendering, background-mode policy, and coordinated shutdown behind one `run` interface. Media orchestration remains in the pipeline module, while cadence classification, output scheduling, persistent logging, the system-tray adapter, and the native backend each have focused modules. This keeps Content preset decisions local to cadence processing without exposing native or FFmpeg details to the interface.

## Current capabilities

- User-selectable RIFE 4.25 inference: Vulkan/ncnn or optional PyTorch CUDA with VapourSynth/vs-rife
- Configurable output rate from 1 to 480 FPS
- Streaming FFmpeg decode and encode without extracting frames to disk
- Movie and Anime content presets with deterministic defaults
- Conservative Anime cadence protection for two- and three-frame held drawings
- Scene-cut protection with a `0.15` difference threshold; always enabled for Anime
- Automatic half-scale UHD flow for 4K Anime sources with manual override
- Original audio and compatible subtitles copied into MKV output
- Bounded memory: three RGB24 frame buffers, one active job, one inference call
- Progress, output FPS, cancellation, atomic completion, and recoverable failed partial outputs
- Bounded FFmpeg diagnostics with five rotating 1 MiB logs and 200 recent in-memory lines
- Background processing enabled by default through the Linux system tray when available; configurable in Settings
- Native source picker, output picker, FPS input, inference-backend selector, and Vulkan device selector
- Runtime FFmpeg capability detection for CPU H.264 and NVIDIA NVENC H.264 output
- Configurable H.264 quality (CRF/CQ), speed preset, profile, encoder threads, and media preservation
- Optional NVIDIA CUDA/NVDEC decode with safe CPU H.264 fallback when NVENC cannot start
- Separate decode, RIFE inference, and encoder-pipe throughput diagnostics
- Input/output paths accepted through the UI or as the first two command-line arguments
- Anime diagnostics for detected held frames and smoothed cadence runs

The default output name is:

```text
movie__rife-4.25__120fps__sc.mkv
```

## Requirements

Runtime:

- Linux
- Vulkan-capable GPU and driver
- `ffmpeg` and `ffprobe` in `PATH`; NVIDIA NVENC/NVDEC options additionally require a compatible NVIDIA driver and FFmpeg build

Build:

- Rust toolchain
- CMake
- C and C++ compilers
- A system Vulkan loader (`libvulkan.so`)

Vulkan headers, ncnn, glslang, the native RIFE sources, and both RIFE 4.25 model formats are pinned in this source tree. The Vulkan backend requires no Python, PyTorch, VapourSynth, CUDA SDK, or system ncnn installation. Release archives include a self-contained Python, PyTorch, VapourSynth, and vs-rife runtime for the optional CUDA backend; only a compatible NVIDIA driver is required on CUDA-capable systems.

## Memory-safe build commands

Use the provided wrapper. It checks for at least 5 GiB of available memory, applies a 6 GiB physical-memory cgroup plus a 2 GiB swap cap, and forces Cargo and CMake to compile one job at a time. The cgroup intentionally leaves virtual address space unrestricted because Vulkan reserves large virtual ranges:

```sh
./scripts/cargo-low-memory.sh build
./scripts/cargo-low-memory.sh test -- --test-threads=1
./scripts/cargo-low-memory.sh run
```

The first native build is substantially longer than later incremental builds.

Before proposing or releasing a change, run the same release-blocking quality gate used by GitHub Actions:

```sh
./scripts/quality-gate.sh
```

It verifies formatting, model checksums, repository hygiene, Clippy warnings, the native ABI, cadence edge cases, cancellation, FFmpeg integration, and an end-to-end RIFE encode. Tests are serialized because the native backend intentionally permits only one active instance.

## Run

```sh
./scripts/cargo-low-memory.sh run
```

Optionally preselect paths:

```sh
./scripts/cargo-low-memory.sh run -- /path/to/input.mp4 /path/to/output.mkv
```

## MVP limitations

- SDR, constant-frame-rate sources are the intended input.
- The decoded interpolation format is RGB24 and output video is H.264 8-bit `yuv420p`.
- Encoding controls are intentionally limited to validated H.264 options; arbitrary FFmpeg arguments, 10-bit output, and alternate containers are not exposed.
- Output defaults to MKV because it preserves a broad set of copied audio and subtitle codecs.
- HDR, timestamp-aware VFR scheduling, 10-bit processing, and Windows packaging are not implemented yet.
- PyTorch CUDA inference uses the bundled Python/PyTorch/VapourSynth/vs-rife runtime and RIFE model; a compatible NVIDIA driver is still required, while Vulkan remains the compatibility fallback.
- NVENC/NVDEC are optional FFmpeg runtime accelerators and are independent of the selected RIFE inference backend.
- Cancellation takes effect between RIFE inference calls; an active GPU call is allowed to finish safely.

## Pinned native sources

- RIFE ncnn Vulkan fork: `TNTwise/rife-ncnn-vulkan` at `13338e38debe2e400b3eeecf6792312d01a692f9`
- ncnn: `ec19da2b615cc8be438ae3d31fd34fe23df03d52`
- glslang fork: `fe88f421038e1bb0a25cd5c1b2dfe505db82d08f`
- Vulkan Headers: `v1.4.341`
- Models: ncnn `rife-v4.25/flownet.param`/`flownet.bin` and PyTorch `flownet_v4.25.pkl` from their pinned RIFE sources

See `legal/` and the license files retained inside `native/vendor/` for attribution and redistribution terms. The RIFE `flownet.bin` model is intentionally versioned because the application cannot perform inference without it; unrelated ncnn example models are excluded.

## Releases

Pushing a tag beginning with `v` runs `.github/workflows/release.yml`. The workflow must pass the complete quality gate before it performs a bounded, serial release build and publishes a Linux x86-64 archive containing the application, native backend, bundled CUDA Python runtime, both RIFE models, README, and licenses. CI runs the gate in the release profile and shares its dependency and native-build cache with the release workflow, so packaging reuses the artifacts already compiled for tests. A cold cache still requires a complete build. A manual workflow run requires a `v`-prefixed release tag and creates the same GitHub release from the selected commit. If that release already exists, the workflow stops before installing dependencies or building.

The archive includes the CUDA Python runtime under `runtime/python/`; users do not need to install Python, PyTorch, VapourSynth, vs-rife, CUDA, cuDNN, or TensorRT separately. A compatible NVIDIA driver is required only when selecting CUDA inference.

After extracting the archive, run the GUI executable:

```sh
./interpolate
```

That file is the application binary with the native backend linked in. Keep `models/rife-v4.25/` and `runtime/python/` next to it. Double-clicking the executable opens the window without a terminal. Running it from an already-open terminal keeps that terminal attached, which is normal Linux behavior; ncnn may print GPU probe lines there. To add a menu entry, copy `share/applications/interpolate.desktop` into `~/.local/share/applications/` after placing `interpolate` on `PATH`.

The target system must still provide a Vulkan driver, `ffmpeg`, and `ffprobe`. Background mode is enabled by default when the desktop implements the freedesktop StatusNotifierItem system-tray protocol, and can be disabled in Settings. On GNOME, that commonly requires an AppIndicator extension.

## Diagnostics

Each job writes bounded application, ffprobe, decoder, and encoder diagnostics to `$XDG_STATE_HOME/interpolate/interpolate.log`, or `~/.local/state/interpolate/interpolate.log` when `XDG_STATE_HOME` is unset. Five 1 MiB files are retained. Cancellation removes partial output; processing failures preserve a partial MKV when one exists and report its location for possible recovery.

## License

Interpolate's original source code is available under the [MIT License](LICENSE). Bundled third-party code and model assets remain subject to their respective licenses and notices.
