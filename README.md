# Interpolate

A native Linux desktop application for GPU video frame interpolation. The interface is written in Rust with GPUI, while RIFE 4.25 inference runs in a focused C++ backend through ncnn and Vulkan.

> Interpolate is under active development. Keep the original source video until you have verified the generated output.

## Current capabilities

- RIFE 4.25 interpolation on Vulkan GPUs
- Configurable output rate from 1 to 480 FPS
- Streaming FFmpeg decode and encode without extracting frames to disk
- Movie and Anime content presets with deterministic defaults
- Conservative Anime cadence protection for two- and three-frame held drawings
- Scene-cut protection with a `0.15` difference threshold; always enabled for Anime
- Automatic half-scale UHD flow for 4K Anime sources with manual override
- Original audio and compatible subtitles copied into MKV output
- Bounded memory: three RGB24 frame buffers, one active job, one inference call
- Progress, output FPS, cancellation, partial-file cleanup, and atomic completion
- Native source picker, output picker, FPS input, and Vulkan device selector
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
- `ffmpeg` and `ffprobe` in `PATH`

Build:

- Rust toolchain
- CMake
- C and C++ compilers
- A system Vulkan loader (`libvulkan.so`)

Vulkan headers, ncnn, glslang, the native RIFE source, and the RIFE 4.25 model are pinned in this source tree. No Python, PyTorch, VapourSynth, CUDA SDK, or system ncnn installation is required.

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
- Output defaults to MKV because it preserves a broad set of copied audio and subtitle codecs.
- HDR, timestamp-aware VFR scheduling, 10-bit processing, NVENC, and Windows packaging are not implemented yet.
- Cancellation takes effect between RIFE inference calls; an active GPU call is allowed to finish safely.

## Pinned native sources

- RIFE ncnn Vulkan fork: `TNTwise/rife-ncnn-vulkan` at `13338e38debe2e400b3eeecf6792312d01a692f9`
- ncnn: `ec19da2b615cc8be438ae3d31fd34fe23df03d52`
- glslang fork: `fe88f421038e1bb0a25cd5c1b2dfe505db82d08f`
- Vulkan Headers: `v1.4.341`
- Model: `rife-v4.25/flownet.param` and `flownet.bin` from the pinned RIFE fork

See `legal/` and the license files retained inside `native/vendor/` for attribution and redistribution terms. The RIFE `flownet.bin` model is intentionally versioned because the application cannot perform inference without it; unrelated ncnn example models are excluded.

## Releases

Pushing a tag beginning with `v` runs `.github/workflows/release.yml`. The workflow must pass the complete quality gate before it performs a bounded, serial release build and publishes a Linux x86-64 archive containing the application, native backend, RIFE model, README, and license. A manual workflow run requires a `v`-prefixed release tag and creates the same GitHub release from the selected commit. If that release already exists, the workflow stops before installing dependencies or building.

After extracting the archive, launch the application through its top-level wrapper so it can locate the packaged native library:

```sh
./interpolate
```

The target system must still provide a Vulkan driver, `ffmpeg`, and `ffprobe`.

## License

Interpolate's original source code is available under the [MIT License](LICENSE). Bundled third-party code and model assets remain subject to their respective licenses and notices.
