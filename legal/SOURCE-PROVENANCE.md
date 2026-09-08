# Native source and model provenance

The native backend is derived from `TNTwise/rife-ncnn-vulkan` commit:

```text
13338e38debe2e400b3eeecf6792312d01a692f9
```

Only its RIFE implementation, custom warp operation, shaders, shader generator, MIT license, and standard RIFE 4.25 model are included. Its image-directory CLI, stb, and WebP dependencies are intentionally excluded because Interpolate passes bounded RGB24 buffers directly.

Pinned dependencies:

```text
ncnn          ec19da2b615cc8be438ae3d31fd34fe23df03d52
glslang       fe88f421038e1bb0a25cd5c1b2dfe505db82d08f
Vulkan-Headers v1.4.341
```

The vendored ncnn build excludes its examples, example model binaries, tools, tests, and benchmarks at configuration time. Example sources and weights are not distributed with Interpolate because they are unrelated to RIFE inference.

Model checksums:

```text
6ba231fb00e4ae82b120f938d9b2df91db32fbf322bd110f29450efaf61848d6  flownet.param
10de487a095e61cb2971c39e3b5e17005a70fba6201c77fb96e063f4423b583f  flownet.bin
```

Local compatibility change:

- FP16 storage remains enabled.
- Legacy Vulkan pack8 is disabled because the pinned modern ncnn shader interface no longer defines the old RIFE port's pack8 arithmetic helpers. Pack1/pack4 Vulkan paths are used instead.

The converted model still requires golden-frame comparison against the PyTorch `vsrife` RIFE 4.25 implementation before declaring pixel-level parity.
