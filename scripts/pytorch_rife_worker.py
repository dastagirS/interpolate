#!/usr/bin/env python3
"""Bounded binary worker for the optional PyTorch/VapourSynth RIFE backend."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy
import torch
import vapoursynth as vapoursynth
import vsrife

RGB_CHANNEL_COUNT = 3
MODEL_NAME = "4.25"
RIFE_MODULO = 64
TIMESTEP_HEADER = struct.Struct("<f")
CUDA_WORKER_READY = b"RIF1"
FRAME_SIZE_MAX = 128 * 1024 * 1024


def fail(message: str) -> None:
    assert message
    assert len(message) < 4096
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(1)


def read_exact(stream, size: int) -> bytes:
    assert size > 0
    assert size <= FRAME_SIZE_MAX
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            fail("CUDA RIFE worker input ended unexpectedly")
        data.extend(chunk)
    assert len(data) == size
    return bytes(data)


def load_network(model_path: Path, device: torch.device):
    assert model_path.is_file()
    assert device.type == "cuda"
    model_directory = str(model_path.parent)
    vsrife.model_dir = model_directory
    from vsrife.IFNet_HDv3_v4_25 import Head, IFNet

    network, encoder = vsrife.init_module(
        MODEL_NAME,
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
    assert network is not None
    return network, encoder


def make_grid(width: int, height: int, device: torch.device):
    assert width > 0
    assert height > 0
    horizontal = torch.linspace(-1.0, 1.0, width, device=device)
    horizontal = horizontal.view(1, 1, 1, width).expand(-1, -1, height, -1)
    vertical = torch.linspace(-1.0, 1.0, height, device=device)
    vertical = vertical.view(1, 1, height, 1).expand(-1, -1, -1, width)
    return torch.cat([horizontal, vertical], 1)


def frame_tensor(frame: bytes, width: int, height: int, device: torch.device):
    assert len(frame) == width * height * RGB_CHANNEL_COUNT
    pixels = numpy.frombuffer(frame, dtype=numpy.uint8)
    pixels = pixels.reshape(height, width, RGB_CHANNEL_COUNT)
    pixels = pixels.transpose(2, 0, 1)
    tensor = torch.from_numpy(pixels.copy()).to(device=device, dtype=torch.float16)
    return tensor.unsqueeze(0).div_(255.0)


def interpolate(
    network,
    encoder,
    frame_before: bytes,
    frame_after: bytes,
    width: int,
    height: int,
    timestep: float,
    device: torch.device,
) -> bytes:
    assert 0.0 < timestep < 1.0
    padded_width = math.ceil(width / RIFE_MODULO) * RIFE_MODULO
    padded_height = math.ceil(height / RIFE_MODULO) * RIFE_MODULO
    image_before = frame_tensor(frame_before, width, height, device)
    image_after = frame_tensor(frame_after, width, height, device)
    if padded_width != width or padded_height != height:
        padding = (0, padded_width - width, 0, padded_height - height)
        image_before = torch.nn.functional.pad(image_before, padding)
        image_after = torch.nn.functional.pad(image_after, padding)

    flow_divisor = torch.tensor(
        [(padded_width - 1.0) / 2.0, (padded_height - 1.0) / 2.0],
        dtype=torch.float,
        device=device,
    )
    grid = make_grid(padded_width, padded_height, device)
    timestep_tensor = torch.full(
        [1, 1, padded_height, padded_width],
        timestep,
        dtype=torch.float16,
        device=device,
    )
    if encoder is None:
        output = network(
            image_before,
            image_after,
            timestep_tensor,
            flow_divisor,
            grid,
        )
    else:
        encoded_before = encoder(image_before)
        encoded_after = encoder(image_after)
        output = network(
            image_before,
            image_after,
            timestep_tensor,
            flow_divisor,
            grid,
            encoded_before,
            encoded_after,
        )
    torch.cuda.synchronize(device)
    output = output[:, :, :height, :width].squeeze(0).float().clamp_(0.0, 1.0)
    pixels = output.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    result = pixels.tobytes()
    assert len(result) == width * height * RGB_CHANNEL_COUNT
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    arguments = parser.parse_args()
    assert 0 < arguments.width <= 16384
    assert 0 < arguments.height <= 16384
    assert arguments.gpu >= 0
    return arguments


def main() -> None:
    arguments = parse_arguments()
    frame_size = arguments.width * arguments.height * RGB_CHANNEL_COUNT
    if frame_size > FRAME_SIZE_MAX:
        fail("CUDA RIFE frame exceeds the safety limit")
    if not torch.cuda.is_available():
        fail("PyTorch reports that CUDA is unavailable")
    if vapoursynth.__api_version__ < 4:
        fail("VapourSynth API version 4 or newer is required")
    device = torch.device("cuda", arguments.gpu)
    network, encoder = load_network(arguments.model, device)
    sys.stdout.buffer.write(CUDA_WORKER_READY)
    sys.stdout.buffer.flush()
    while True:
        timestep_bytes = sys.stdin.buffer.read(TIMESTEP_HEADER.size)
        if not timestep_bytes:
            return
        if len(timestep_bytes) != TIMESTEP_HEADER.size:
            fail("CUDA RIFE worker received a truncated timestep")
        (timestep,) = TIMESTEP_HEADER.unpack(timestep_bytes)
        frame_before = read_exact(sys.stdin.buffer, frame_size)
        frame_after = read_exact(sys.stdin.buffer, frame_size)
        output = interpolate(
            network,
            encoder,
            frame_before,
            frame_after,
            arguments.width,
            arguments.height,
            timestep,
            device,
        )
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
