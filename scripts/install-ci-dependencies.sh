#!/bin/sh
set -eu

PACKAGE_COUNT_MAX=32
package_count=21

test "$package_count" -gt 0
test "$package_count" -le "$PACKAGE_COUNT_MAX"

sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libasound2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libvulkan-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxcb1-dev \
    libxcb-randr0-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxcb-xinput-dev \
    libxcb-xkb-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    mesa-vulkan-drivers \
    pkg-config \
    vulkan-tools

test -x /usr/bin/ffmpeg
test -x /usr/bin/vulkaninfo
