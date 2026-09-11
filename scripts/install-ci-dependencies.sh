#!/bin/sh
set -eu

PACKAGE_COUNT_MAX=32
package_count=23

test "$package_count" -gt 0
test "$package_count" -le "$PACKAGE_COUNT_MAX"

# GitHub-hosted Ubuntu images include third-party repositories that are not
# needed by this project. Disable Chrome's repository because stale mirror
# metadata can make an otherwise valid apt update fail with a hash mismatch.
sudo rm -f \
    /etc/apt/sources.list.d/google-chrome.list \
    /etc/apt/sources.list.d/google-chrome.sources
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
    python3-pip \
    python3-venv \
    vulkan-tools

test -x /usr/bin/ffmpeg
test -x /usr/bin/vulkaninfo
