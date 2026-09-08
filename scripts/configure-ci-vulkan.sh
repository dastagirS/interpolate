#!/bin/sh
set -eu

ICD_FILE_COUNT_MAX=4
lavapipe_icd_file=
icd_file_count=0
for candidate in \
    /usr/share/vulkan/icd.d/lvp_icd*.json \
    /etc/vulkan/icd.d/lvp_icd*.json
do
    if [ ! -f "$candidate" ]; then
        continue
    fi
    icd_file_count=$((icd_file_count + 1))
    if [ "$icd_file_count" -gt "$ICD_FILE_COUNT_MAX" ]; then
        echo "more than $ICD_FILE_COUNT_MAX Lavapipe ICD files were found" >&2
        exit 1
    fi
    if [ -z "$lavapipe_icd_file" ]; then
        lavapipe_icd_file=$candidate
    fi
done

test "$icd_file_count" -ge 1
test -n "$lavapipe_icd_file"
test -r "$lavapipe_icd_file"
test -n "${GITHUB_ENV:-}"
echo "VK_ICD_FILENAMES=$lavapipe_icd_file" >> "$GITHUB_ENV"
VK_ICD_FILENAMES="$lavapipe_icd_file" vulkaninfo --summary

test -s "$GITHUB_ENV"
test -r "$lavapipe_icd_file"
