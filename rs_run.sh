#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
sudo /usr/bin/env -i PATH="$PATH" rs-enumerate-devices
sudo /usr/bin/env -i PATH="$PATH" realsense-viewer
