# User Guide

**Language**: [中文](USER_GUIDE.md) | English  
**Back**: [README](../README_EN.md)

This guide is for first-time DepthForge users and follows the flow from setup to export. If your environment is ready, jump to section 2.

## 1. Setup & Installation

### General
- Python 3.10+
- GPU is optional (Open3D CPU mode works)
- D435 on USB 3.0 is recommended

### Windows 10/11
1. Install Intel RealSense SDK (includes drivers and `realsense-viewer`)
2. Install Python dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**Common issues**
- Device not detected: confirm RealSense drivers and SDK are installed.
- Choppy preview: lower resolution/FPS, disable some depth filters.
- PyVistaQt fails to render: update GPU drivers, verify `vtk` and `pyvistaqt`.

### Ubuntu 20.04/22.04
1. Install librealsense and udev rules (official recommended packages):
   ```bash
   sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
   ```
2. Install pyrealsense2 (if system package is outdated, use pip with matching system libraries)
3. Install Python dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**Common issues**
- Permission denied: ensure udev rules are applied and replug the device.
- USB 3.0 power: prefer direct connection or a powered hub.
- Qt launch failure: install `libxcb` and related Qt runtime dependencies.

### macOS 13/14
1. On macOS, D435 typically requires **building librealsense from source** (Homebrew works if `RSUSB` is enabled)  
   - `pyrealsense2` must match your local `librealsense` version
2. Install Python dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**Note**: realtime capture may be limited on macOS; `.bag` playback is recommended.

**Common issues**
- Crash on start (segmentation fault): verify `rs-enumerate-devices`/`realsense-viewer` first (or run `rs_run.sh`). If it only crashes in realtime mode, try setting `device.color_format` to `rgb8`.

## 2. GUI Workflow (Recommended)
1. **Connect camera**: ensure D435 is on USB 3.0, launch the GUI.
2. **Select mode**: choose `realtime / semi / offline` in the right panel.
3. **Set bag**:
   - Offline: set `bag path` to replay `.bag`.
   - Recording: check "Record .bag" and set `bag path`.
4. **Start preview**: click "Start" to show RGB/Depth preview and logs.
5. **Scan**: move steadily; use "Pause" if you need to reposition; "Resume" to continue.
6. **Stop**: click "Stop" to finalize session and save metadata.
7. **Global optimize**: click "Global Optimize" for loop closure and pose graph optimization.
8. **Export**: choose mesh format (PLY/OBJ/STL) or point cloud format (PLY/PCD), then export.
9. **Session management**: "Save Session" or "Load Session" to review past results.

Tip: the 3D viewer supports **pointcloud/mesh switch**, **reset camera**, and **toggle axes**.

## 3. Mode Recommendations
- **Realtime**:
  - Low latency, quick preview.
  - Suggestion: disable "Preview Mesh" and increase `fusion.preview_voxel`.
- **Semi**:
  - Balanced performance/quality; recommended for most uses.
  - Suggestion: set `fusion_stride` to 2–4; preview point cloud.
- **Offline**:
  - Highest quality; ideal for final results.
  - Suggestion: use `.bag` playback and enable global optimization.

## 4. Handheld Scanning Posture & Path
- **Distance**: keep **0.2–1.0m**; too close fails depth; too far loses detail.
- **Speed**: move smoothly; avoid fast turns.
- **Overlap**: keep ≥ 60% overlap between frames.
- **Path**:
  - Small objects: circle the object + slight top/bottom coverage.
  - Medium objects/upper body: front → left → back → right to form a loop.
- **Texture**: add stickers or newspapers for low-texture surfaces.
- **Material**: avoid shiny/black surfaces; change angle or apply matte spray.

## 5. Quick Parameter Presets
Use these as starting points in the Parameter panel:

| Scene | depth_min_m / depth_max_m | fusion.voxel_length | fusion.sdf_trunc | keyframe.min_translation | keyframe.min_rotation_deg | loop_closure.temporal_gap | loop_closure.max_distance |
|---|---|---|---|---|---|---|---|
| Small objects | 0.2 / 0.8 | 0.003 | 0.015 | 0.02 | 4.0 | 20 | 0.3 |
| Medium objects | 0.3 / 1.0 | 0.006 | 0.03 | 0.04 | 6.0 | 30 | 0.5 |

Tips:
- Unstable tracking: reduce `keyframe.min_translation` and move slower.
- Noisy surfaces: increase `fusion.sdf_trunc` and enable spatial/temporal filters.
- Loop closure issues: increase texture and adjust `temporal_gap`/`max_candidates`.

## 6. Export Format Guide
- **PLY (mesh/point cloud)**: keeps vertex colors; best default choice.
- **OBJ (mesh)**: widely supported; color support depends on software.
- **STL (mesh)**: geometry only; best for 3D printing.
- **PCD (point cloud)**: common in PCL ecosystem.

Exports are stored under `sessions/<session_id>/exports/`.
