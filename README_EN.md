# DepthForge — Handheld 3D Scanning & Reconstruction for RealSense D435

**Language**: [中文](README.md) | English

DepthForge is a practical handheld RGB-D scanning and reconstruction toolkit for Intel RealSense D435. It turns “capture → tracking → fusion → optimization → export” into a clear, repeatable workflow: real-time preview when you need speed, and offline reconstruction when you need quality.

## Highlights
- Three modes: realtime / semi / offline (supports `.bag` playback)
- Loop closure + pose-graph global optimization to reduce drift
- Decoupled TSDF fusion and preview for flexible speed/quality tradeoffs
- GUI and CLI entry points for demos or batch runs
- Export formats: PLY / OBJ / STL / PCD
- Session management and parameter panel for reproducible experiments

## Use Cases
- Small tabletop objects, mid-scale objects within 1m, upper-body scans
- Teaching, demos, rapid prototyping, and parameter studies

## Quick Start

### Requirements
- Python 3.10+
- Intel RealSense SDK (drivers + `realsense-viewer`)
- D435 on USB 3.0 is recommended

### Install
```bash
python -m pip install -U pip
python -m pip install -e .
```

### Run
```bash
# GUI
realsense-scanner-gui

# CLI
realsense-scanner-cli --mode realtime
```

> For OS-specific setup and common issues, see the docs below.

## Documentation
- User Guide (中文): `docs/USER_GUIDE.md`
- User Guide (English): `docs/USER_GUIDE_EN.md`
- Technical Notes (中文): `docs/TECHNICAL.md`
- Technical Notes (English): `docs/TECHNICAL_EN.md`

## Project Layout
```
DepthForge/
├─ configs/
│  └─ default.yaml
├─ docs/
├─ src/scanner/
│  ├─ app.py
│  ├─ cli.py
│  ├─ core/
│  ├─ devices/
│  ├─ fusion/
│  ├─ geometry/
│  ├─ io/
│  ├─ loop_closure/
│  ├─ tracking/
│  ├─ ui/
│  └─ utils/
├─ tests/
├─ cli.py
├─ pyproject.toml
└─ README.md
```

## Contributing
Issues and PRs are welcome. If you have practical tips on scanning posture, parameter tuning, or device compatibility, please share them.

## License
MIT License. See `LICENSE`.
