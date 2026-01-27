# DepthForge — RealSense D435 手持 3D 扫描与重建

**语言**：中文 | [English](README_EN.md)

DepthForge 是一个面向 Intel RealSense D435 的手持 RGB-D 扫描与三维重建项目。它把“采集 → 跟踪 → 融合 → 优化 → 导出”串成一条清晰可复用的流程：既能实时预览，又能离线输出高质量模型。

## 亮点
- 三种工作模式：realtime / semi / offline（支持 `.bag` 回放）
- 回环检测 + 位姿图全局优化，显著降低漂移
- TSDF 融合与预览解耦，性能与质量可自由平衡
- GUI 与 CLI 双入口，适合演示和批处理
- 模型导出：PLY / OBJ / STL / PCD
- 会话管理与参数面板，便于重复实验与对比

## 适用场景
- 桌面级小物体、1m 以内中等物体、人体上半身
- 教学/展示、快速原型、扫描参数研究

## 快速开始

### 依赖
- Python 3.10+
- Intel RealSense SDK（含驱动与 `realsense-viewer`）
- D435 建议 USB 3.0 直连

### 安装
```bash
python -m pip install -U pip
python -m pip install -e .
```

### 运行
```bash
# GUI
realsense-scanner-gui

# CLI
realsense-scanner-cli --mode realtime
```

> 不同系统的安装细节与常见问题请参考文档。

## 文档
- 用户指南（中文）：`docs/USER_GUIDE.md`
- User Guide (English): `docs/USER_GUIDE_EN.md`
- 技术细节（中文）：`docs/TECHNICAL.md`
- Technical (English): `docs/TECHNICAL_EN.md`

## 目录结构
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

## 贡献
欢迎 issue / PR。如果你在扫描姿态、参数建议或设备兼容性方面有经验，也非常欢迎补充。

## License
MIT License. See `LICENSE`.
