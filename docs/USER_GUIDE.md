# 用户指南

**语言**：中文 | [English](USER_GUIDE_EN.md)  
**返回**：[README](../README.md)

这份指南面向第一次使用 DepthForge 的用户，按“连接设备 → 扫描 → 优化 → 导出”的顺序整理。若已完成安装，可直接从第 2 节开始。

## 1. 安装与环境准备

### 通用准备
- Python 3.10+
- GPU 非必须（Open3D CPU 模式即可）
- D435 建议连接 USB 3.0

### Windows 10/11
1. 安装 Intel RealSense SDK（官方安装包包含驱动与 `realsense-viewer`）
2. 安装 Python 依赖：
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**常见问题**
- 设备无法识别：确认已安装 RealSense 驱动与 SDK。
- 画面卡顿：降低分辨率或 FPS，关闭部分深度滤波。
- PyVistaQt 显示失败：升级显卡驱动，确认 `vtk` 与 `pyvistaqt` 安装成功。

### Ubuntu 20.04/22.04
1. 安装 librealsense 与 udev 规则（官方建议方式）：
   ```bash
   sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
   ```
2. 安装 pyrealsense2（如系统包未提供可用版本，建议使用 pip 与系统库匹配版本）
3. 安装 Python 依赖：
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**常见问题**
- 权限不足：确认 udev 规则已生效，重新插拔设备。
- USB 3.0 供电不足：使用直连或带供电 HUB。
- Qt 无法启动：确认已安装系统依赖 `libxcb` 等 Qt 运行库。

### macOS 13/14
1. D435 在 macOS 上需要 **自行编译 librealsense**（Homebrew 可用，但需开启 `RSUSB` 后端）  
   - `pyrealsense2` 需要与本地 `librealsense` 版本匹配
2. 安装 Python 依赖：
   ```bash
   python -m pip install -U pip
   python -m pip install -e .
   ```

**注意**：macOS 下实时采集可能受限，建议使用 `.bag` 离线回放模式。

## 2. GUI 操作流程（推荐顺序）
1. **连接相机**：确认 D435 连接 USB 3.0，打开 GUI。
2. **选择模式**：在右侧「模式」选择 `realtime / semi / offline`。
3. **设置 bag**：
   - 离线：填写 `bag 路径`，系统将回放 `.bag`。
   - 录制：勾选「录制 .bag」并填写 `bag 路径`。
4. **开始预览**：点击「开始」，左侧显示 RGB/Depth 预览，下方日志显示状态。
5. **扫描过程**：保持稳定移动；需要调整姿态时点「暂停」，继续时点「继续」。
6. **结束采集**：点击「停止」结束会话并保存轨迹/关键帧元数据。
7. **全局优化**：点击「全局优化」进行回环检测与位姿图优化（建议在停止后执行）。
8. **导出模型**：选择网格格式（PLY/OBJ/STL）或点云格式（PLY/PCD），点击「导出」。
9. **会话管理**：可「保存会话」或「加载会话」查看历史导出成果。

提示：右侧三维视图支持 **点云/网格切换**、**重置相机** 与 **显示坐标轴**。

## 3. 三种模式的使用建议
- **实时模式（realtime）**：
  - 优点：低延迟，边扫边看；适合快速预览。
  - 建议：关闭 `预览 Mesh`，适当增大 `fusion.preview_voxel`。
- **半实时模式（semi）**：
  - 优点：跟踪实时、融合抽帧，性能与质量平衡；推荐日常使用。
  - 建议：`fusion_stride` 设为 2–4；预览点云即可。
- **离线模式（offline）**：
  - 优点：最高质量；可多次重建调参；适合最终交付。
  - 建议：使用 `.bag` 回放，开启回环优化，并启用 `预览 Mesh` 检查细节。

## 4. 推荐手持扫描姿态与路径
- **距离**：保持 **0.2–1.0m**，过近会深度失效，过远细节不足。
- **速度**：匀速移动，避免快速抖动或突然转向。
- **重叠**：保证视野重叠 ≥ 60%，有利于跟踪与回环。
- **路径**：
  - 小物体：绕物体一圈 + 轻微抬高/俯视覆盖顶部与侧面。
  - 中等物体/上半身：正面 → 左侧 → 背面 → 右侧，形成闭环。
- **纹理**：对弱纹理表面可贴报纸/标记以提高特征稳定性。
- **材质**：反光或黑色材质易出错，建议改变角度或使用消光喷雾。

## 5. 参数快速指南（两套推荐）
下面为常用起点，可在「参数」面板微调：

| 场景 | depth_min_m / depth_max_m | fusion.voxel_length | fusion.sdf_trunc | keyframe.min_translation | keyframe.min_rotation_deg | loop_closure.temporal_gap | loop_closure.max_distance |
|---|---|---|---|---|---|---|---|
| 小物体（桌面级） | 0.2 / 0.8 | 0.003 | 0.015 | 0.02 | 4.0 | 20 | 0.3 |
| 中等物体（0.5–1m） | 0.3 / 1.0 | 0.006 | 0.03 | 0.04 | 6.0 | 30 | 0.5 |

补充建议：
- 若跟踪不稳：减小 `keyframe.min_translation`、降低移动速度。
- 若模型噪点多：增大 `fusion.sdf_trunc`，并开启 spatial/temporal 滤波。
- 若回环不稳定：提高纹理，或增大 `loop_closure.temporal_gap` 与 `max_candidates`。

## 6. 导出格式说明
- **PLY（网格/点云）**：保留顶点颜色，兼容性好，推荐默认格式。
- **OBJ（网格）**：通用模型格式，适合 DCC/游戏引擎，但颜色支持依赖软件。
- **STL（网格）**：仅几何，适合 3D 打印；无颜色。
- **PCD（点云）**：PCL 生态通用；适合点云算法处理。

导出文件默认存放在会话目录 `sessions/<session_id>/exports/`。
