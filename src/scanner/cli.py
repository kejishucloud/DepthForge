from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from scanner.core.keyframe import KeyframePolicy, KeyframeSelector
from scanner.core.types import Keyframe, TrackingStatus
from scanner.devices.realsense_device import RealSenseDevice
from scanner.fusion.tsdf import TSDFVolumeManager
from scanner.geometry.exporter import export_mesh, export_point_cloud
from scanner.geometry.postprocess import postprocess_mesh
from scanner.io.config import load_config
from scanner.io.logger import setup_logger
from scanner.io.session import SessionManager
from scanner.loop_closure.pose_graph import PoseGraphOptimizer
from scanner.tracking.odometry import RgbdOdometryTracker
from scanner.tracking.quality import QualityThresholds, evaluate_quality
from scanner.utils.matrix import identity, pose_distance
from scanner.utils.path import ensure_dir


def run_pipeline(config: Dict[str, Any], output_root: Path, optimize: bool) -> None:
    """
    在 CLI 下执行采集/重建/导出全流程。
    :param config: 参数介绍
    :param output_root: 参数介绍
    :param optimize: 参数介绍
    :return: 返回介绍
    """
    session_manager = SessionManager(output_root)
    session = session_manager.create_session(config)
    log_path = session.root / "logs" / "session.log"
    logger = setup_logger(log_path)
    session.logs_path = str(log_path)

    device = RealSenseDevice(config.get("device", {}))
    tracker = RgbdOdometryTracker(config.get("tracking", {}).get("odom", {}))
    quality_cfg = QualityThresholds(**config.get("tracking", {}).get("quality", {}))
    tsdf = TSDFVolumeManager(config.get("fusion", {}))
    key_cfg = config.get("keyframe", {})
    key_selector = KeyframeSelector(
        KeyframePolicy(
            min_translation=float(key_cfg.get("min_translation", 0.03)),
            min_rotation_deg=float(key_cfg.get("min_rotation_deg", 5.0)),
            min_quality=float(key_cfg.get("min_quality", 0.2)),
            min_interval=int(key_cfg.get("min_interval", 10)),
            max_keyframes=int(key_cfg.get("max_keyframes", 300)),
            mode_overrides=key_cfg.get("mode_overrides", {}),
        ),
        mode=config.get("scan", {}).get("mode", "offline"),
    )

    keyframes: list[Keyframe] = []
    trajectory: list[np.ndarray] = []
    current_pose = identity()
    prev_frame = None
    status = TrackingStatus.OK

    logger.info("开始采集")
    device.connect()

    try:
        while True:
            frame = device.read_frame()
            if frame is None:
                break

            if prev_frame is None:
                prev_frame = frame
                trajectory.append(current_pose.copy())
                if key_selector.should_add(frame.index, current_pose, 1.0):
                    keyframes.append(
                        Keyframe(
                            index=frame.index,
                            pose=current_pose.copy(),
                            color=frame.color_rgb.copy(),
                            depth=frame.depth_mm.copy(),
                            intrinsics=key_selector.intrinsics_to_dict(frame.intrinsics),
                            quality=1.0,
                        )
                    )
                    key_selector.update(frame.index, current_pose.copy(), keyframes)
                continue

            odom_result = tracker.estimate(prev_frame, frame, np.eye(4))
            fitness = odom_result.fitness
            rmse = odom_result.rmse
            pose_update = odom_result.pose

            icp_cfg = config.get("tracking", {}).get("icp_refine", {})
            icp_success = False
            if bool(icp_cfg.get("enable", True)):
                icp_result = tracker.refine_icp(
                    prev_frame,
                    frame,
                    pose_update,
                    float(icp_cfg.get("max_correspondence", 0.03)),
                )
                if icp_result.success:
                    icp_success = True
                    pose_update = icp_result.pose
                    fitness = icp_result.fitness
                    rmse = icp_result.rmse

            delta_translation, delta_rotation = pose_distance(identity(), pose_update)
            if not odom_result.success and not icp_success:
                status = TrackingStatus.LOST
            else:
                status = evaluate_quality(fitness, rmse, delta_translation, delta_rotation, quality_cfg)
            if status == TrackingStatus.LOST:
                logger.warning("跟踪丢失，跳过该帧融合")
                prev_frame = frame
                continue

            current_pose = current_pose @ pose_update
            trajectory.append(current_pose.copy())

            if key_selector.should_add(frame.index, current_pose, float(fitness)):
                keyframes.append(
                    Keyframe(
                        index=frame.index,
                        pose=current_pose.copy(),
                        color=frame.color_rgb.copy(),
                        depth=frame.depth_mm.copy(),
                        intrinsics=key_selector.intrinsics_to_dict(frame.intrinsics),
                        quality=float(fitness),
                    )
                )
                key_selector.update(frame.index, current_pose.copy(), keyframes)

            mode = config.get("scan", {}).get("mode", "realtime")
            fusion_stride = int(config.get("scan", {}).get("fusion_stride", 2))
            if mode in ("realtime", "offline") or (mode == "semi" and frame.index % fusion_stride == 0):
                tsdf.integrate_frame(frame, current_pose)
            prev_frame = frame

    except KeyboardInterrupt:
        logger.info("用户中断采集")
    finally:
        device.disconnect()

    session.keyframe_indices = [kf.index for kf in keyframes]
    session.trajectory = trajectory
    session_manager.save_session(session)

    if optimize and keyframes:
        optimizer = PoseGraphOptimizer(config.get("loop_closure", {}))
        result = optimizer.build(keyframes)
        optimizer.optimize(result.pose_graph)
        optimized_poses = [node.pose for node in result.pose_graph.nodes]
        tsdf.integrate_keyframes(keyframes, optimized_poses)

    mesh = tsdf.extract_mesh()
    mesh = postprocess_mesh(
        mesh,
        simplify_target=int(config.get("mesh", {}).get("simplify_target_triangles", 200000)),
        smooth_iterations=int(config.get("mesh", {}).get("smooth_iterations", 5)),
        remove_small=bool(config.get("mesh", {}).get("remove_small_components", True)),
        min_triangles=int(config.get("mesh", {}).get("min_triangles", 500)),
    )
    pcd = tsdf.extract_point_cloud()

    export_dir = ensure_dir(session.root / "exports")
    mesh_fmt = config.get("export", {}).get("default_mesh_format", "ply")
    cloud_fmt = config.get("export", {}).get("default_cloud_format", "ply")
    mesh_path = export_mesh(mesh, export_dir / "mesh", mesh_fmt)
    cloud_path = export_point_cloud(pcd, export_dir / "cloud", cloud_fmt)
    session.exports.append(
        {
            "type": "mesh",
            "path": str(mesh_path),
            "format": mesh_fmt,
            "params": {
                "simplify_target": config.get("mesh", {}).get("simplify_target_triangles"),
                "smooth_iterations": config.get("mesh", {}).get("smooth_iterations"),
                "remove_small": config.get("mesh", {}).get("remove_small_components"),
                "min_triangles": config.get("mesh", {}).get("min_triangles"),
                "keep_largest": config.get("mesh", {}).get("keep_largest"),
            },
        }
    )
    session.exports.append({"type": "cloud", "path": str(cloud_path), "format": cloud_fmt, "params": {}})
    session_manager.save_session(session)

    logger.info(f"导出完成: {mesh_path} | {cloud_path}")


def main() -> None:
    """
    CLI 入口，解析参数并启动流程。
    :return: 返回介绍
    """
    parser = argparse.ArgumentParser(description="RealSense D435 手持 3D 扫描 CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["realtime", "semi", "offline"], default="offline")
    parser.add_argument("--bag", type=str, default="", help="输入或输出的 bag 路径")
    parser.add_argument("--record", action="store_true", help="录制 bag")
    parser.add_argument("--optimize", action="store_true", help="启用回环优化")
    parser.add_argument("--output", type=str, default="sessions", help="会话输出目录")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config["scan"]["mode"] = args.mode

    if args.mode == "offline":
        config["device"]["playback_bag"] = True
        config["device"]["bag_path"] = args.bag
        config["device"]["playback_real_time"] = False
        if not args.bag:
            raise ValueError("离线模式需要提供 --bag 路径")
    else:
        config["device"]["playback_bag"] = False
        config["device"]["record_bag"] = args.record
        if args.record:
            config["device"]["bag_path"] = args.bag
            if not args.bag:
                raise ValueError("录制模式需要提供 --bag 路径")

    output_root = Path(args.output)
    run_pipeline(config, output_root, args.optimize)


if __name__ == "__main__":
    main()
