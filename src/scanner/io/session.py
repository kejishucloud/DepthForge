from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scanner.core.types import SessionInfo
from scanner.utils.path import ensure_dir, make_session_dir


class SessionManager:
    def __init__(self, root: Path) -> None:
        self._root = ensure_dir(root)

    def create_session(self, config: Dict[str, Any]) -> SessionInfo:
        """
        创建新会话目录与基础元数据。
        :param config: 当前扫描配置
        :return: 新建的会话信息对象
        """
        session_id = time.strftime("%Y%m%d_%H%M%S")
        session_dir = make_session_dir(self._root, session_id)
        ensure_dir(session_dir / "exports")
        ensure_dir(session_dir / "logs")
        return SessionInfo(session_id=session_id, root=session_dir, config=config)

    def save_session(self, session: SessionInfo) -> Path:
        """
        保存会话 JSON 元数据。
        :param session: 会话信息对象
        :return: 保存后的 session.json 路径
        """
        session_path = session.root / "session.json"
        data = {
            "session_id": session.session_id,
            "version": session.version,
            "config": session.config,
            "keyframe_indices": session.keyframe_indices,
            "trajectory": [pose.tolist() for pose in session.trajectory],
            "bag_path": session.bag_path,
            "logs_path": session.logs_path,
            "exports": session.exports,
        }
        with session_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return session_path

    def load_session(self, session_path: Path) -> SessionInfo:
        """
        从 JSON 加载会话元数据。
        :param session_path: session.json 路径
        :return: 会话信息对象
        """
        with session_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        session = SessionInfo(
            session_id=data["session_id"],
            root=session_path.parent,
            config=data["config"],
            keyframe_indices=data.get("keyframe_indices", []),
            trajectory=[np.array(pose) for pose in data.get("trajectory", [])],
            bag_path=data.get("bag_path"),
            logs_path=data.get("logs_path"),
            exports=data.get("exports", []),
            version=data.get("version", "0.1"),
        )
        return session
