from pathlib import Path

import numpy as np

from scanner.io.session import SessionManager


def test_session_save_load(tmp_path: Path):
    config = {"version": "0.1"}
    manager = SessionManager(tmp_path)
    session = manager.create_session(config)
    session.keyframe_indices = [1, 2, 3]
    session.trajectory = [np.eye(4)]
    saved = manager.save_session(session)
    loaded = manager.load_session(saved)
    assert loaded.session_id == session.session_id
    assert loaded.keyframe_indices == [1, 2, 3]
    assert len(loaded.trajectory) == 1
