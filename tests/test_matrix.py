import numpy as np

from scanner.utils.matrix import identity, invert, pose_delta


def test_identity_invert():
    pose = identity()
    inv = invert(pose)
    assert np.allclose(inv, np.eye(4))


def test_pose_delta():
    pose_a = identity()
    pose_b = identity()
    pose_b[0, 3] = 1.0
    delta = pose_delta(pose_a, pose_b)
    assert abs(delta.translation - 1.0) < 1e-6
