import pytest
import numpy as np

from gepa.sim import BulletSimEnv


@pytest.mark.skipif(True, reason="Requires pybullet and URDFs available in path")
def test_reset_and_state_dims():
    env = BulletSimEnv(urdf_path="kuka_iiwa/model.urdf", gui=False)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert env.num_joints > 0
    assert obs.shape[0] == env.num_joints * 2  # positions + velocities
    env.close()


@pytest.mark.skipif(True, reason="Requires pybullet and URDFs available in path")
def test_step_action_roundtrip():
    env = BulletSimEnv(urdf_path="kuka_iiwa/model.urdf", gui=False)
    env.reset()
    action = np.zeros(env.num_joints, dtype=np.float32)
    result = env.step(action)
    assert isinstance(result.observation, np.ndarray)
    assert isinstance(result.reward, float)
    env.close()
