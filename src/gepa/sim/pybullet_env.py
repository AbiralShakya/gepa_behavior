from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception:  # pragma: no cover - allows tests to run without pybullet
    p = None
    pybullet_data = None


@dataclass
class EnvStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict


class BulletSimEnv:
    """A lightweight, research-ready PyBullet environment wrapper.

    Supports headless/GUI, multiple robots via URDF, and exposes a clean API
    for RL/behavior models.
    """

    def __init__(
        self,
        urdf_path: str,
        gui: bool = False,
        timestep: float = 1.0 / 240.0,
        gravity: float = -9.81,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_orientation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_steps_per_episode: int = 1000,
        enable_camera: bool = False,
        camera_width: int = 128,
        camera_height: int = 128,
        camera_fov: float = 60.0,
        camera_distance: float = 1.5,
        camera_yaw: float = 45.0,
        camera_pitch: float = -35.0,
    ) -> None:
        self.urdf_path = urdf_path
        self.gui = gui
        self.timestep = timestep
        self.gravity = gravity
        self.base_position = base_position
        self.base_orientation_euler = base_orientation_euler
        self.max_steps_per_episode = max_steps_per_episode

        self.enable_camera = enable_camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fov = camera_fov
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch

        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.num_joints: int = 0
        self.step_count: int = 0

    def reset(self) -> np.ndarray:
        if p is None:
            raise RuntimeError("PyBullet is not available. Please install pybullet.")

        if self.client_id is not None:
            p.disconnect(self.client_id)

        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.timestep)
        p.loadURDF("plane.urdf")

        base_orientation_quat = p.getQuaternionFromEuler(self.base_orientation_euler)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.base_position,
            base_orientation_quat,
            useFixedBase=True,
        )
        self.num_joints = p.getNumJoints(self.robot_id)
        self.step_count = 0

        for j in range(self.num_joints):
            p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0)

        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot_id, list(range(self.num_joints)))
        positions = np.array([s[0] for s in joint_states], dtype=np.float32)
        velocities = np.array([s[1] for s in joint_states], dtype=np.float32)
        obs = np.concatenate([positions, velocities], axis=0)
        return obs

    def render_rgb(self) -> Optional[np.ndarray]:
        if not self.enable_camera or p is None:
            return None
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.base_position,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        aspect = self.camera_width / float(self.camera_height)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=aspect,
            nearVal=0.01,
            farVal=10.0,
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        w, h, rgba, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer,
        )
        arr = np.array(rgba, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr.reshape((h, w, 4))
        rgb = arr[..., :3]
        return rgb

    def apply_action(self, action: np.ndarray) -> None:
        assert action.shape[0] == self.num_joints, (
            f"Action dim {action.shape[0]} != num_joints {self.num_joints}"
        )
        for j in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(action[j]),
                force=500,
            )

    def step(self, action: np.ndarray) -> EnvStepResult:
        self.apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        observation = self.get_observation()
        reward = 0.0
        done = self.step_count >= self.max_steps_per_episode
        info: Dict = {"step": self.step_count}
        if self.enable_camera:
            info["rgb"] = self.render_rgb()
        return EnvStepResult(observation, reward, done, info)

    def close(self) -> None:
        if self.client_id is not None and p is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def __enter__(self) -> "BulletSimEnv":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
