from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    """Represents a detected object in the scene with its properties."""
    name: str  # Object identifier (e.g., "red_cube", "banana")
    color: str  # Object color description
    shape: str  # Object shape description
    position: np.ndarray  # 3D position in world coordinates [x, y, z]
    quaternion: np.ndarray  # Orientation as quaternion [w, x, y, z]

    def as_dict(self) -> dict[str, Any]:
        """Convert the detected object to a dictionary for logging/serialization."""
        return {
            "name": self.name,
            "color": self.color,
            "shape": self.shape,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
        }


@dataclass(frozen=True)
class CameraObservation:
    """Represents RGB-D camera observation with calibration data."""
    name: str  # Camera identifier (e.g., "overhead", "front")
    rgb: np.ndarray  # RGB image array [height, width, 3]
    depth: np.ndarray  # Depth image array [height, width] in meters
    position: np.ndarray  # Camera position in world coordinates [x, y, z]
    rotation: np.ndarray  # Camera rotation matrix [3, 3]
    fovy_degrees: float  # Vertical field of view in degrees

    def metadata_dict(self) -> dict[str, Any]:
        """Extract camera metadata without the heavy image arrays."""
        return {
            "name": self.name,
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth.shape),
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "fovy_degrees": self.fovy_degrees,
        }


@dataclass(frozen=True)
class Observation:
    """Complete observation from the robot including vision and proprioception."""
    time: float  # Simulation time in seconds
    instruction: str  # Natural language instruction for the task
    joint_position: np.ndarray  # Current joint angles [7 DOF arm]
    joint_velocity: np.ndarray  # Current joint velocities [7 DOF arm]
    ee_position: np.ndarray  # End-effector position [x, y, z]
    ee_quaternion: np.ndarray  # End-effector orientation [w, x, y, z]
    gripper_opening: float  # Gripper opening state [0.0=closed, 1.0=open]
    cameras: tuple[CameraObservation, ...]  # Tuple of camera observations

    def compact_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary without heavy image arrays."""
        return {
            "time": self.time,
            "joint_position": self.joint_position.tolist(),
            "joint_velocity": self.joint_velocity.tolist(),
            "ee_position": self.ee_position.tolist(),
            "ee_quaternion": self.ee_quaternion.tolist(),
            "gripper_opening": self.gripper_opening,
            # Never serialize image arrays into JSONL logs. The metadata is enough
            # to audit which camera payload the policy received.
            "cameras": [camera.metadata_dict() for camera in self.cameras],
        }


@dataclass(frozen=True)
class JointPositionCommand:
    """Command to move arm joints to specific angles."""
    joint_position: np.ndarray  # Target joint angles [7 DOF arm]
    gripper_opening: float = 1.0  # Gripper opening [0.0=closed, 1.0=open]

    def __post_init__(self) -> None:
        """Validate joint position and gripper opening ranges."""
        q = np.asarray(self.joint_position, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"joint_position must have shape (7,), got {q.shape}")
        if not 0.0 <= float(self.gripper_opening) <= 1.0:
            raise ValueError("gripper_opening must be in [0, 1]")
        object.__setattr__(self, "joint_position", q)


@dataclass(frozen=True)
class CartesianDeltaCommand:
    """Optional VLA-friendly action; use CartesianDeltaAdapter to turn it into joint targets.
    
    This command type is suitable for VLA policies that output delta actions
    in Cartesian space rather than absolute joint positions.
    """
    translation: np.ndarray  # Delta translation [dx, dy, dz] in meters
    rotation_vector: np.ndarray  # Delta rotation as rotation vector [rx, ry, rz]
    gripper_opening: float  # Gripper opening [0.0=closed, 1.0=open]

    def __post_init__(self) -> None:
        """Validate delta action dimensions."""
        translation = np.asarray(self.translation, dtype=np.float64)
        rotation = np.asarray(self.rotation_vector, dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3,):
            raise ValueError("translation and rotation_vector must both have shape (3)")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation_vector", rotation)


@dataclass
class PolicyDecision:
    """Decision returned by the policy at each control step.
    
    This data class represents a decision made by the policy at each control step.
    """
    command: JointPositionCommand | CartesianDeltaCommand  # Action command to execute
    stage: str  # Current stage name (e.g., "PREGRASP", "GRASP", "VERIFY")
    rationale: str  # Human-readable explanation of the decision
    target_id: str | None = None  # ID of target object being manipulated
    done: bool = False  # Whether the episode should terminate
    request_retry: bool = False  # Whether to request a retry of current action
    debug: dict[str, Any] = field(default_factory=dict)  # Additional debug information

    def compact_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary for logging."""
        command = self.command
        if isinstance(command, JointPositionCommand):
            command_dict = {
                "mode": "joint_position",
                "joint_position": command.joint_position.tolist(),
                "gripper_opening": command.gripper_opening,
            }
        else:
            command_dict = {
                "mode": "cartesian_delta",
                "translation": command.translation.tolist(),
                "rotation_vector": command.rotation_vector.tolist(),
                "gripper_opening": command.gripper_opening,
            }
        return {
            "command": command_dict,
            "stage": self.stage,
            "rationale": self.rationale,
            "target_id": self.target_id,
            "done": self.done,
            "request_retry": self.request_retry,
            "debug": self.debug,
        }
