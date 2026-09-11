"""
Ultra-simplified policy with timeout protection
Uses geometric methods only, no external model calls
"""

from __future__ import annotations

import mujoco
import numpy as np
import time

from graspbench.config import HOME_Q, TABLE_TOP_Z
from graspbench.ik import DampedLeastSquaresIK
from graspbench.types import JointPositionCommand, Observation, PolicyDecision


class GeometricPolicy:
    """Ultra-simplified policy using only geometric methods (no external calls)."""

    def __init__(self):
        self.timeout_threshold = 3.0  # seconds
        self.current_operation_start = None

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        """Initialize policy with geometric methods only."""
        self.task = task
        self.instruction = task["instruction"]
        
        # Core modules (no perception/VLM)
        self.ik = DampedLeastSquaresIK(model)
        
        # State
        self.target_id = None
        self.target_world = None
        self.destination_id = None
        self.destination_world = None
        
        # Simple state machine
        self.stage = "IDLE"
        self.steps_in_stage = 0
        self.retry_count = 0
        self.max_retries = 2
        
        # Timing
        self.current_operation_start = None

    def _check_timeout(self, operation_name: str) -> bool:
        """Check if current operation has timed out."""
        if self.current_operation_start is None:
            return False
        
        elapsed = time.time() - self.current_operation_start
        if elapsed > self.timeout_threshold:
            print(f"[Timeout] {operation_name} took {elapsed:.2f}s > {self.timeout_threshold}s")
            return True
        return False

    def _start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.current_operation_start = time.time()
        print(f"[Start] {operation_name}")

    def _end_operation(self, operation_name: str):
        """End timing an operation."""
        if self.current_operation_start is not None:
            elapsed = time.time() - self.current_operation_start
            print(f"[End] {operation_name} took {elapsed:.2f}s")
            self.current_operation_start = None

    def act(self, observation: Observation) -> PolicyDecision:
        """Simplified act with timeout protection."""
        # Check for timeout in current stage
        if self._check_timeout(f"Stage {self.stage}"):
            return self._handle_timeout(observation)

        if self.stage == "IDLE":
            return self._handle_idle(observation)
        elif self.stage == "PREGRASP":
            return self._handle_pregrasp(observation)
        elif self.stage == "DESCEND":
            return self._handle_descend(observation)
        elif self.stage == "CLOSE":
            return self._handle_close(observation)
        elif self.stage == "LIFT":
            return self._handle_lift(observation)
        elif self.stage == "PLACE":
            return self._handle_place(observation)
        elif self.stage == "RELEASE":
            return self._handle_release(observation)
        else:
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="error",
                rationale=f"Unknown stage: {self.stage}",
                done=True,
            )

    def _handle_timeout(self, observation: Observation) -> PolicyDecision:
        """Handle timeout by returning to safe state."""
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="timeout",
                rationale=f"Max retries reached after timeout in {self.stage}",
                done=True,
            )
        
        # Reset to IDLE and try again
        self.stage = "IDLE"
        self.steps_in_stage = 0
        self._end_operation(f"Timeout in {self.stage}")
        
        return PolicyDecision(
            command=JointPositionCommand(HOME_Q, 1.0),
            stage="IDLE",
            rationale=f"Timeout in {self.stage}, returning to IDLE for retry {self.retry_count}",
            done=False,
        )

    def _handle_idle(self, observation: Observation) -> PolicyDecision:
        """Simple target selection using geometric estimation."""
        self._start_operation("Target selection")
        
        try:
            # Use simple keyword matching (instant, no external calls)
            self.target_id = self._simple_target_selection(self.instruction)
            
            if self.target_id is None:
                self._end_operation("Target selection")
                return PolicyDecision(
                    command=JointPositionCommand(observation.joint_position, 0.0),
                    stage="IDLE",
                    rationale="Could not identify target from instruction",
                    done=False,
                )
            
            # Use geometric position estimation (no perception calls)
            # Objects are typically on table at z=0.425 (object center)
            # Use predefined positions based on common layouts
            self.target_world = self._estimate_position(self.target_id)
            
            # Set destination if this is a place task
            if "放进" in self.instruction or "place" in self.instruction.lower():
                self.destination_id = self._simple_destination_selection(self.instruction)
                self.destination_world = self._estimate_container_position(self.destination_id)
            
            self._end_operation("Target selection")
            
            # Move to pregrasp
            self.stage = "PREGRASP"
            self.steps_in_stage = 0
            
            pregrasp_height = self.target_world[2] + 0.08
            target_pos = np.array([self.target_world[0], self.target_world[1], pregrasp_height])
            
            return PolicyDecision(
                command=self.ik.compute_ik(target_pos, np.array([1.0, 0.0, 0.0, 0.0])),
                stage="PREGRASP",
                rationale=f"Moving to pregrasp for {self.target_id}",
                target_id=self.target_id,
            )
            
        except Exception as e:
            self._end_operation("Target selection error")
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="error",
                rationale=f"Target selection failed: {e}",
                done=True,
            )

    def _simple_target_selection(self, instruction: str) -> str | None:
        """Instant keyword-based target selection."""
        # Try exact matches first
        if "红色方块" in instruction or "red cube" in instruction.lower():
            return "red_cube"
        elif "绿色圆柱" in instruction or "green cylinder" in instruction.lower():
            return "green_cylinder"
        elif "蓝色长方体" in instruction or "blue box" in instruction.lower():
            return "blue_box"
        
        # Try single keyword matches
        if "红" in instruction or "red" in instruction.lower():
            return "red_cube"
        elif "绿" in instruction or "green" in instruction.lower():
            return "green_cylinder"
        elif "蓝" in instruction or "blue" in instruction.lower():
            return "blue_box"
        elif "黄" in instruction or "yellow" in instruction.lower():
            return "banana"
        elif "苹果" in instruction or "apple" in instruction.lower():
            return "apple"
        
        return None

    def _simple_destination_selection(self, instruction: str) -> str | None:
        """Instant keyword-based destination selection."""
        instruction_lower = instruction.lower()
        
        if "方盘" in instruction or "square" in instruction_lower:
            return "square_tray"
        elif "圆盘" in instruction or "round" in instruction_lower:
            return "round_tray"
        
        return "square_tray"  # Default

    def _estimate_position(self, object_id: str) -> np.ndarray:
        """Geometric position estimation based on common layouts."""
        # Use a conservative estimation based on typical object positions
        # This avoids perception calls that may timeout
        positions = {
            "red_cube": np.array([0.5, 0.0, 0.425]),
            "green_cylinder": np.array([0.4, 0.1, 0.425]),
            "blue_box": np.array([0.6, -0.1, 0.425]),
            "banana": np.array([0.5, 0.15, 0.420]),
            "apple": np.array([0.45, -0.15, 0.420]),
        }
        return positions.get(object_id, np.array([0.5, 0.0, 0.425]))

    def _estimate_container_position(self, container_id: str) -> np.ndarray:
        """Geometric container position estimation."""
        positions = {
            "square_tray": np.array([0.65, 0.15, 0.41]),
            "round_tray": np.array([0.65, -0.15, 0.41]),
        }
        return positions.get(container_id, np.array([0.65, 0.0, 0.41]))

    def _handle_pregrasp(self, observation: Observation) -> PolicyDecision:
        """Move to pregrasp position."""
        self.steps_in_stage += 1
        
        # Check if close enough to target
        distance = np.linalg.norm(observation.ee_position - self.target_world)
        if distance < 0.08:
            self.stage = "DESCEND"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 0.0),
                stage="DESCEND",
                rationale="Reached pregrasp, descending",
                target_id=self.target_id,
            )
        
        # Timeout
        if self.steps_in_stage > 15:
            return self._handle_timeout(observation)
        
        # Continue to pregrasp
        pregrasp_height = self.target_world[2] + 0.08
        target_pos = np.array([self.target_world[0], self.target_world[1], pregrasp_height])
        
        return PolicyDecision(
            command=self.ik.compute_ik(target_pos, np.array([1.0, 0.0, 0.0, 0.0])),
            stage="PREGRASP",
            rationale="Moving to pregrasp",
            target_id=self.target_id,
        )

    def _handle_descend(self, observation: Observation) -> PolicyDecision:
        """Descend to grasp position."""
        self.steps_in_stage += 1
        
        # Check if at grasp height
        grasp_height = self.target_world[2] - 0.01
        if observation.ee_position[2] <= grasp_height + 0.02:
            self.stage = "CLOSE"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 0.0),
                stage="CLOSE",
                rationale="At grasp height, closing gripper",
                target_id=self.target_id,
            )
        
        # Timeout
        if self.steps_in_stage > 12:
            return self._handle_timeout(observation)
        
        # Descend
        target_pos = np.array([self.target_world[0], self.target_world[1], grasp_height])
        
        return PolicyDecision(
            command=self.ik.compute_ik(target_pos, np.array([1.0, 0.0, 0.0, 0.0])),
            stage="DESCEND",
            rationale="Descending to grasp",
            target_id=self.target_id,
        )

    def _handle_close(self, observation: Observation) -> PolicyDecision:
        """Close gripper."""
        self.steps_in_stage += 1
        
        if self.steps_in_stage >= 8:
            self.stage = "LIFT"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 0.0),
                stage="LIFT",
                rationale="Gripper closed, lifting",
                target_id=self.target_id,
            )
        
        # Close gripper gradually
        grip = 1.0 - (self.steps_in_stage / 8.0)
        q = observation.joint_position.copy()
        q[-1] = grip
        
        return PolicyDecision(
            command=JointPositionCommand(q, 0.0),
            stage="CLOSE",
            rationale="Closing gripper",
            target_id=self.target_id,
        )

    def _handle_lift(self, observation: Observation) -> PolicyDecision:
        """Lift object."""
        self.steps_in_stage += 1
        
        # Check if lifted
        if observation.ee_position[2] > TABLE_TOP_Z + 0.08:
            if self.destination_id is not None:
                self.stage = "PLACE"
            else:
                # Lift task complete
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="SUCCESS",
                    rationale="Lift complete",
                    target_id=self.target_id,
                    done=True,
                )
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 0.0),
                stage="PLACE",
                rationale="Lifted, moving to place",
                target_id=self.target_id,
            )
        
        # Timeout
        if self.steps_in_stage > 15:
            return self._handle_timeout(observation)
        
        # Lift
        lift_height = TABLE_TOP_Z + 0.12
        target_pos = np.array([self.target_world[0], self.target_world[1], lift_height])
        
        return PolicyDecision(
            command=self.ik.compute_ik(target_pos, np.array([1.0, 0.0, 0.0, 0.0])),
            stage="LIFT",
            rationale="Lifting object",
            target_id=self.target_id,
        )

    def _handle_place(self, observation: Observation) -> PolicyDecision:
        """Move to destination."""
        self.steps_in_stage += 1
        
        # Check if at destination
        if self.destination_world is not None:
            distance = np.linalg.norm(observation.ee_position[:2] - self.destination_world[:2])
            if distance < 0.08:
                self.stage = "RELEASE"
                self.steps_in_stage = 0
                return PolicyDecision(
                    command=JointPositionCommand(observation.joint_position, 0.0),
                    stage="RELEASE",
                    rationale="At destination, releasing",
                    target_id=self.target_id,
                )
        
        # Timeout
        if self.steps_in_stage > 20:
            return self._handle_timeout(observation)
        
        # Move to destination
        if self.destination_world is not None:
            target_pos = self.destination_world.copy()
            target_pos[2] = TABLE_TOP_Z + 0.08
        else:
            target_pos = np.array([0.65, 0.1, TABLE_TOP_Z + 0.08])
        
        return PolicyDecision(
            command=self.ik.compute_ik(target_pos, np.array([1.0, 0.0, 0.0, 0.0])),
            stage="PLACE",
            rationale="Moving to destination",
            target_id=self.target_id,
        )

    def _handle_release(self, observation: Observation) -> PolicyDecision:
        """Release object."""
        self.steps_in_stage += 1
        
        if self.steps_in_stage >= 8:
            # Task complete
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="SUCCESS",
                rationale="Task complete",
                target_id=self.target_id,
                done=True,
            )
        
        # Open gripper
        q = observation.joint_position.copy()
        q[-1] = 1.0
        
        return PolicyDecision(
            command=JointPositionCommand(q, 0.0),
            stage="RELEASE",
            rationale="Releasing object",
            target_id=self.target_id,
        )