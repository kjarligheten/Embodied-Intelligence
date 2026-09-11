"""
Simplified student policy for testing without VLM dependency
Uses SAM3 for detection and simple keyword matching for instruction understanding
"""

from __future__ import annotations

import mujoco
import numpy as np

from graspbench.camera import camera_by_name
from graspbench.config import HOME_Q, TABLE_TOP_Z, OBJECT_SPEC_BY_NAME
from graspbench.ik import DampedLeastSquaresIK
from graspbench.perception import FoundationModelPerception, ModelServiceError
from graspbench.types import JointPositionCommand, Observation, PolicyDecision


class SimpleStudentPolicy:
    """Simplified policy using SAM3 detection + keyword matching (no VLM)."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        """Initialize policy state for a new episode."""
        self.task = task
        self.instruction = task["instruction"]

        # Initialize framework modules
        self.ik = DampedLeastSquaresIK(model)
        self.perception = FoundationModelPerception()

        # State variables
        self.target_id = None
        self.target_world = None
        self.destination_id = None
        self.destination_world = None

        # State machine
        self.stage = "IDLE"
        self.steps_in_stage = 0
        self.max_steps_per_stage = 50
        self.retry_count = 0
        self.max_retries = 3

        # Grasp parameters
        self.grasp_height_offset = 0.01
        self.pregrasp_height_offset = 0.10
        self.gripper_close_steps = 20
        self.lift_stabilization_steps = 8

    def _get_object_spec(self, object_id: str):
        """Get object specification with fallback."""
        if object_id in OBJECT_SPEC_BY_NAME:
            return OBJECT_SPEC_BY_NAME[object_id]
        return OBJECT_SPEC_BY_NAME.get("red_cube")

    def _select_target_by_keyword(self, instruction: str, detections: dict) -> str | None:
        """Select target using keyword matching with detections."""
        instruction_lower = instruction.lower()
        
        # Filter out containers - only consider graspable objects
        graspable_objects = {
            obj_id: obj for obj_id, obj in detections.items() 
            if "tray" not in obj_id.lower() and "plate" not in obj_id.lower()
        }
        
        if not graspable_objects:
            return None
        
        # Keyword to object ID mapping
        keyword_map = {
            'red': ['red_cube', 'apple'],
            'green': ['green_cylinder', 'mustard_bottle'],
            'blue': ['blue_box', 'potted_meat_can'],
            'yellow': ['banana'],
            'orange': ['orange'],
            'black': ['marker'],
            'metal': ['scissors'],
            'cube': ['red_cube'],
            'cylinder': ['green_cylinder'],
            'box': ['blue_box'],
            'brick': ['blue_box'],
            'banana': ['banana'],
            'apple': ['apple'],
            'orange': ['orange'],
            'bottle': ['mustard_bottle'],
            'can': ['potted_meat_can'],
            'scissors': ['scissors'],
            'marker': ['marker'],
        }
        
        # Find matching keywords in instruction
        matched_objects = []
        for keyword, obj_ids in keyword_map.items():
            if keyword in instruction_lower:
                for obj_id in obj_ids:
                    if obj_id in graspable_objects:
                        matched_objects.append(obj_id)
        
        # Return first match
        if matched_objects:
            return matched_objects[0]
        
        # No match, return first available graspable object
        return list(graspable_objects.keys())[0] if graspable_objects else None

    def _select_destination_by_keyword(self, instruction: str, detections: dict) -> str | None:
        """Select destination container using keyword matching."""
        instruction_lower = instruction.lower()
        
        containers = [obj_id for obj_id in detections.keys() if "tray" in obj_id.lower()]
        
        if not containers:
            return None
        
        if len(containers) == 1:
            return containers[0]
        
        # Sort by x position
        containers.sort(key=lambda x: detections[x].position[0])
        
        if "方盘" in instruction_lower or "square" in instruction_lower:
            return containers[-1]  # Rightmost
        elif "圆盘" in instruction_lower or "round" in instruction_lower:
            return containers[0]  # Leftmost
        
        return containers[-1]  # Default to rightmost

    def act(self, observation: Observation) -> PolicyDecision:
        """Return a policy decision based on current observation."""
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
        elif self.stage == "VERIFY":
            return self._handle_verify(observation)
        else:
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="error",
                rationale=f"Unknown stage: {self.stage}",
                target_id=self.target_id,
                done=True,
            )

    def _handle_idle(self, observation: Observation) -> PolicyDecision:
        """Detect objects and select target using SAM3 + keyword matching."""
        camera = camera_by_name(observation, "overhead")

        try:
            # Detect all objects in the scene using SAM3
            detections, sam_evidence = self.perception.detect_scene(camera)

            # Select target using keyword matching
            self.target_id = self._select_target_by_keyword(self.instruction, detections)
            
            if not self.target_id or self.target_id not in detections:
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="error",
                    rationale=f"Failed to select target from instruction: {self.instruction}",
                    target_id=None,
                    done=True,
                )

            self.target_world = detections[self.target_id].position.copy()
            
            # Set grasp parameters based on object
            spec = self._get_object_spec(self.target_id)
            self.grasp_height_offset = spec.half_height * 0.3
            self.pregrasp_height_offset = 0.10
            self.gripper_close_steps = 20
            self.lift_stabilization_steps = 8
            
            # For place tasks, also select destination
            if "place" in self.instruction.lower() or "put" in self.instruction.lower() or "放" in self.instruction:
                self.destination_id = self._select_destination_by_keyword(self.instruction, detections)
                if self.destination_id and self.destination_id in detections:
                    self.destination_world = detections[self.destination_id].position.copy()

            self.stage = "PREGRASP"
            self.steps_in_stage = 0
            self.retry_count = 0

            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="PREGRASP",
                rationale=f"Selected target: {self.target_id} using SAM3 + keyword matching. Starting grasp pipeline.",
                target_id=self.target_id,
                debug={
                    "target_world": self.target_world.tolist(),
                    "detections": list(detections.keys()),
                },
            )

        except ModelServiceError as exc:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="model_error",
                    rationale=f"Model service failed after {self.retry_count} retries: {exc}",
                    target_id=None,
                    done=True,
                )
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 1.0),
                stage="model_error",
                rationale=f"Model service failed, retrying ({self.retry_count}/{self.max_retries}): {exc}",
                target_id=None,
                request_retry=True,
                done=False,
            )

    def _handle_pregrasp(self, observation: Observation) -> PolicyDecision:
        """Move to safe position above target."""
        self.steps_in_stage += 1

        pregrasp_height = self.target_world[2] + self.pregrasp_height_offset
        target_pos = np.array([self.target_world[0], self.target_world[1], pregrasp_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=150,
        )

        if ik_result.converged and ik_result.position_error < 0.015:
            self.stage = "DESCEND"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 1.0),
                stage="DESCEND",
                rationale="Reached pregrasp position, descending to grasp.",
                target_id=self.target_id,
            )

        if self.steps_in_stage > self.max_steps_per_stage:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="timeout",
                    rationale="Pregrasp timeout, returning to home.",
                    target_id=self.target_id,
                    done=True,
                )
            self.stage = "IDLE"
            self.target_world = None
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="IDLE",
                rationale="Pregrasp timeout, retrying from IDLE.",
                target_id=self.target_id,
                request_retry=True,
                done=False,
            )

        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="PREGRASP",
            rationale="Moving to pregrasp position.",
            target_id=self.target_id,
        )

    def _handle_descend(self, observation: Observation) -> PolicyDecision:
        """Descend to grasp height."""
        self.steps_in_stage += 1

        grasp_height = self.target_world[2] + self.grasp_height_offset
        target_pos = np.array([self.target_world[0], self.target_world[1], grasp_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=150,
        )

        if ik_result.converged and ik_result.position_error < 0.01:
            self.stage = "CLOSE"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 1.0),
                stage="CLOSE",
                rationale="Reached grasp height, closing gripper.",
                target_id=self.target_id,
            )

        if self.steps_in_stage > self.max_steps_per_stage:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="timeout",
                    rationale="Descend timeout, returning to home.",
                    target_id=self.target_id,
                    done=True,
                )
            self.stage = "PREGRASP"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="PREGRASP",
                rationale="Descend timeout, retrying from PREGRASP.",
                target_id=self.target_id,
                request_retry=True,
                done=False,
            )

        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="DESCEND",
            rationale="Descending to grasp height.",
            target_id=self.target_id,
        )

    def _handle_close(self, observation: Observation) -> PolicyDecision:
        """Close gripper to grasp object."""
        self.steps_in_stage += 1

        gripper_action = max(0.0, 1.0 - self.steps_in_stage / self.gripper_close_steps)

        if self.steps_in_stage >= self.gripper_close_steps:
            self.stage = "LIFT"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, gripper_action),
                stage="LIFT",
                rationale="Gripper closed, lifting object.",
                target_id=self.target_id,
            )

        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, gripper_action),
            stage="CLOSE",
            rationale=f"Closing gripper ({self.steps_in_stage}/{self.gripper_close_steps}).",
            target_id=self.target_id,
        )

    def _handle_lift(self, observation: Observation) -> PolicyDecision:
        """Lift object to safe height."""
        self.steps_in_stage += 1

        lift_height = TABLE_TOP_Z + 0.15
        target_pos = np.array([self.target_world[0], self.target_world[1], lift_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=150,
        )

        if ik_result.converged and ik_result.position_error < 0.015:
            if self.destination_id:
                self.stage = "PLACE"
            else:
                self.stage = "VERIFY"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 1.0),
                stage=self.stage,
                rationale="Lifted object, moving to next stage.",
                target_id=self.target_id,
            )

        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "VERIFY"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="VERIFY",
                rationale="Lift timeout, proceeding to verify.",
                target_id=self.target_id,
            )

        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="LIFT",
            rationale="Lifting object.",
            target_id=self.target_id,
        )

    def _handle_place(self, observation: Observation) -> PolicyDecision:
        """Move to destination and release object."""
        self.steps_in_stage += 1

        if self.destination_world is None:
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="error",
                rationale="No destination set for place task.",
                target_id=self.target_id,
                done=True,
            )

        place_height = TABLE_TOP_Z + 0.10
        target_pos = np.array([self.destination_world[0], self.destination_world[1], place_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=150,
        )

        if ik_result.converged and ik_result.position_error < 0.015:
            self.stage = "VERIFY"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 0.0),
                stage="VERIFY",
                rationale="Reached destination, releasing object.",
                target_id=self.target_id,
            )

        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "VERIFY"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="VERIFY",
                rationale="Place timeout, proceeding to verify.",
                target_id=self.target_id,
            )

        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="PLACE",
            rationale="Moving to destination.",
            target_id=self.target_id,
        )

    def _handle_verify(self, observation: Observation) -> PolicyDecision:
        """Verify grasp success and return to home."""
        self.steps_in_stage += 1

        if self.steps_in_stage > self.max_steps_per_stage:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                return PolicyDecision(
                    command=JointPositionCommand(HOME_Q, 1.0),
                    stage="timeout",
                    rationale="Verify timeout, returning to home.",
                    target_id=self.target_id,
                    done=True,
                )
            self.stage = "IDLE"
            self.target_world = None
            self.destination_world = None
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="IDLE",
                rationale="Verify timeout, retrying from IDLE.",
                target_id=self.target_id,
                request_retry=True,
                done=False,
            )

        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 0.0),
            stage="VERIFY",
            rationale="Verifying grasp success.",
            target_id=self.target_id,
        )