from __future__ import annotations

import mujoco
import numpy as np

from graspbench.camera import camera_by_name
from graspbench.config import HOME_Q, TABLE_TOP_Z
from graspbench.ik import DampedLeastSquaresIK
from graspbench.perception import FoundationModelPerception, ModelServiceError
from graspbench.types import JointPositionCommand, Observation, PolicyDecision
from graspbench.vlm import OpenAICompatibleVLM


class VLMAndIKTemplate:
    """VLM + SAM3 starter; ``act`` runs on the framework's single async worker.
    
    This template demonstrates how to integrate VLM for target grounding,
    SAM3 for visual perception, and numerical IK for motion control.
    """

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        """Initialize the VLM, SAM3, and IK modules for a new episode."""
        self.task = task
        self.ik = DampedLeastSquaresIK(model)
        self.perception = FoundationModelPerception()
        self.vlm = OpenAICompatibleVLM()
        self.target_id = None
        self.target_world = None
        self.model_evidence = {}
        
        # State machine variables
        self.stage = "IDLE"  # Current stage: IDLE, PREGRASP, DESCEND, CLOSE, LIFT, VERIFY
        self.steps_in_stage = 0  # Counter for steps in current stage
        self.max_steps_per_stage = 50  # Maximum steps before timeout

    def act(self, observation: Observation) -> PolicyDecision:
        """Return a policy decision based on current observation.
        
        The observation is a stable snapshot. Do not retain env.data or
        launch another per-step thread: the evaluator keeps physics live
        while this event-triggered remote request is pending.
        """
        # Stage: IDLE - Detect objects and select target using VLM
        if self.stage == "IDLE":
            camera = camera_by_name(observation, "overhead")
            try:
                # Detect all objects in the scene using SAM3
                detections, sam_evidence = self.perception.detect_scene(camera)
                
                # Use VLM to select the target based on instruction
                grounding = self.vlm.ground_target(
                    observation.instruction, camera, detections, sam_evidence
                )
            except ModelServiceError as exc:
                return PolicyDecision(
                    command=JointPositionCommand(observation.joint_position, 1.0),
                    stage="model_error",
                    rationale=f"Model service failed; hold safely: {exc}",
                    done=True,
                )
            self.target_id = grounding.target_id
            self.target_world = detections[self.target_id].position.copy()
            self.model_evidence = {
                "grounding": grounding.as_dict(),
                "sam3": sam_evidence[self.target_id].as_dict(),
            }
            self.stage = "PREGRASP"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="PREGRASP",
                rationale=f"VLM selected target: {self.target_id}. Starting grasp pipeline.",
                target_id=self.target_id,
                debug={
                    "target_world": self.target_world.tolist(),
                    "model_evidence": self.model_evidence,
                },
            )
        
        # Stage: PREGRASP - Move to safe position above target
        elif self.stage == "PREGRASP":
            return self._handle_pregrasp(observation)
        
        # Stage: DESCEND - Lower gripper to grasp height
        elif self.stage == "DESCEND":
            return self._handle_descend(observation)
        
        # Stage: CLOSE - Close gripper to grasp object
        elif self.stage == "CLOSE":
            return self._handle_close(observation)
        
        # Stage: LIFT - Lift object to safe height
        elif self.stage == "LIFT":
            return self._handle_lift(observation)
        
        # Stage: VERIFY - Verify task completion
        elif self.stage == "VERIFY":
            return self._handle_verify(observation)
        
        # Unknown stage - return to home
        else:
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="error",
                rationale=f"Unknown stage: {self.stage}",
                target_id=self.target_id,
                done=True,
            )
    
    def _handle_pregrasp(self, observation: Observation) -> PolicyDecision:
        """Handle PREGRASP stage: move to safe position above target."""
        self.steps_in_stage += 1
        
        # Calculate pregrasp position (15cm above target)
        pregrasp_height = self.target_world[2] + 0.15
        target_pos = np.array([self.target_world[0], self.target_world[1], pregrasp_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Solve IK to reach pregrasp position
        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=100,
        )
        
        # Check if reached pregrasp position
        if ik_result.converged and ik_result.position_error < 0.01:
            self.stage = "DESCEND"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 1.0),
                stage="DESCEND",
                rationale="Reached pregrasp position, descending to grasp.",
                target_id=self.target_id,
            )
        
        # Timeout handling
        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "IDLE"
            self.target_world = None  # Re-detect on next cycle
            self.steps_in_stage = 0
        
        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="PREGRASP",
            rationale="Moving to pregrasp position.",
            target_id=self.target_id,
        )
    
    def _handle_descend(self, observation: Observation) -> PolicyDecision:
        """Handle DESCEND stage: lower gripper to grasp height."""
        self.steps_in_stage += 1
        
        # Calculate grasp position (slightly above object center)
        grasp_height = self.target_world[2] + 0.02
        target_pos = np.array([self.target_world[0], self.target_world[1], grasp_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=100,
        )
        
        # Check if reached grasp height
        if ik_result.converged and ik_result.position_error < 0.005:
            self.stage = "CLOSE"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 1.0),
                stage="CLOSE",
                rationale="Reached grasp height, closing gripper.",
                target_id=self.target_id,
            )
        
        # Timeout handling
        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "PREGRASP"
            self.steps_in_stage = 0
        
        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 1.0),
            stage="DESCEND",
            rationale="Descending to grasp height.",
            target_id=self.target_id,
        )
    
    def _handle_close(self, observation: Observation) -> PolicyDecision:
        """Handle CLOSE stage: close gripper to grasp object."""
        self.steps_in_stage += 1
        
        # Hold position while closing gripper (10 steps)
        if self.steps_in_stage < 10:
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, 0.0),
                stage="CLOSE",
                rationale="Closing gripper.",
                target_id=self.target_id,
            )
        
        # Transition to LIFT stage
        self.stage = "LIFT"
        self.steps_in_stage = 0
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 0.0),
            stage="LIFT",
            rationale="Gripper closed, lifting object.",
            target_id=self.target_id,
        )
    
    def _handle_lift(self, observation: Observation) -> PolicyDecision:
        """Handle LIFT stage: lift object to safe height."""
        self.steps_in_stage += 1
        
        # Calculate lift position (20cm above table)
        lift_height = TABLE_TOP_Z + 0.20
        target_pos = np.array([observation.ee_position[0], observation.ee_position[1], lift_height])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        ik_result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=100,
        )
        
        # Check if reached lift height
        if ik_result.converged and ik_result.position_error < 0.01:
            self.stage = "VERIFY"
            self.steps_in_stage = 0
            return PolicyDecision(
                command=JointPositionCommand(ik_result.joint_position, 0.0),
                stage="VERIFY",
                rationale="Object lifted, verifying grasp.",
                target_id=self.target_id,
            )
        
        # Timeout handling
        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "IDLE"
            self.target_world = None
            self.steps_in_stage = 0
        
        return PolicyDecision(
            command=JointPositionCommand(ik_result.joint_position, 0.0),
            stage="LIFT",
            rationale="Lifting object.",
            target_id=self.target_id,
        )
    
    def _handle_verify(self, observation: Observation) -> PolicyDecision:
        """Handle VERIFY stage: check if task is complete."""
        self.steps_in_stage += 1
        
        # Check if object is lifted above table
        if observation.ee_position[2] > TABLE_TOP_Z + 0.10:
            # Task complete
            return PolicyDecision(
                command=JointPositionCommand(HOME_Q, 1.0),
                stage="SUCCESS",
                rationale="Task completed successfully.",
                target_id=self.target_id,
                done=True,
            )
        
        # Timeout handling
        if self.steps_in_stage > self.max_steps_per_stage:
            self.stage = "IDLE"
            self.target_world = None
            self.steps_in_stage = 0
        
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 0.0),
            stage="VERIFY",
            rationale="Verifying grasp success.",
            target_id=self.target_id,
        )
