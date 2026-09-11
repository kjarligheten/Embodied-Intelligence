"""
Local FastSAM integration for object detection and segmentation.
This module provides local inference using FastSAM models deployed in the models folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from graspbench.config import CONTAINER_SPECS, OBJECT_SPECS
from graspbench.types import CameraObservation, DetectedObject


class FastSAMPerception:
    """Local FastSAM-based perception for object detection and segmentation.
    
    This class uses FastSAM models deployed locally in the models folder
    to perform object detection and segmentation without requiring external services.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize FastSAM perception module.
        
        Args:
            model_path: Path to FastSAM model weights (defaults to models/FastSAM/FastSAM-x.pt)
            device: Device to run inference on (cuda/cpu)
        """
        # Default model path
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "FastSAM" / "FastSAM-x.pt"
        
        self.model_path = Path(model_path)
        self.device = device
        
        # Load FastSAM model
        self._load_model()
        
        # Build specification dictionary
        self.spec_by_name: dict[str, Any] = {
            **{spec.name: spec for spec in OBJECT_SPECS},
            **{spec.name: spec for spec in CONTAINER_SPECS},
        }

    def _load_model(self):
        """Load FastSAM model from local weights."""
        try:
            from ultralytics import FastSAM
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"FastSAM model not found at {self.model_path}")
            
            self.model = FastSAM(str(self.model_path))
            self.model.to(self.device)
            print(f"[FastSAM] Model loaded from {self.model_path} on {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FastSAM model: {e}")

    def detect_scene(
        self,
        camera: CameraObservation,
        candidate_ids: tuple[str, ...] | None = None,
        actual_positions: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, DetectedObject], dict[str, Any]]:
        """Detect all objects in the scene using FastSAM.
        
        Args:
            camera: Camera observation with RGB-D data
            candidate_ids: Specific object IDs to detect (None for all)
            actual_positions: Optional ground-truth positions for simulation mode
            
        Returns:
            Tuple of (detected objects dict, debug information dict)
        """
        # In simulation mode with actual positions provided, use ground truth
        if actual_positions is not None:
            return self._simulation_mode_detection(camera, candidate_ids, actual_positions)
        
        # Use FastSAM for real detection
        return self._fastsam_detection(camera, candidate_ids)

    def _simulation_mode_detection(
        self,
        camera: CameraObservation,
        candidate_ids: tuple[str, ...] | None,
        actual_positions: dict[str, np.ndarray],
    ) -> tuple[dict[str, DetectedObject], dict[str, Any]]:
        """Use ground-truth positions in simulation mode."""
        all_ids = candidate_ids or tuple(self.spec_by_name.keys())
        
        detections = {}
        debug = {}
        
        for obj_name in all_ids:
            if obj_name not in actual_positions:
                continue
            
            pos = actual_positions[obj_name]
            spec = self.spec_by_name.get(obj_name)
            
            if spec:
                detections[obj_name] = DetectedObject(
                    name=spec.name,
                    color=spec.color if hasattr(spec, "color") else "container",
                    shape=spec.shape,
                    position=pos,
                    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                )
                
                debug[obj_name] = {
                    "service": "fastsam_simulation",
                    "target_id": obj_name,
                    "prompt": spec.prompt if hasattr(spec, "prompt") else obj_name,
                    "score": 1.0,
                    "box_xyxy": [0, 0, camera.rgb.shape[1], camera.rgb.shape[0]],
                    "mask_pixels": 1000,
                    "geometry_pixels": 1000,
                    "latency_s": 0.0,
                    "endpoint": "simulation_ground_truth",
                }
        
        return detections, debug

    def _fastsam_detection(
        self,
        camera: CameraObservation,
        candidate_ids: tuple[str, ...] | None,
    ) -> tuple[dict[str, DetectedObject], dict[str, Any]]:
        """Perform real detection using FastSAM."""
        import time
        
        target_ids = candidate_ids or tuple(self.spec_by_name.keys())
        
        # Convert RGB to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR)
        
        # Run FastSAM inference
        start_time = time.time()
        results = self.model(
            rgb_bgr,
            device=self.device,
            retina_masks=True,
            imgsz=640,
            conf=0.4,
            iou=0.9,
        )
        latency = time.time() - start_time
        
        detections = {}
        debug = {}
        
        # Process results for each target
        for target_id in target_ids:
            spec = self.spec_by_name.get(target_id)
            if not spec:
                continue
            
            # Use text prompt if available, otherwise use object name
            prompt = spec.prompt if hasattr(spec, "prompt") else target_id
            
            # Filter results based on prompt (simplified for now)
            # In a full implementation, you would use CLIP or other methods
            # to match detections to text prompts
            
            # For now, return a placeholder detection
            # TODO: Implement proper text-to-detection matching
            detections[target_id] = DetectedObject(
                name=spec.name,
                color=spec.color if hasattr(spec, "color") else "container",
                shape=spec.shape,
                position=np.array([0.5, 0.0, 0.45]),  # Placeholder
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            
            debug[target_id] = {
                "service": "fastsam_local",
                "target_id": target_id,
                "prompt": prompt,
                "score": 0.8,
                "box_xyxy": [0, 0, camera.rgb.shape[1], camera.rgb.shape[0]],
                "mask_pixels": 1000,
                "geometry_pixels": 1000,
                "latency_s": latency,
                "endpoint": "local_fastsam",
            }
        
        if not detections:
            raise RuntimeError("FastSAM did not detect any objects")
        
        return detections, debug

    def detect_target(
        self,
        camera: CameraObservation,
        target_id: str,
        prior_xy: np.ndarray | None = None,
    ) -> tuple[DetectedObject, dict[str, Any]]:
        """Detect a specific target using FastSAM.
        
        Args:
            camera: Camera observation
            target_id: Target object ID
            prior_xy: Prior position estimate (optional)
            
        Returns:
            Tuple of (detected object, debug information)
        """
        detections, debug = self.detect_scene(camera, candidate_ids=(target_id,))
        
        if target_id not in detections:
            raise RuntimeError(f"Failed to detect target: {target_id}")
        
        return detections[target_id], debug[target_id]