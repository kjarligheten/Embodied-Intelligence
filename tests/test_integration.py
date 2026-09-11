"""
Integration tests with live SAM3 service.
Tests perception-control pipeline with real segmentation model.
"""
import os
import pytest

import numpy as np
from graspbench.config import HOME_Q, TABLE_TOP_Z, TaskSpec
from graspbench.env import GraspEnv
from graspbench.perception import FoundationModelPerception
from graspbench.types import Observation
from policies.student_policy import StudentPolicy

# Only run these tests when GRASPBENCH_RUN_SAM3_INTEGRATION is set
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env():
    """Create environment for integration tests."""
    env = GraspEnv()
    yield env
    env.close()


@pytest.fixture(scope="module")
def policy():
    """Create policy instance."""
    return StudentPolicy()


@pytest.fixture(scope="module")
def perception():
    """Create perception module with live SAM3."""
    return FoundationModelPerception()


@pytest.mark.integration
def test_seed_101_red_cube(env, policy, perception):
    """Test with seed 101, target should be red_cube."""
    seed = 101
    target = "red_cube"
    
    # Create TaskSpec for this test
    task = TaskSpec(
        episode_id="test_101",
        instruction="pick up the red cube",
        target=target,
        seed=seed,
        split="public",
        scene_objects=("red_cube", "green_cylinder", "blue_box")
    )
    
    env.reset(task)
    observation = env.observe()
    
    # Detect scene with live SAM3 (use overhead camera at index 0)
    detections, debug = perception.detect_scene(observation.cameras[0])
    
    # Verify red_cube was detected
    assert target in detections, f"Target {target} not detected. Found: {list(detections.keys())}"
    
    # Verify geometry is valid (relaxed for simulation mode)
    red_cube = detections[target]
    assert red_cube.position[2] > TABLE_TOP_Z - 0.02, "Target z position below table"
    assert red_cube.position[2] < TABLE_TOP_Z + 0.15, "Target z position too high"


@pytest.mark.integration
def test_seed_211_green_cylinder(env, policy, perception):
    """Test with seed 211, target should be green_cylinder."""
    seed = 211
    target = "green_cylinder"
    
    # Create TaskSpec for this test
    task = TaskSpec(
        episode_id="test_211",
        instruction="pick up the green cylinder",
        target=target,
        seed=seed,
        split="public",
        scene_objects=("red_cube", "green_cylinder", "blue_box")
    )
    
    env.reset(task)
    observation = env.observe()
    
    # Detect scene with live SAM3 (use overhead camera at index 0)
    detections, debug = perception.detect_scene(observation.cameras[0])
    
    # Verify green_cylinder was detected
    assert target in detections, f"Target {target} not detected. Found: {list(detections.keys())}"
    
    # Verify geometry is valid (relaxed for simulation mode)
    green_cyl = detections[target]
    assert green_cyl.position[2] > TABLE_TOP_Z - 0.02, "Target z position below table"
    assert green_cyl.position[2] < TABLE_TOP_Z + 0.35, "Target z position too high"


@pytest.mark.integration
def test_seed_307_blue_box(env, policy, perception):
    """Test with seed 307, target should be blue_box."""
    seed = 307
    target = "blue_box"
    
    # Create TaskSpec for this test
    task = TaskSpec(
        episode_id="test_307",
        instruction="pick up the blue box",
        target=target,
        seed=seed,
        split="public",
        scene_objects=("red_cube", "green_cylinder", "blue_box")
    )
    
    env.reset(task)
    observation = env.observe()
    
    # Detect scene with live SAM3 (use overhead camera at index 0)
    detections, debug = perception.detect_scene(observation.cameras[0])
    
    # Verify blue_box was detected
    assert target in detections, f"Target {target} not detected. Found: {list(detections.keys())}"
    
    # Verify geometry is valid (relaxed for simulation mode)
    blue_box = detections[target]
    assert blue_box.position[2] > TABLE_TOP_Z - 0.02, "Target z position below table"
    assert blue_box.position[2] < TABLE_TOP_Z + 0.15, "Target z position too high"
