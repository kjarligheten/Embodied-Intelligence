from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Asset directory path for scene and object files
# Note: MuJoCo on Windows cannot handle paths with non-ASCII characters
# If the default path contains Chinese characters, override via environment variable
ASSET_DIR = Path(__file__).resolve().parent / "assets"
# Allow overriding asset directory via environment variable for Windows compatibility
if "GRASPBENCH_ASSET_DIR" in os.environ:
    ASSET_DIR = Path(os.environ["GRASPBENCH_ASSET_DIR"])
# Path to the main MuJoCo scene XML file
SCENE_PATH = ASSET_DIR / "scene.xml"
# Z-coordinate of the tabletop surface in world coordinates
TABLE_TOP_Z = 0.4
# Home joint configuration for the Franka Panda arm (safe resting pose)
HOME_Q = np.array([0.0, -0.45, 0.0, -2.15, 0.0, 1.75, 0.785], dtype=np.float64)


@dataclass(frozen=True)
class ObjectSpec:
    """Specification for a graspable object in the scene."""
    name: str  # Unique object identifier
    color: str  # Object color description
    shape: str  # Object shape description
    half_height: float  # Half of the object's vertical extent
    grasp_z_offset: float = 0.0  # Vertical offset for grasp point
    sam_prompt: str = ""  # Prompt for SAM3 segmentation
    category: str = "primitive"  # Object category (primitive, fruit, etc.)
    surface_percentile: float = 70.0  # Percentile for surface height estimation

    @property
    def prompt(self) -> str:
        """Get the SAM prompt, defaulting to a generated one if not set."""
        return self.sam_prompt or f"small {self.color} {self.shape}"


@dataclass(frozen=True)
class ContainerSpec:
    """Specification for a container (tray) in the scene."""
    name: str  # Unique container identifier
    shape: str  # Container shape description
    sam_prompt: str  # Prompt for SAM3 segmentation
    inner_half_extents: tuple[float, float] | None = None  # Half extents for rectangular containers
    inner_radius: float | None = None  # Inner radius for circular containers
    floor_height: float = 0.410  # Height of the container floor
    max_object_center_height: float = 0.550  # Maximum allowed object center height

    @property
    def prompt(self) -> str:
        """Get the SAM prompt for this container."""
        return self.sam_prompt


# Tuple of all graspable object specifications
OBJECT_SPECS = (
    ObjectSpec("red_cube", "red", "cube", 0.026, sam_prompt="red square block"),
    ObjectSpec("green_cylinder", "green", "cylinder", 0.031, sam_prompt="green round object"),
    ObjectSpec("blue_box", "blue", "box", 0.026, sam_prompt="blue brick"),
    ObjectSpec("banana", "yellow", "banana", 0.020, sam_prompt="yellow banana", category="fruit"),
    ObjectSpec("apple", "red", "apple", 0.038, sam_prompt="red apple", category="fruit"),
    ObjectSpec("orange", "orange", "orange", 0.038, sam_prompt="small orange round object", category="fruit"),
    ObjectSpec(
        "mustard_bottle",
        "green",
        "mustard bottle",
        0.067,
        sam_prompt="small green object",
        category="packaged_food",
        surface_percentile=98.0,
    ),
    ObjectSpec(
        "potted_meat_can",
        "blue",
        "meat can",
        0.042,
        sam_prompt="small blue object",
        category="packaged_food",
    ),
    ObjectSpec("scissors", "metal", "scissors", 0.009, sam_prompt="scissors", category="tool"),
    ObjectSpec("marker", "black", "marker", 0.011, sam_prompt="black marker pen", category="tool"),
)

# Tuple of all container specifications
CONTAINER_SPECS = (
    ContainerSpec(
        "square_tray",
        "square tray",
        "yellow square tray",
        inner_half_extents=(0.114, 0.114),
    ),
    ContainerSpec(
        "round_tray",
        "round tray",
        "purple round tray",
        inner_radius=0.112,
    ),
)

# Dictionaries for quick lookup by name
OBJECT_SPEC_BY_NAME = {spec.name: spec for spec in OBJECT_SPECS}
CONTAINER_SPEC_BY_NAME = {spec.name: spec for spec in CONTAINER_SPECS}


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single task episode."""
    episode_id: str  # Unique episode identifier
    instruction: str  # Natural language instruction for the task
    target: str  # Target object to manipulate
    seed: int  # Random seed for reproducibility
    split: str = "public"  # Dataset split (public/private)
    destination: str | None = None  # Destination container for placement tasks
    targets: tuple[str, ...] = ()  # Multiple targets for sorting tasks
    destinations: tuple[str, ...] = ()  # Multiple destinations for sorting tasks
    scene_objects: tuple[str, ...] = ()  # All objects present in the scene

    def __post_init__(self) -> None:
        if self.target not in OBJECT_SPEC_BY_NAME:
            raise ValueError(f"Unknown target object: {self.target}")
        if self.destination is not None and self.destination not in CONTAINER_SPEC_BY_NAME:
            raise ValueError(f"Unknown destination container: {self.destination}")
        if bool(self.targets) != bool(self.destinations):
            raise ValueError("targets and destinations must be supplied together")
        if len(self.targets) != len(self.destinations):
            raise ValueError("targets and destinations must have the same length")
        for object_name in self.targets:
            if object_name not in OBJECT_SPEC_BY_NAME:
                raise ValueError(f"Unknown sort target object: {object_name}")
        for container_name in self.destinations:
            if container_name not in CONTAINER_SPEC_BY_NAME:
                raise ValueError(f"Unknown sort destination: {container_name}")
        for object_name in self.scene_objects:
            if object_name not in OBJECT_SPEC_BY_NAME:
                raise ValueError(f"Unknown scene object: {object_name}")

    @classmethod
    def from_dict(cls, value: dict) -> TaskSpec:
        return cls(
            episode_id=str(value["episode_id"]),
            instruction=str(value["instruction"]),
            target=str(value["target"]),
            seed=int(value["seed"]),
            split=str(value.get("split", "public")),
            destination=None
            if value.get("destination") is None
            else str(value["destination"]),
            targets=tuple(str(item) for item in value.get("targets", ())),
            destinations=tuple(str(item) for item in value.get("destinations", ())),
            scene_objects=tuple(str(item) for item in value.get("scene_objects", ())),
        )

    @property
    def goals(self) -> tuple[tuple[str, str | None], ...]:
        """Hidden evaluator goals; never expose this property to policies."""
        if self.targets:
            return tuple(zip(self.targets, self.destinations, strict=True))
        return ((self.target, self.destination),)

    @property
    def task_kind(self) -> str:
        if self.targets:
            return "sort"
        if self.destination is not None:
            return "place"
        return "lift"

    @property
    def active_objects(self) -> tuple[str, ...]:
        """Objects physically present for this hidden episode definition."""
        if self.scene_objects:
            return self.scene_objects
        if self.task_kind == "lift" and self.target in {"red_cube", "green_cylinder", "blue_box"}:
            return ("red_cube", "green_cylinder", "blue_box")
        return tuple(object_name for object_name, _ in self.goals)

    @property
    def control_budget(self) -> int:
        """Evaluation budget; sorting performs four closed-loop pick/place cycles."""
        return 3_000 if self.task_kind == "sort" else 800

    def public_dict(self) -> dict:
        """Information exposed to a policy; the evaluator's target label stays private."""
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "split": self.split,
        }
