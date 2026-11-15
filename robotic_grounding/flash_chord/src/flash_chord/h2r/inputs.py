# SPDX-License-Identifier: Apache-2.0
"""The sealed task and this method's preparation, readable in each of the method's Python environments."""

from pathlib import Path

from .h2r_io import load_task
from .h2r_io.files import file_hashes, read_json, sha256, verify_files, write_json

__all__ = ["BINDING", "DIGITS", "FORMAT", "TRAINING_CLOCKS", "frame_names", "load_input", "load_prepared",
           "read_json", "seal_prepared", "sha256", "write_json"]

FORMAT = "flashchord-h2r-v2"  # v1 bound the former human/robot bundles
BINDING = "h2r_binding.json"
ROBOT = "sharpa_wave"  # the native IK, link geometry and contact links exist for this hand only
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
WRIST_LIMIT_RAD = 6.2  # range of the continuous wrist decoder (wrist.py, action.py)
# (control Hz, physics substeps): default 20/5 (100 Hz); 20/6 and 30/4 (120 Hz) are explicit alternatives.
TRAINING_CLOCKS = ((20, 5), (20, 6), (30, 4))


def load_input(path):
    """Load a task directory and reject what the native method cannot represent."""
    if not (Path(path) / "task.json").is_file():
        raise ValueError(f"{path} is not an h2r-bench task directory (no task.json); "
                         f"the former human/robot bundles are not read")
    task = load_task(path)
    robot = task.robot
    if robot.name != ROBOT:
        raise ValueError(f"FlashCHORD's Sharpa IK and hand model do not cover robot {robot.name!r}")
    for side in robot.hands:
        for name in robot.semantics["hands"][side]["wrist_rotation_joints"]:
            joint = robot.compiled["joints"][name]
            if not joint["limited"] or joint["range"] != [-WRIST_LIMIT_RAD, WRIST_LIMIT_RAD]:
                raise ValueError(f"The wrist decoder requires +/-{WRIST_LIMIT_RAD} rad limits on {name}")
    for obj in task.demo.objects.values():
        if not obj.root_joint:
            raise ValueError(f"FlashCHORD's object reset and virtual object control need a free root; "
                             f"interaction object {obj.name!r} has a fixed root")
    return task


def load_prepared(binding_path):
    """The binding of a prepared directory and its sealed task; other formats and changed files are rejected."""
    path = Path(binding_path).resolve()
    binding = read_json(path)
    if binding.get("format") != FORMAT:
        raise ValueError(f"Unsupported FlashCHORD preparation {binding.get('format')!r} (expected {FORMAT}): "
                         f"prepare again from a task directory")
    verify_files(path.parent, binding["files"])
    task = load_input(path.parent / "task")
    if task.task_id != binding["task_id"]:
        raise ValueError("Prepared files belong to a different task")
    return binding, task


def seal_prepared(output, method):
    """Write the binding over a finished preparation; the task copy is sealed by its own task_id."""
    output = Path(output).resolve()
    task = load_input(output / "task")
    files = {name: digest for name, digest in file_hashes(output, exclude=(BINDING,)).items()
             if not name.startswith("task/")}
    write_json(output / BINDING, {
        "format": FORMAT, "task_id": task.task_id, "demo_id": task.demo.demo_id, "robot_id": task.robot.robot_id,
        "dataset": task.dataset, "profile": task.profile, "hand_sides": list(task.hand_sides),
        "duration_s": task.duration_s, "method": method, "files": files})
    return output / BINDING


def frame_names(side):
    """Sharpa frames tracked by the native IK and reference: palm, distal links, fingertips."""
    return [f"{side}_hand_C_MC", *[f"{side}_{d}_DP" for d in DIGITS], *[f"{side}_{d}_fingertip" for d in DIGITS]]
