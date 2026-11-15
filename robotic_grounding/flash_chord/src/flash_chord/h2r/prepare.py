# SPDX-License-Identifier: Apache-2.0
"""Prepare native Sharpa references from one sealed H2R task in isolated environments."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .h2r_io import export_task
from .inputs import DIGITS, frame_names, load_input, read_json, seal_prepared, sha256, write_json

# The method's own preparation settings; the sealed binding records them.
METHOD = {
    "contact_threshold_m": 0.01,
    "requested_surface_samples_per_part": 4096,
    "ik_max_iter": 100,
    "ik_frequency": 200.0,
    "ik_convergence_threshold": 1e-6,
}
# Upstream Sharpa retargeting entries (robotic_grounding/scripts/retarget/<dataset>_to_sharpa.py): the default
# `--mano_to_robot_scale` and the dataset's MANO link-to-site wrist offset used for the first IK seed.
IK_SETTINGS = {
    "arctic": {"mano_to_robot_scale": 1.0, "wrist_offset_xyzw": [-0.5, 0.5, 0.5, 0.5]},  # arctic_to_sharpa.py
    "oakink2": {"mano_to_robot_scale": 1.2, "wrist_offset_xyzw": None},  # oakink2_to_sharpa.py
    "hot3d": {"mano_to_robot_scale": 1.2, "wrist_offset_xyzw": None},  # hot3d_to_sharpa.py
    "taco": {"mano_to_robot_scale": 1.2, "wrist_offset_xyzw": None},  # taco_to_sharpa.py
}
# Bench's MANO16 + five tips -> the native wrist/finger-chain order.
NATIVE_JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
NATIVE_TIP_SLOTS = [4, 8, 12, 16, 20]


def ik_settings(dataset):
    """The upstream Sharpa retargeting recipe of one dataset; there is no default recipe."""
    if dataset not in IK_SETTINGS:
        raise ValueError(f"No Sharpa IK recipe for dataset {dataset!r}; known: {sorted(IK_SETTINGS)}")
    return IK_SETTINGS[dataset]


def check_ik_model(solver, model, state, side, fingers, wrist_ids, base):
    """Compare derived Pinocchio FK to common MuJoCo FK at nonzero poses."""
    import mujoco

    from .wrist import wrist_coordinates

    rng = np.random.default_rng(17)
    finger_ids = [model.joint(name).qposadr[0] for name in fingers]
    ranges = np.array([model.joint(name).range for name in fingers])
    np.testing.assert_allclose(solver.robot.model.lowerPositionLimit[7:], ranges[:, 0], atol=1e-8)
    np.testing.assert_allclose(solver.robot.model.upperPositionLimit[7:], ranges[:, 1], atol=1e-8)
    original_q = solver.configuration.q.copy()
    max_error = 0.0
    for _ in range(5):
        q = solver.robot.q0.copy()
        q[:3] = rng.uniform(-0.3, 0.3, 3)
        q[3:7] = Rotation.from_euler("XYZ", rng.uniform(-2, 2, 3)).as_quat()
        q[7:] = rng.uniform(ranges[:, 0], ranges[:, 1])
        solver.configuration.update(q)
        state.qpos[:] = 0
        state.qpos[wrist_ids] = wrist_coordinates(q[:3], q[[6, 3, 4, 5]], base)
        state.qpos[finger_ids] = q[7:]
        mujoco.mj_forward(model, state)
        for name in frame_names(side)[:6]:
            pose = solver.configuration.get_transform_frame_to_world(name)
            common = state.body(name)
            error = np.max(np.abs(pose.translation - common.xpos))
            max_error = max(max_error, float(error))
            np.testing.assert_allclose(pose.translation, common.xpos, atol=2e-6, rtol=0)
            np.testing.assert_allclose(pose.rotation, common.xmat.reshape(3, 3), atol=2e-6, rtol=0)
        for digit in DIGITS:
            pose = solver.configuration.get_transform_frame_to_world(f"{side}_{digit}_tip_site")
            np.testing.assert_allclose(
                pose.translation, state.body(f"{side}_{digit}_fingertip").xpos, atol=2e-6, rtol=0
            )
            # Native task-site orientation is a half turn around distal Z.
            expected = state.body(f"{side}_{digit}_DP").xmat.reshape(3, 3) @ np.diag([-1, -1, 1])
            np.testing.assert_allclose(pose.rotation, expected, atol=2e-6, rtol=0)
    # The native solver reuses Pinocchio's cached transforms on its first
    # iteration. A geometry check must not change that initialization.
    solver.configuration.update(original_q)
    return max_error


def native_mano(mano_root, assets, device):
    """The method's MANO layer (manotorch); licensed models stay outside output artifacts."""
    import torch
    from v2d.task_library_loader.lib.read_mano import MANO

    # manotorch expects <root>/models.
    with tempfile.TemporaryDirectory() as temporary:
        models = Path(temporary) / "models"
        models.mkdir()
        for asset in assets:
            source = Path(mano_root) / asset["name"]
            if sha256(source) != asset["sha256"]:
                raise ValueError(f"MANO asset differs from common input: {source}")
            (models / source.name).symlink_to(source.resolve())
        # h2r-mano-v1: absolute rotations over the flat mean hand, translation of the wrist joint.
        return MANO(temporary, device=torch.device(device), flat_hand_mean=True, center_idx=0)


def features(output, mano_root, device):
    """Native joint frames and hand-object contact labels of every participating hand."""
    import torch
    import trimesh
    from robotic_grounding.retarget.params import NUM_MANO_LINKS
    from v2d.task_library_loader.lib.contact_utils import (
        approximate_contact_with_id,
        compute_hand_link_contact_positions,
        find_object_contact_positions,
    )

    output = Path(output)
    demo = load_input(output / "task").demo
    convention = demo.reference["mano"]
    mano = native_mano(mano_root, convention["assets"], device)
    if convention["left_shapedirs_x_mirrored"]:
        mano.left_mano_layer.th_shapedirs[:, 0, :] *= -1  # the same left shape basis as the common decoding
    count = len(demo.time_s)
    sampled = []
    for i, part in enumerate(demo.part_order):
        mesh = trimesh.load(demo.path / demo.scene["resources"][part]["visual"], force="mesh", process=False)
        points, faces = trimesh.sample.sample_surface_even(mesh, METHOD["requested_surface_samples_per_part"],
                                                           seed=1 + i)
        sampled.append((points, -mesh.face_normals[faces]))
    result = {}
    checks = {"surface_samples_per_part": [len(points) for points, _ in sampled],
              "surface_seeds": list(range(1, len(sampled) + 1))}

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    for side, hand in demo.hands.items():
        pose = Rotation.from_quat(hand.mano_quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(count, 48)
        with torch.no_grad():
            decoded = mano.forward(side, tensor(hand.mano_shape), tensor(pose[:, :3]),
                                   tensor(hand.mano_trans), tensor(pose[:, 3:]))
        # The native layer supplies its anatomical joint frames and closed faces. The geometry is
        # the demo's; decoding it again only proves that both read the same MANO parameters.
        differences = np.linalg.norm(decoded["joints"].cpu().numpy() - hand.joints_m[:, NATIVE_JOINT_ORDER], axis=-1)
        error = float(np.max(np.delete(differences, NATIVE_TIP_SLOTS, axis=1)))
        vertices_error = float(np.max(np.linalg.norm(decoded["vertices"].cpu().numpy() - hand.vertices_m, axis=-1)))
        if max(error, vertices_error) > 1e-4:
            raise ValueError(f"{side} MANO geometry differs: joints={error}m, surface={vertices_error}m")
        # manotorch and the common format select different surface vertices as fingertips.
        # Keep the method's native feature, taken from the SAME human surface.
        native_tips = [745, 317, 444 if side == "right" else 445, 556, 673]
        joints = hand.joints_m[:, NATIVE_JOINT_ORDER].copy()
        joints[:, NATIVE_TIP_SLOTS] = hand.vertices_m[:, native_tips]
        checks[side] = {
            "shared_joint_max_error_m": error,
            "vertex_max_error_m": vertices_error,
            "native_tip_vertex_ids": native_tips,
            "native_vs_bench_tip_max_distance_m": float(differences[:, NATIVE_TIP_SLOTS].max()),
        }
        result[f"{side}_joints"] = joints
        result[f"{side}_joints_wxyz"] = decoded["joints_wxyz"].cpu().numpy()
        faces = decoded["faces"].cpu().numpy()
        positions, normals, ids = [], [], []
        for t in range(count):
            points_w, normals_w, part_ids = [], [], []
            for index, (part, (points, normal)) in enumerate(zip(demo.part_order, sampled)):
                position, quaternion = demo.part_pose(part)
                rotation = Rotation.from_quat(quaternion[t][[1, 2, 3, 0]])
                points_w.append(rotation.apply(points) + position[t])
                normals_w.append(rotation.apply(normal))
                part_ids.extend([index + 1] * len(points))
            points_w, normals_w, part_ids = (
                tensor(np.concatenate(points_w)),
                tensor(np.concatenate(normals_w)),
                tensor(part_ids),
            )
            vertices = tensor(hand.vertices_m[t])
            mesh = trimesh.Trimesh(hand.vertices_m[t], faces, process=False)
            contact = approximate_contact_with_id(
                points_w,
                normals_w,
                part_ids,
                vertices,
                tensor(np.array(mesh.vertex_normals)),
                threshold=METHOD["contact_threshold_m"],
            )
            pos, normal = tensor(np.zeros((NUM_MANO_LINKS, 3))), tensor(np.zeros((NUM_MANO_LINKS, 3)))
            pid = tensor(np.zeros(NUM_MANO_LINKS))
            if len(contact[-1]):
                hand_pos, _, pid = compute_hand_link_contact_positions(tensor(joints[t]), *contact[2:])
                pos, normal = find_object_contact_positions(hand_pos, points_w, normals_w)
                # A failed projection is an inactive slot, never an active zero normal.
                valid = (torch.linalg.vector_norm(normal, dim=-1) > 1e-6) & (pid > 0)
                pid = torch.where(valid, pid, 0)
                normal = torch.where(valid[:, None], torch.nn.functional.normalize(normal, dim=-1), 0)
                pos = torch.where(valid[:, None], pos, 0)
            positions.append(pos.cpu().numpy())
            normals.append(normal.cpu().numpy())
            ids.append(pid.cpu().numpy())
        result[f"{side}_contact_pos"] = np.asarray(positions)
        result[f"{side}_contact_normal"] = np.asarray(normals)
        result[f"{side}_contact_id"] = np.asarray(ids, dtype=np.int32)
    np.savez_compressed(output / "features.npz", **result)
    write_json(output / "checks" / "mano.json", checks)


def ik(output):
    """Native Sharpa IK of every participating hand, written as the native reference table."""
    import mujoco
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
    from robotic_grounding.retarget.retarget_utils import wrist_pose_from_mano_joint0

    from .wrist import wrist_coordinates

    output = Path(output)
    task = load_input(output / "task")
    demo, semantics = task.demo, task.robot.semantics
    settings = ik_settings(task.dataset)
    with np.load(output / "features.npz", allow_pickle=False) as data:
        features = dict(data)
    model = mujoco.MjModel.from_xml_path(str(output / "assets" / "robot.xml"))
    state = mujoco.MjData(model)
    parts = list(demo.part_order)
    joints = [joint for obj in demo.objects.values() for joint in obj.joints]
    cells = {
        "fps": demo.reference_hz,
        "sequence_id": f"{task.dataset}/{task.profile}",
        "robot_name": task.robot.name,
        "source_dataset": task.dataset,
        "object_name": "+".join(demo.objects),
        "object_body_names": parts,
        "object_body_position": demo.arrays["part_pos_m"].tolist(),
        "object_body_wxyz": demo.arrays["part_quat_wxyz"].tolist(),
        # One column per object joint, in the order of the scene's interaction objects.
        "object_articulation": np.stack([joint.position_rad for joint in joints], axis=1).tolist() if joints else [],
        "object_urdf_paths": [],
        "object_mesh_paths": [str(Path("task/demo") / demo.scene["resources"][p]["visual"]) for p in parts],
    }
    cells["object_mesh_radius"] = [
        float(np.max(np.linalg.norm(trimesh.load(output / p, force="mesh").vertices, axis=-1)))
        for p in cells["object_mesh_paths"]
    ]
    joint_q = np.zeros((len(demo.time_s), len(semantics["joint_names"])))
    report = {}
    for side in task.hand_sides:
        solver = SharpaHandKinematics(
            side=side,
            robot_asset_path=str(output / "assets" / f"ik_{side}.xml"),
            source_model="mano",
            use_relative_frames=False,
            max_iter=METHOD["ik_max_iter"],
            frequency=METHOD["ik_frequency"],
            frame_tasks_converged_threshold=METHOD["ik_convergence_threshold"],
        )
        fingers = list(solver.robot_finger_joint_names.values())
        hand = semantics["hands"][side]
        if set(fingers) != set(hand["finger_joints"]):
            raise ValueError(f"The native {side} IK joints differ from the robot's finger joints")
        names = frame_names(side)
        q = None
        previous = None
        wrists, quats, finger_values, frames, errors, iterations = [], [], [], [], [], []
        finger_ids = [model.joint(name).qposadr[0] for name in fingers]
        wrist_ids = [model.joint(name).qposadr[0]
                     for name in hand["wrist_translation_joints"] + hand["wrist_rotation_joints"]]
        state.qpos[:] = 0
        mujoco.mj_forward(model, state)
        base = state.body(hand["wrist_body"]).xpos.copy()
        fk_error = check_ik_model(solver, model, state, side, fingers, wrist_ids, base)
        for t in range(len(joint_q)):
            if q is None:
                pos, quat = wrist_pose_from_mano_joint0(
                    features[f"{side}_joints"][t, 0],
                    features[f"{side}_joints_wxyz"][t, 0],
                    link_to_site_quat_xyzw=settings["wrist_offset_xyzw"],
                )
                q = solver.robot.q0.copy()
                q[:3] = pos
                q[3:7] = quat
            try:
                result = solver.compute(
                    features[f"{side}_joints"][t], features[f"{side}_joints_wxyz"][t],
                    source_to_robot_scale=settings["mano_to_robot_scale"], qpos=q
                )
            except Exception as error:
                raise RuntimeError(f"IK failed for {side}, frame {t}, time {demo.time_s[t]}") from error
            q = result["q"]
            wrist = wrist_coordinates(q[:3], q[[6, 3, 4, 5]], base, previous)
            previous = wrist[3:]
            joint_q[t, wrist_ids] = wrist
            joint_q[t, finger_ids] = q[7:]
            state.qpos[:] = joint_q[t]
            mujoco.mj_forward(model, state)
            palm = state.body(hand["wrist_body"])
            np.testing.assert_allclose(palm.xpos, q[:3], atol=2e-6, rtol=0)
            if abs(np.dot(palm.xquat, q[[6, 3, 4, 5]])) < 1 - 1e-9:
                raise ValueError(f"Wrist FK mismatch at {side} frame {t}")
            wrists.append(palm.xpos.copy())
            quats.append(palm.xquat.copy())
            finger_values.append(q[7:].copy())
            frames.append([np.r_[state.body(n).xpos, state.body(n).xquat].tolist() for n in names])
            errors.append(result["frame_task_errors"])
            iterations.append(result["num_optimization_iterations"])
            if t % 100 == 0:
                print(f"IK {side}: {t + 1}/{len(joint_q)} frames", flush=True)
        for key, values in (
            ("wrist_position", wrists),
            ("wrist_wxyz", quats),
            ("finger_joints", finger_values),
            ("frames", frames),
        ):
            cells[f"robot_{side}_{key}"] = np.asarray(values).tolist()
        cells[f"{side}_robot_finger_joint_names"] = fingers
        cells[f"{side}_robot_frame_names"] = names
        for key, source in (("positions", "pos"), ("normals", "normal"), ("part_ids", "id")):
            cells[f"mano_{side}_object_contact_{key}"] = features[f"{side}_contact_{source}"].tolist()
        report[side] = {
            **settings,
            "fk_max_error_m": fk_error,
            "frame_task_names": list(solver.frame_tasks),
            "frame_task_errors_m": errors,
            "iterations": iterations,
        }
    lower, upper = model.jnt_range.T
    invalid = ~np.isfinite(joint_q) | (joint_q < lower - 1e-5) | (joint_q > upper + 1e-5)
    if invalid.any():
        frame, joint = np.argwhere(invalid)[0]
        raise ValueError(
            f"Invalid IK at time {demo.time_s[frame]}, {semantics['joint_names'][joint]}={joint_q[frame, joint]}"
        )
    pq.write_table(pa.Table.from_pylist([cells]), output / "processed.parquet")
    np.savez_compressed(
        output / "robot_reference.npz", joint_q=joint_q, joint_names=semantics["joint_names"], time_s=demo.time_s
    )
    write_json(output / "checks" / "ik.json", report)


def prepare(args):
    from .assets import prepare_assets

    task = load_input(args.task)
    ik_settings(task.dataset)  # fail before any work on a dataset without a recipe
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checks").mkdir()
    # The preparation keeps its own sealed copy of the task: checkpoints refer to this directory.
    export_task(task, output / "task")
    prepare_assets(load_input(output / "task"), output / "assets")
    stage = ["-m", "flash_chord.h2r.prepare"]
    subprocess.run([args.loader_python, *stage, "features", "--output", str(output),
                    "--mano-root", args.mano_root, "--device", args.device], check=True)
    subprocess.run([args.ik_python, *stage, "ik", "--output", str(output)], check=True)
    seal(output)


def seal(output):
    output = Path(output).resolve()
    task = load_input(output / "task")
    for name in ("processed.parquet", "robot_reference.npz", "checks/mano.json", "checks/ik.json"):
        if not (output / name).is_file():
            raise ValueError(f"Incomplete preparation: {name}")
    settings = ik_settings(task.dataset)
    ik_report = read_json(output / "checks/ik.json")
    if any(ik_report[side].get(key) != value for side in task.hand_sides for key, value in settings.items()):
        raise ValueError("IK settings differ from the selected dataset recipe; rerun IK before sealing")
    print(seal_prepared(output, {**METHOD, **settings}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "features", "ik", "seal"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", help="sealed h2r-bench task directory (standalone or inside a suite)")
    parser.add_argument("--mano-root")
    parser.add_argument("--loader-python")
    parser.add_argument("--ik-python")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "prepare":
        for name in ("task", "mano_root", "loader_python", "ik_python"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for prepare")
        prepare(args)
    elif args.stage == "features":
        features(args.output, args.mano_root, args.device)
    elif args.stage == "ik":
        ik(args.output)
    else:
        seal(args.output)


if __name__ == "__main__":
    main()
