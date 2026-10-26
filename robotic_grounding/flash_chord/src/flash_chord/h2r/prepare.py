# SPDX-License-Identifier: Apache-2.0
"""Prepare native Sharpa references from fixed H2R inputs in isolated environments."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .inputs import DIGITS, FORMAT, ROBOT, SIDES, digest, frame_names, human_bundle, robot_bundle, write_json


def check_ik_model(solver, model, state, side, fingers, wrist_ids, base):
    """Compare derived Pinocchio FK to common MuJoCo FK at nonzero poses."""
    import mujoco

    from .wrist import wrist_coordinates

    rng = np.random.default_rng(17)
    finger_ids = [model.joint(name).qposadr[0] for name in fingers]
    ranges = np.array([model.joint(name).range for name in fingers])
    np.testing.assert_allclose(solver.robot.model.lowerPositionLimit[7:], ranges[:, 0], atol=1e-8)
    np.testing.assert_allclose(solver.robot.model.upperPositionLimit[7:], ranges[:, 1], atol=1e-8)
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
    return max_error


def features(output, mano_root, device):
    import torch
    import trimesh
    from robotic_grounding.retarget.params import NUM_MANO_LINKS
    from v2d.task_library_loader.lib.contact_utils import (
        approximate_contact_with_id,
        compute_hand_link_contact_positions,
        find_object_contact_positions,
    )
    from v2d.task_library_loader.lib.read_mano import MANO

    output = Path(output)
    _, ref, scene, arrays = human_bundle(output / "human")
    # manotorch expects <root>/models; licensed models stay outside output artifacts.
    with tempfile.TemporaryDirectory() as temporary:
        models = Path(temporary) / "models"
        models.mkdir()
        for side in SIDES:
            source = Path(mano_root) / f"MANO_{side.upper()}.pkl"
            if digest(source) != next(a["sha256"] for a in ref["mano"]["assets"] if a["name"] == source.name):
                raise ValueError(f"MANO asset differs from common input: {source}")
            (models / source.name).symlink_to(source.resolve())
        mano = MANO(temporary, device=torch.device(device), flat_hand_mean=False, center_idx=None)
    count = len(arrays["time_s"])
    # Bench's MANO16 + five tips -> the native wrist/finger-chain order.
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    sampled = []
    for i, part in enumerate(("bottom", "top")):
        mesh = trimesh.load(output / "human" / scene["resources"][part]["visual"], force="mesh", process=False)
        points, faces = trimesh.sample.sample_surface_even(mesh, 4096, seed=1 + i)
        sampled.append((points, -mesh.face_normals[faces]))
    result = {}
    checks = {"surface_samples_per_part": [len(points) for points, _ in sampled], "surface_seeds": [1, 2]}
    for side in SIDES:

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        with torch.no_grad():
            decoded = mano.forward(
                side,
                tensor(arrays[f"{side}_mano_shape"]),
                tensor(arrays[f"{side}_mano_rot"]),
                tensor(arrays[f"{side}_mano_trans"]),
                tensor(arrays[f"{side}_mano_pose"]),
            )
        joints = decoded["joints"].cpu().numpy()
        differences = np.linalg.norm(joints - arrays[f"{side}_joints_m"][:, order], axis=-1)
        tip_slots = [4, 8, 12, 16, 20]
        error = float(np.max(np.delete(differences, tip_slots, axis=1)))
        vertices_error = float(
            np.max(np.linalg.norm(decoded["vertices"].cpu().numpy() - arrays[f"{side}_vertices_m"], axis=-1))
        )
        if max(error, vertices_error) > 1e-4:
            raise ValueError(f"{side} MANO geometry differs: joints={error}m, surface={vertices_error}m")
        # manotorch and smplx select different surface vertices as fingertips.
        # Keep the method's native feature, but verify it against the SAME human surface.
        native_tips = [745, 317, 444 if side == "right" else 445, 556, 673]
        np.testing.assert_allclose(
            joints[:, tip_slots], arrays[f"{side}_vertices_m"][:, native_tips], atol=1e-6, rtol=0
        )
        checks[side] = {
            "shared_joint_max_error_m": error,
            "vertex_max_error_m": vertices_error,
            "native_tip_vertex_ids": native_tips,
            "native_vs_bench_tip_max_distance_m": float(differences[:, tip_slots].max()),
        }
        result[f"{side}_joints"] = joints
        result[f"{side}_joints_wxyz"] = decoded["joints_wxyz"].cpu().numpy()
        positions, normals, ids = [], [], []
        for t in range(count):
            points_w, normals_w, part_ids = [], [], []
            for part, (points, normal) in enumerate(sampled):
                rotation = Rotation.from_quat(arrays["part_quat_wxyz"][t, part][[1, 2, 3, 0]])
                points_w.append(rotation.apply(points) + arrays["part_pos_m"][t, part])
                normals_w.append(rotation.apply(normal))
                part_ids.extend([part + 1] * len(points))
            points_w, normals_w, part_ids = (
                tensor(np.concatenate(points_w)),
                tensor(np.concatenate(normals_w)),
                tensor(part_ids),
            )
            mesh = trimesh.Trimesh(decoded["vertices"][t].cpu().numpy(), decoded["faces"].cpu().numpy(), process=False)
            contact = approximate_contact_with_id(
                points_w,
                normals_w,
                part_ids,
                decoded["vertices"][t],
                tensor(np.array(mesh.vertex_normals)),
                threshold=0.01,
            )
            pos, normal = tensor(np.zeros((NUM_MANO_LINKS, 3))), tensor(np.zeros((NUM_MANO_LINKS, 3)))
            pid = tensor(np.zeros(NUM_MANO_LINKS))
            if len(contact[-1]):
                hand_pos, _, pid = compute_hand_link_contact_positions(decoded["joints"][t], *contact[2:])
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
    import mujoco
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
    from robotic_grounding.retarget.retarget_utils import wrist_pose_from_mano_joint0

    from .wrist import wrist_coordinates

    output = Path(output)
    _, ref, scene, arrays = human_bundle(output / "human")
    _, semantics, _ = robot_bundle(output / "robot")
    with np.load(output / "features.npz", allow_pickle=False) as data:
        features = dict(data)
    model = mujoco.MjModel.from_xml_path(str(output / "assets" / "robot.xml"))
    state = mujoco.MjData(model)
    cells = {
        "fps": 30.0,
        "sequence_id": f"{ref['subject']}/{ref['sequence']}",
        "robot_name": "sharpa_wave",
        "source_dataset": "arctic",
        "object_name": ref["object"],
        "object_body_names": ["bottom", "top"],
        "object_body_position": arrays["part_pos_m"].tolist(),
        "object_body_wxyz": arrays["part_quat_wxyz"].tolist(),
        "object_articulation": arrays["object_hinge_rad"].tolist(),
        "object_urdf_paths": [],
        "object_mesh_paths": [str(Path("human") / scene["resources"][p]["visual"]) for p in ("bottom", "top")],
    }
    cells["object_mesh_radius"] = [
        float(np.max(np.linalg.norm(trimesh.load(output / p, force="mesh").vertices, axis=-1)))
        for p in cells["object_mesh_paths"]
    ]
    joint_q = np.zeros((len(arrays["time_s"]), 56))
    report = {}
    for side in SIDES:
        solver = SharpaHandKinematics(
            side=side,
            robot_asset_path=str(output / "assets" / f"ik_{side}.xml"),
            source_model="mano",
            use_relative_frames=False,
            max_iter=100,
            frequency=200.0,
            frame_tasks_converged_threshold=1e-6,
        )
        fingers = list(solver.robot_finger_joint_names.values())
        names = frame_names(side)
        q = None
        previous = None
        wrists, quats, finger_values, frames, errors, iterations = [], [], [], [], [], []
        finger_ids = [model.joint(name).qposadr[0] for name in fingers]
        wrist_ids = [model.joint(f"{side}_{axis}_joint").qposadr[0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
        state.qpos[:] = 0
        mujoco.mj_forward(model, state)
        base = state.body(f"{side}_hand_C_MC").xpos.copy()
        fk_error = check_ik_model(solver, model, state, side, fingers, wrist_ids, base)
        for t in range(len(joint_q)):
            if q is None:
                pos, quat = wrist_pose_from_mano_joint0(
                    features[f"{side}_joints"][t, 0],
                    features[f"{side}_joints_wxyz"][t, 0],
                    link_to_site_quat_xyzw=np.array([-0.5, 0.5, 0.5, 0.5]),
                )
                q = solver.robot.q0.copy()
                q[:3] = pos
                q[3:7] = quat
            try:
                result = solver.compute(
                    features[f"{side}_joints"][t], features[f"{side}_joints_wxyz"][t], source_to_robot_scale=1.0, qpos=q
                )
            except Exception as error:
                raise RuntimeError(f"IK failed for {side}, source frame {arrays['source_frame'][t]}") from error
            q = result["q"]
            wrist = wrist_coordinates(q[:3], q[[6, 3, 4, 5]], base, previous)
            previous = wrist[3:]
            joint_q[t, wrist_ids] = wrist
            joint_q[t, finger_ids] = q[7:]
            state.qpos[:] = joint_q[t]
            mujoco.mj_forward(model, state)
            palm = state.body(f"{side}_hand_C_MC")
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
            f"Invalid IK at source frame {arrays['source_frame'][frame]}, "
            f"{semantics['joint_names'][joint]}={joint_q[frame, joint]}"
        )
    pq.write_table(pa.Table.from_pylist([cells]), output / "processed.parquet")
    np.savez_compressed(
        output / "robot_reference.npz", joint_q=joint_q, joint_names=semantics["joint_names"], time_s=arrays["time_s"]
    )
    write_json(output / "checks" / "ik.json", report)


def prepare(args):
    from .assets import prepare_assets

    human_bundle(args.human_bundle)
    robot_bundle(args.robot_bundle)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checks").mkdir()
    shutil.copytree(args.human_bundle, output / "human")
    shutil.copytree(args.robot_bundle, output / "robot")
    prepare_assets(output / "human", output / "robot", output / "assets")
    subprocess.run(
        [
            args.loader_python,
            "-m",
            "flash_chord.h2r.prepare",
            "features",
            "--output",
            str(output),
            "--mano-root",
            args.mano_root,
            "--device",
            args.device,
        ],
        check=True,
    )
    subprocess.run([args.ik_python, "-m", "flash_chord.h2r.prepare", "ik", "--output", str(output)], check=True)
    seal(output)


def seal(output):
    output = Path(output).resolve()
    human_manifest, ref, _, _ = human_bundle(output / "human")
    robot_bundle(output / "robot")
    for name in ("processed.parquet", "robot_reference.npz", "checks/mano.json", "checks/ik.json"):
        if not (output / name).is_file():
            raise ValueError(f"Incomplete preparation: {name}")
    binding = {
        "format": FORMAT,
        "robot_name": ROBOT,
        "human_bundle_id": human_manifest["bundle_id"],
        "scene_id": human_manifest["scene_id"],
        "source_frame_range": ref["source_frame_range"],
        "robot_manifest_sha256": digest(output / "robot" / "robot.json"),
        "reference_hz": 30,
        "duration_s": ref["duration_s"],
        "method": {
            "contact_threshold_m": 0.01,
            "requested_surface_samples_per_part": 4096,
            "surface_seeds": [1, 2],
            "ik_max_iter": 100,
            "ik_frequency": 200.0,
            "ik_convergence_threshold": 1e-6,
            "mano_to_robot_scale": 1.0,
        },
        "files": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "h2r_binding.json"
        },
    }
    write_json(output / "h2r_binding.json", binding)
    print(output / "h2r_binding.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "features", "ik", "seal"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--human-bundle")
    parser.add_argument("--robot-bundle")
    parser.add_argument("--mano-root")
    parser.add_argument("--loader-python")
    parser.add_argument("--ik-python")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "prepare":
        for name in ("human_bundle", "robot_bundle", "mano_root", "loader_python", "ik_python"):
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
