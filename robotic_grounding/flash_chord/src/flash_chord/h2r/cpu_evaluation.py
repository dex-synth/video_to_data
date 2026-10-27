"""Standalone CPU evaluation on the exact common task, without a bench runtime.

MuJoCo owns persistent float64 integration state. Float32 observation/control
boundaries follow the H2R CPU protocol; observations never overwrite that state.
The controller retains native CUDA feature, actor and action arithmetic.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controller import Controller
from .inputs import content_hash, digest, member, read_json, verify_files, write_json


class CpuTask:
    """One independent MuJoCo world loaded from an immutable H2R task."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        task = self.task = read_json(self.path / "task.json")
        identity = {key: value for key, value in task.items() if key != "task_id"}
        if task["format"] != "h2r-evaluation-v1" or content_hash(identity) != task["task_id"]:
            raise ValueError("Invalid H2R evaluation task")
        verify_files(self.path, task["files"])
        if (task["observation"].get("contact_schema") != "h2r-body-contacts-v1"
                or task["initialization"]["warmup_ticks"] != 0):
            raise ValueError("CPU evaluation requires the H2R contact schema and zero warmup")
        source = member(self.path, task["scene"])
        original = mujoco.MjModel.from_xml_path(str(source))
        tree = ET.parse(source)
        for node in tree.iter():
            if "file" in node.attrib:
                node.set("file", str((source.parent / node.get("file")).resolve()))
        compiler = tree.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(tree.getroot(), "compiler")
        compiler.set("discardvisual", "true")
        sensors = tree.find("sensor")
        if sensors is None:
            sensors = ET.SubElement(tree.getroot(), "sensor")
        active = np.flatnonzero(original.geom_contype | original.geom_conaffinity)
        colliding = {original.body(int(b)).name for b in original.geom_bodyid[active]}
        self.pairs = [(a, b) for a in task["semantics"]["body_names"]
                      for b in task["object"]["parts"].values() if a in colliding and b in colliding]
        self.contact_bodies = sorted({b for pair in self.pairs for b in pair})
        for i, (a, b) in enumerate(self.pairs):
            ET.SubElement(sensors, "contact", name=f"cpu_contact_{i}", body1=a, body2=b,
                          reduce="none", num="65", data="found force dist pos normal tangent")
        for prefix, bodies, fields in (
            ("solve", self.contact_bodies, ("pos", "quat")),
            ("current", task["observation"]["bodies"], ("pos", "quat", "linvel", "angvel")),
        ):
            for body in bodies:
                for field in fields:
                    ET.SubElement(sensors, "frame" + field, name=f"cpu_{prefix}_{field}_{body}",
                                  objtype="xbody", objname=body)
        # Preserve authored physical values and ordering; no MjSpec.to_xml or
        # Newton model reconstruction belongs in this CPU evaluation path.
        self.model = m = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
        if m.opt.timestep != task["dt_s"]:
            raise ValueError("Task and scene physics timesteps differ")
        self.data, self.kinematics = mujoco.MjData(m), mujoco.MjData(m)
        self.names = task["semantics"]["joint_names"]
        self.q_ids = [int(m.joint(n).qposadr[0]) for n in self.names]
        self.dof_ids = [int(m.joint(n).dofadr[0]) for n in self.names]
        self.actuator_ids = []
        for name in self.names:
            matches = np.flatnonzero(m.actuator_trnid[:, 0] == m.joint(name).id)
            if len(matches) != 1:
                raise ValueError(f"Expected one common position actuator for {name}")
            self.actuator_ids.append(int(matches[0]))
        if set(self.actuator_ids) != set(range(m.nu)):
            raise ValueError("Common task contains non-robot actuators")
        self.object_ids = [int(m.joint(n).qposadr[0]) for n in task["object"]["joints"].values()]
        self.object_dof_ids = [int(m.joint(n).dofadr[0]) for n in task["object"]["joints"].values()]
        self.sensors = {m.sensor(i).name: slice(int(m.sensor_adr[i]), int(m.sensor_adr[i] + m.sensor_dim[i]))
                        for i in range(m.nsensor)}

    def hello(self):
        task, m = self.task, self.model
        fields = ("task_id", "bundle_id", "robot_manifest_sha256", "robot_name", "dt_s",
                  "control_ticks", "end_tick", "source_frame_range", "object", "observation")
        return {**{key: task[key] for key in fields}, "protocol": "h2r-evaluation-v1",
                "joint_names": self.names, "joint_units": task["semantics"]["joint_units"],
                "ctrl_range": m.actuator_ctrlrange[self.actuator_ids].tolist(),
                "ctrl_limited": m.actuator_ctrllimited[self.actuator_ids].astype(bool).tolist(),
                "episodes": [{"episode_id": "native-cpu"}]}

    def reset(self, initial):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q = m.keyframe("reference_initial").qpos.copy()
        if set(initial) != set(self.names) or not np.isfinite(list(initial.values())).all():
            raise ValueError("Invalid initial robot state")
        for name, value in initial.items():
            joint = m.joint(name)
            if joint.limited[0] and not joint.range[0] - 1e-6 <= value <= joint.range[1] + 1e-6:
                raise ValueError(f"Initial joint outside mechanical range: {name}")
        q[self.q_ids] = [initial[n] for n in self.names]
        root = int(m.joint(self.task["object"]["root_joint"]).qposadr[0])
        q[root:root + 7] = self.task["initialization"]["object_pose_wxyz"]
        for semantic, joint in self.task["object"]["joints"].items():
            q[int(m.joint(joint).qposadr[0])] = self.task["initialization"]["object_joints"][semantic]
        check = mujoco.MjData(m)
        check.qpos[:] = q
        mujoco.mj_forward(m, check)
        if any(c.dist < -self.task["initialization"]["max_penetration_m"] for c in check.contact):
            raise ValueError("Initial penetration exceeds the common task allowance")
        d.qpos[:] = q.astype(np.float32)
        mujoco.mj_forward(m, d)

    def step(self, targets):
        values = np.asarray([targets[n] for n in self.names], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite control")
        self.data.ctrl[self.actuator_ids] = values
        mujoco.mj_step(self.model, self.data)

    def snapshot(self, tick):
        m, d, k = self.model, self.data, self.kinematics
        if not np.isclose(d.time, tick * self.task["dt_s"], rtol=0, atol=1e-9):
            raise ValueError("CPU physics clock changed unexpectedly")
        q, dq = d.qpos.astype(np.float32), d.qvel.astype(np.float32)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("Non-finite CPU physics state")
        k.qpos[:], k.qvel[:] = q, dq
        mujoco.mj_kinematics(m, k)
        mujoco.mj_comPos(m, k)
        mujoco.mj_comVel(m, k)
        mujoco.mj_sensorPos(m, k)
        mujoco.mj_sensorVel(m, k)
        bodies = {body: np.concatenate([k.sensordata[self.sensors[f"cpu_current_{f}_{body}"]]
                                        for f in ("pos", "quat", "linvel", "angvel")]).tolist()
                  for body in self.task["observation"]["bodies"]}
        # H2R's CPU sensor bridge publishes float32 last-solve measurements.
        # Keep this conversion before local-point/force transforms, not after.
        solve = d.sensordata.astype(np.float32)
        poses = {b: (solve[self.sensors[f"cpu_solve_pos_{b}"]],
                     Rotation.from_quat(solve[self.sensors[f"cpu_solve_quat_{b}"]][[1, 2, 3, 0]]))
                 for b in self.contact_bodies}
        contacts = []
        for i, (a, b) in enumerate(self.pairs):
            rows = solve[self.sensors[f"cpu_contact_{i}"]].reshape(65, 14)
            if not np.isfinite(rows).all() or rows[-1, 0] != 0:
                raise ValueError(f"Invalid or overflowing contacts for {a}/{b}")
            for row in rows[:64]:
                if row[0] == 0:
                    continue
                force, distance, midpoint = row[1:4], row[4], row[5:8]
                normal, tangent = row[8:11], row[11:14]
                basis = np.stack((normal, tangent, np.cross(normal, tangent)))
                contacts.append({
                    "body1": a, "body2": b,
                    "point1_local": poses[a][1].inv().apply(midpoint - .5 * distance * normal - poses[a][0]).tolist(),
                    "point2_local": poses[b][1].inv().apply(midpoint + .5 * distance * normal - poses[b][0]).tolist(),
                    "force_on_body2_world": (force @ basis).tolist(),
                })
        return {"episode_id": "native-cpu", "done": tick == self.task["end_tick"],
                "tick": tick, "time_s": tick * self.task["dt_s"],
                "robot_q": q[self.q_ids].tolist(), "robot_dq": dq[self.dof_ids].tolist(),
                "bodies": bodies, "contacts": {"solve_tick": tick - 1 if tick else None, "records": contacts}}


def evaluate(task, checkpoint, prepared, output, *, metric_spec):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    world = CpuTask(task)
    spec = read_json(Path(metric_spec) / "metric_spec.json")
    identity = {key: value for key, value in spec.items() if key != "spec_id"}
    if content_hash(identity) != spec["spec_id"] or spec["task_id"] != world.task["task_id"]:
        raise ValueError("Metric specification identity or task differs")
    verify_files(Path(metric_spec), spec["files"])
    controller = Controller(checkpoint, prepared, world.hello())
    world.reset(controller.initial)
    rows, controls = [], []
    target = controller.initial
    for tick in range(world.task["end_tick"] + 1):
        frame = world.snapshot(tick)
        wrists = {}
        for side, hand in spec["hands"].items():
            pose = np.asarray(frame["bodies"][hand["wrist_body"]][:7])
            local = np.asarray(hand["wrist_frame_in_body"], dtype=float)
            rotation = Rotation.from_quat(pose[3:], scalar_first=True)
            # Apply the declared measurement frame, including quaternion
            # normalization/composition even when the local frame is identity.
            wrists[f"wrist_{side}_pose_wxyz"] = np.concatenate((
                pose[:3] + rotation.apply(local[:3]),
                (rotation * Rotation.from_quat(local[3:], scalar_first=True)).as_quat(scalar_first=True)))
        target_values = [target[n] for n in world.names]
        rows.append({"tick": tick, "time_s": frame["time_s"], "robot_q": frame["robot_q"],
                     "physics_time_s": float(np.float32(world.data.time)),
                     "robot_dq": frame["robot_dq"],
                     "object_q": world.data.qpos[world.object_ids].astype(np.float32).tolist(),
                     "object_dq": world.data.qvel[world.object_dof_ids].astype(np.float32).tolist(),
                     "object_bottom_state": frame["bodies"]["object_bottom"],
                     "object_top_state": frame["bodies"]["object_top"],
                     **wrists, "submitted_target": target_values, "force_solve_tick": tick - 1,
                     "effective_target": np.where(world.model.actuator_ctrllimited[world.actuator_ids],
                         np.clip(target_values, *world.model.actuator_ctrlrange[world.actuator_ids].T), target_values),
                     "actuator_force": world.data.actuator_force[world.actuator_ids].astype(np.float32).tolist()})
        if frame["done"]:
            break
        if tick % world.task["control_ticks"] == 0:
            command, diagnostics = controller.step([frame])
            target = command["commands"]["native-cpu"]["joint_targets"]
            controls.append({"tick": tick, "target": [target[n] for n in world.names], **diagnostics})
        world.step(target)
    np.savez_compressed(output / "physical.npz", **{key: np.asarray([r[key] for r in rows]) for key in rows[0]})
    np.savez_compressed(output / "controls.npz", **{key: np.asarray([r[key] for r in controls]) for key in controls[0]})
    write_json(output / "run.json", {"task": str(world.path), "task_id": world.task["task_id"],
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": digest(checkpoint),
        "prepared": str(Path(prepared).resolve()), "simulator": "mujoco", "mujoco_version": mujoco.__version__,
        "metric_spec": str(Path(metric_spec).resolve()), "metric_spec_id": spec["spec_id"], "hands": spec["hands"],
        "physics_steps": len(rows) - 1, "control_updates": len(controls), "complete": True,
        "scope": "Standalone method evaluation on the common task; not a bench-owned rollout."})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "checkpoint", "prepared", "output", "metric-spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
