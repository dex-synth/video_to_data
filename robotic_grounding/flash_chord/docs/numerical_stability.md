# FlashSAC numerical stability

The H2R recipe checks physical state before admitting a transition to replay. A
finite value alone is insufficient: a divergent simulation can produce enormous
velocities that overflow a float16 policy projection.

## Physical-state validity

`env.state_validity` enables per-world checks in `envs/validity.py`. The H2R recipe
uses these loose numerical failure bounds:

| Quantity | Bound |
| --- | --- |
| Body translation from its authored initial position | 10 m per axis |
| Body linear velocity and scalar prismatic velocity | 20 m/s per component |
| Body angular velocity and scalar revolute velocity | 200 rad/s per component |
| Scalar joint position (default: learnable termination) | Authored range extended by 50% of its width at each end |

`env.state_validity.joint_position_limit_mode` selects how scalar-joint position
violations are handled:

| Mode | Episode | Replay / learning |
| --- | --- | --- |
| `ignore` | No extra termination for joint position | No filtering for joint position |
| `discard` | Terminate as an invalid state; zero transition reward | Drop the failed transition and n-step windows containing it |
| `terminate` (default) | Ordinary failure termination, like object tracking error | Retain the terminal transition and n-step windows, with the existing termination penalty and no terminal bootstrap |

In `terminate` mode the joint-limit flag is applied **before** objective evaluation,
so both the scalar reward and the stored `termination` objective term include the
penalty. Simultaneous object and joint failures incur this term once. The default
termination weight is -100; at 20 Hz its contribution to the unnormalized step
reward is -5 because objective composition multiplies by the frame duration.
`termination/joint_position_limit_fraction_per_completed_episode` reports this
cause (after wrist/object causes in the existing exclusive-cause priority).
A simultaneous nonfinite or other invalid-state failure still discards the sample;
joint termination does not make numerically invalid data eligible for replay.

The margin remains `env.state_validity.joint_range_margin_fraction=0.5`; these
modes do not change physical joint limits or clip actuator targets. Nonfinite,
joint-speed and body-bound checks and ordinary task termination remain active.
Both `discard` and `terminate` reject an out-of-bounds **reset** state: reset is not
a policy transition and has no action to penalize. `ignore` skips that position
check too.

Checkpoints saved before this option existed always discarded joint-limit
failures. Resuming any checkpoint requires the same mode: a saved config without
the field is treated as `discard`, and a mismatch is rejected rather than silently
switching to the `terminate` default.

All joint coordinates, velocities and body poses must also be finite. These are
failure bounds, not measured hardware limits or task success thresholds. Tune them
for another task using its reference and normal rollout states. Setting
`env.state_validity=null` disables the additional bounds.

A numerically invalid world terminates with zero transition reward and follows the existing
reset policy. The state itself is never clipped. The learner excludes the failed
transition and every pending n-step sample that touches it. Other worlds remain
eligible for replay. Post-reset physical state, observation and critic context
must be valid before generating another action; an invalid reset stops training.

`transition_invalid_reason` is a per-world bit mask: nonfinite=1, body position=2,
linear velocity=4, angular velocity=8, scalar joint position=16, scalar joint
velocity=32. In `terminate` mode bit 16 is routed to ordinary termination and
removed from `transition_invalid_reason`; other invalid bits are preserved.
`reset_invalid_reason` retains the raw check bits. Metrics named
`episode/invalid_*` count both nonfinite and out-of-bounds failures, replacing the
former `episode/nonfinite_*` names. Failed episodes remain in the denominator.
Joint-only failures in `terminate` mode count as ordinary terminated episodes,
not invalid episodes.

## Network features and update transactions

`training.network.normalized_feature_limit=100` clips each BatchNorm output,
after normalization and affine transformation, to [-100, 100]. Raw observations
and physical state retain their units. Training and checkpoint-driven inference
construct the same network from the saved configuration. Old configurations that
omit this field retain the previous unclipped behavior (`None`). Clipping does
not repair NaNs or guarantee finite computation for arbitrary corrupt weights.

Each optimizer attempt commits actor, critic, target critic, temperature,
BatchNorm statistics, Adam moments and scheduler clocks together. A skipped
gradient update or nonfinite candidate state/metric rolls back the entire attempt.
The RNG and attempt counter advance so training can sample a different batch.
One rejected attempt backs off the previous loss scale once; accepted attempts
retain the existing per-update growth behavior. `loss_scale_min` defaults to 1,
preventing repeated backoff from reaching zero.

At logging boundaries the runner checks the compact training state and actual
Adam update counts. `optimization.max_failed_updates` defaults to 1024: if a
critic makes no progress over that many attempts, or an actor over that many
actor periods, training saves `failure/state.safetensors` and `failure/reason.json`
and raises an error. Detection can lag by a logging interval. A nonfinite initial
or running learner state also stops training. These diagnostic files are not
published as usable policy checkpoints. Normal checkpoint publication validates
the whole compact learner, even if its actor alone is finite.

These protections contain numerical failures; they do not fix the underlying
contact or solver instability. Short training checks establish numerical
behavior, not final task success or benchmark performance.
