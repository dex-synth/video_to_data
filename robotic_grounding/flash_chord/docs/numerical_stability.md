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
| Scalar joint position | Authored range extended by 50% of its width at each end |

All joint coordinates, velocities and body poses must also be finite. These are
failure bounds, not measured hardware limits or task success thresholds. Tune them
for another task using its reference and normal rollout states. Setting
`env.state_validity=null` disables the additional bounds.

A failed world terminates with zero transition reward and follows the existing
reset policy. The state itself is never clipped. The learner excludes the failed
transition and every pending n-step sample that touches it. Other worlds remain
eligible for replay. Post-reset physical state, observation and critic context
must be valid before generating another action; an invalid reset stops training.

`transition_invalid_reason` is a per-world bit mask: nonfinite=1, body position=2,
linear velocity=4, angular velocity=8, scalar joint position=16, scalar joint
velocity=32. `reset_invalid_reason` uses the same bits. Metrics named
`episode/invalid_*` count both nonfinite and out-of-bounds failures, replacing the
former `episode/nonfinite_*` names. Failed episodes remain in the denominator.

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
