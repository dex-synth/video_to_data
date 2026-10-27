# V2D and h2r_bench input boundary

This FlashCHORD fork trains only from sealed h2r_bench inputs. The public entry is
`task.parquet=/path/to/prepared/h2r_binding.json`, using the sole `h2r_sharpa_flash_sac` recipe.
See [the H2R workflow](docs/h2r_inputs.md) for preparation and execution.

The task reference, object/support scene and complete robot come from h2r_bench. The method's
`h2r.prepare` uses V2D MANO/contact and Sharpa IK components to derive its private reference.
That reference is stored as `processed.parquet` inside the preparation package and is covered
by the binding's file hashes. It is not a standalone public training input.

Low-level ManoSharpa and motion_v1 loaders, semantic binding/contact helpers and algorithm modules remain in
this source tree for internal reuse and unit tests. Their presence does not declare native
Sharpa, Dexmate or G1 training support. The native experiment recipes and PPO training entry
have been removed; training, resume, diagnostic evaluation and policy viewing reject native
inputs and checkpoints before constructing an environment.

FlashCHORD's vendored Sharpa, Vega/Dexmate and G1 URDFs/meshes, adapters and SONIC assets are removed.
The sibling V2D assets remain available to preparation through its independent loader/IK
environments: `h2r/assets.py` reads Sharpa MJCF site orientations, then attaches those IK objectives
to the common hand subtree and uses common fingertip positions. These templates are required for
preparation and are not loaded as the physical robot. H2R physics uses the copied common robot and scene assets, not a same-named
native asset selected from that tree. Historical reference corpora and checkpoints are not
rewritten or migrated. Formal physical evaluation remains owned by h2r_bench.
