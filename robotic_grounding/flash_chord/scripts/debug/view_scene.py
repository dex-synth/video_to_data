# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display the complete sealed H2R robot/object/support scene at a reference frame."""

from flash_chord.h2r.robot import H2rSharpaHands
from flash_chord.lifecycle.reset import reset_scene_to_frame
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.visualization.viewer import ViewerApp


class ViewScene(ViewerApp):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        setup = setup_scene(
            parquet=args.binding,
            control_fps=30,
            motion_speed=1.0,
            embodiment=H2rSharpaHands(args.binding),
            collision=CollisionPolicy(ground=False),
            world_count=1,
            include_support=True,
            decompose_objects=False,
        )
        if not 0 <= args.frame < setup.reference.num_frames:
            raise ValueError(f"frame must be in [0, {setup.reference.num_frames})")
        self.model = setup.scene.model
        self.state = self.model.state()
        reset_scene_to_frame(setup.scene, setup.reference, args.frame, self.state)
        self.viewer.set_model(self.model)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--binding", required=True, help="prepared h2r_binding.json")
        parser.add_argument("--frame", type=int, default=0, help="reference frame to display")


if __name__ == "__main__":
    ViewScene.launch()
