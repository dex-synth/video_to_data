# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display the common robot from a sealed H2R preparation package."""

import newton

from flash_chord.h2r.builder import CommonModelBuilder
from flash_chord.h2r.robot import H2rSharpaHands
from flash_chord.visualization.viewer import ViewerApp


class ViewRobot(ViewerApp):
    """Show the common robot at its default joint pose."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        embodiment = H2rSharpaHands(args.binding)
        builder = CommonModelBuilder(gravity=-9.81)
        embodiment.build(builder)
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.set_model(self.model)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--binding", required=True, help="prepared h2r_binding.json")


if __name__ == "__main__":
    ViewRobot.launch()
