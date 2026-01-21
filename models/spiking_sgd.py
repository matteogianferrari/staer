"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from models.spiking_losses.losses import TSCELoss, CELoss


class SpikingSgd(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'spiking-sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the SER model.
        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)

        parser.add_argument('--T', type=int, default=2, required=True,
                            help='Time steps for SNNs. Select between [1, 2, 4].')

        parser.add_argument(
            '--temp_sep', type=int, default=1, required=True,
            help='Applies temporal separation to the CE [0 for CE or 1 for TSCE].'
        )

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(SpikingSgd, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Creates the loss with or without temporal separation based on the 'temp_sep' arg
        if args.temp_sep:
            self.ce_loss = TSCELoss()
        else:
            self.ce_loss = CELoss()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        # inputs.shape: [T, B, C, H, W] or [T, 2B, C, H, W]
        inputs = inputs.transpose(0, 1).contiguous()

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.ce_loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
