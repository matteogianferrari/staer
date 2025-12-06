import torch
import torchvision.transforms as transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss


class Ser(ContinualModel):
    """Spiking Neural Networks with Experience Replay for Continual learning (SER)."""
    NAME = 'ser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the SER model.

        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)

        parser.add_argument(
            '--T', type=int, default=2, required=True,
            help='Temporal dimension for the SNN. Select between [1, 2, 4].'
        )

        parser.add_argument(
            '--temp_sep', type=int, default=1, required=True,
            help='Applies temporal separation to the CE [0 for CE or 1 for TSCE].'
        )

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The SER model maintains a buffer of previously seen examples and uses
        them to augment the current batch during training.
        """
        super(Ser, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Model's specific args
        # Temporal dimension
        self.T = args.T

        # Temporal separation
        self.temp_sep = args.temp_sep

        # Creates the loss with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            self.ce_loss = TSCELoss()
        else:
            self.ce_loss = CELoss()

        # Creates the buffer and its transforms
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        self.opt.zero_grad()

        # not_aug_inputs.shape: [B, C, H, W]
        # inputs.shape: [B, T, C, H, W]
        # labels.shape: [B]
        B = inputs.shape[0]

        # SER
        if not self.buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of data and their labels
            # Applies the transforms to the data
            # buf_inputs.shape: [B, T, C, H, W]
            # buf_labels.shape: [B]
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # Augments the current batch with the mini-batch of data from the buffer
            # inputs.shape: [2B, T, C, H, W]
            inputs = torch.cat((inputs, buf_inputs))
            # labels.shape: [2B]
            labels = torch.cat((labels, buf_labels))

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        # inputs.shape: [T, B, C, H, W] or [T, 2B, C, H, W]
        inputs = inputs.transpose(0, 1).contiguous()

        # The model processes the data
        # outputs.shape: [T, B, K] or [T, 2B, K]
        outputs = self.net(inputs)

        # CE loss computation with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            tsce_loss_raw = self.ce_loss(outputs, labels)
            loss = tsce_loss_raw
        else:
            ce_loss_raw = self.ce_loss(outputs, labels)
            loss = ce_loss_raw

        # Backprop
        loss.backward()
        self.opt.step()

        # Adds to the buffer the current non-augmented data and their labels
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:B])

        return loss.item()
