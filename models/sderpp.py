import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, SdtwBuffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_losses.losses import TSCELoss, CELoss, MSELoss


class SDer(ContinualModel):
    """Spiking version of Continual learning via Dark Experience Replay."""
    NAME = 'sderpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument(
            '--T', type=int, default=2, required=True,
            help='Temporal dimension for the SNN. Select between [1, 2, 4].'
        )

        parser.add_argument(
            '--temp_sep', type=int, default=1, required=True,
            help='Applies temporal separation to the CE [0 for CE or 1 for TSCE].'
        )

        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')

        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(SDer, self).__init__(backbone, loss, args, transform, dataset=dataset)

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

        self.mse_loss = MSELoss()

        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        # inputs.shape: [B, T, C, H, W]

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        # inputs.shape: [T, B, C, H, W]
        inputs = inputs.transpose(0, 1).contiguous()

        # The model processes the data
        # outputs.shape: [T, B, K]
        outputs = self.net(inputs)

        # CE loss computation with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            tsce_loss_raw = self.ce_loss(outputs, labels)
        else:
            ce_loss_raw = self.ce_loss(outputs, labels)

        mse_loss_raw = 0
        mse_loss = 0
        tsce_buf_loss = 0
        ce_buf_loss = 0
        if not self.buffer.is_empty():
            # buf_inputs.shape: [B, T, C, H, W]
            # buf_logits.shape: [B, T, K]
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # The buffer inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
            # buf_inputs.shape: [T, B, C, H, W]
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()

            # The buffer logits are transposed to the shape [T, B, K] for compatibility with SNN model
            # buf_logits.shape: [T, B, K]
            buf_logits = buf_logits.transpose(0, 1).contiguous()

            # Creates the outputs for the buffer examples
            # buf_outputs.shape: [T, B, K]
            buf_outputs = self.net(buf_inputs)

            # Computes the mse
            mse_loss_raw = self.mse_loss(buf_outputs, buf_logits)
            mse_loss = self.args.alpha * mse_loss_raw

            # buf_inputs.shape: [B, T, C, H, W]
            # buf_logits.shape: [B]
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # The buffer inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
            # buf_inputs.shape: [T, B, C, H, W]
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()

            # Creates the outputs for the buffer examples
            # buf_outputs.shape: [T, B, K]
            buf_outputs = self.net(buf_inputs)

            # CE loss computation with or without temporal separation based on the 'temp_sep' arg
            if self.temp_sep:
                tsce_buf_loss_raw = self.ce_loss(buf_outputs, buf_labels)
                tsce_buf_loss = self.args.beta * tsce_buf_loss_raw
            else:
                ce_buf_loss_raw = self.ce_loss(buf_outputs, buf_labels)
                ce_buf_loss = self.args.beta * ce_buf_loss_raw

        if self.temp_sep:
            loss = tsce_loss_raw + mse_loss + tsce_buf_loss
        else:
            loss = ce_loss_raw + mse_loss + ce_buf_loss

        # Backprop
        loss.backward()
        self.opt.step()

        # To add the outputs of the network of shape [T, B, K] to the buffer, its shape must be transposed to
        # [B, T, K] for compatibility with Mammoth
        outputs = outputs.transpose(0, 1).contiguous()

        # Adds to the buffer the current non-augmented data and their outputs
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item()
