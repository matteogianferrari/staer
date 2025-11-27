import torch
import torchvision.transforms as transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss


class SEr(ContinualModel):
    """Continual learning via Experience Replay for SNN."""
    NAME = 'ser'
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

        parser.add_argument('--temp_sep', type=int, default=1, required=True,
                            help='Applies temporal separation to the CE [0 for CE or 1 for TSCE].')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The SER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(SEr, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.T = args.T
        self.temp_sep = args.temp_sep

        if self.temp_sep:
            self.ce_loss = TSCELoss()
        else:
            self.ce_loss = CELoss()

        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # print(f"buf_input.shape: {buf_inputs.shape}")
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
        inputs = inputs.transpose(0, 1).contiguous()

        outputs = self.net(inputs)

        if self.temp_sep:
            tsce_loss = self.ce_loss(outputs, labels)
            loss = tsce_loss
        else:
            ce_loss = self.ce_loss(outputs, labels)
            loss = ce_loss

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        return loss.item()
