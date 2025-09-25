import torch
import torchvision.transforms as transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, TSKLLoss, EntropyReg, TSMSELoss


class Tser(ContinualModel):
    """Continual learning via Experience Replay for SNN."""
    NAME = 'tser'
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

        parser.add_argument('--tau', type=int, default=6,
                            help='Hyperparameter that regulates the temperature of the softmax.')

        parser.add_argument('--alpha', type=float, default=1e-1,
                            help='Hyperparameter that balances the 2 terms of the loss.')

        parser.add_argument('--gamma', type=float, default=1e-3,
                            help='Hyperparameter that weights the regularization term.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The TSER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Tser, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

        self.T = args.T
        self.tau = args.tau
        self.alpha = args.alpha
        self.gamma = args.gamma

        self.tsce_loss = TSCELoss(tau=self.tau)
        self.tskl_loss = TSKLLoss(tau=self.tau)
        self.e_reg = EntropyReg(tau=self.tau)
        self.mse_loss = TSMSELoss()
        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        TSER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        self.opt.zero_grad()

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
        inputs = inputs.transpose(0, 1).contiguous()
        # print(f"inputs shape: {inputs.shape}")

        s_logits = self.net(inputs)
        # print(f"s_logits.shape: {s_logits.shape}")

        # TSCE loss
        # Can be always computed, even when the buffer is empty
        loss = (1 - self.alpha) * self.tsce_loss(s_logits=s_logits, targets=labels)
        # loss = self.tsce_loss(s_logits=s_logits, targets=labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )

            # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()
            buf_logits = buf_logits.transpose(0, 1).contiguous()

            buf_outputs = self.net(buf_inputs)
            # print(f"buf_outputs.shape: {buf_outputs.shape}")
            # TSKL loss
            # Can be computed only when the buffer is not empty
            loss_tskl = self.alpha * (self.tau ** 2) * self.tskl_loss(t_logits=buf_logits, s_logits=buf_outputs)
            loss += loss_tskl

            # TSMSE loss
            # loss_tsmse = self.alpha * self.mse_loss(t_logits=buf_logits, s_logits=buf_outputs)
            # loss += loss_tsmse

        # ER
        # Can be always computed, even when the buffer is empty
        loss_er = self.gamma * self.e_reg(s_logits=s_logits)
        loss -= loss_er

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, logits=s_logits.data)

        return loss.item()
