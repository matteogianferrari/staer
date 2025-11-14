import torch
import torchvision.transforms as transforms
from utils import binary_to_boolean_type
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import CELoss, TSKLLoss
from models.spiking_er.soft_dtw import SoftDTW


class Tserta(ContinualModel):
    """Continual learning via Experience Replay for SNN with temporal alignment."""
    NAME = 'tserta'
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

        parser.add_argument('--tau', type=int, default=5,
                            help='Hyperparameter that regulates the temperature of the softmax in the KL.')

        parser.add_argument('--alpha', type=float, default=0.14,
                            help='Hyperparameter that balances the KL term of the loss.')

        parser.add_argument('--beta', type=float, default=1e-4,
                            help='Hyperparameter that balances the SDTW term of the loss.')

        parser.add_argument('--sdtw_gamma', type=float, default=1.0,
                            help='Native Hyperparameter of SDTW.')

        parser.add_argument('--sdtw_norm', type=binary_to_boolean_type, default=1,
                            help='Native Hyperparameter of SDTW on data normalization.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The TSER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Tserta, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

        self.T = args.T
        self.tau = args.tau
        self.alpha = args.alpha
        self.beta = args.beta
        self.sdtw_gamma = args.sdtw_gamma
        self.sdtw_norm = args.sdtw_norm

        self.ce_loss = CELoss()
        self.tskl_loss = TSKLLoss(tau=self.tau)
        self.sdtw_loss = SoftDTW(gamma=self.sdtw_gamma, normalize=self.sdtw_norm)

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

        # CE loss
        # Can be always computed, even when the buffer is empty
        loss_ce_raw = self.ce_loss(s_logits=s_logits, targets=labels)
        loss_ce = (1 - self.alpha) * loss_ce_raw
        loss = loss_ce

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )
            
            # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()
            # The logits are transposed to the shape [T, B, K]
            buf_logits = buf_logits.transpose(0, 1).contiguous()

            buf_outputs = self.net(buf_inputs)
            # print(f"buf shape: {buf_outputs.shape}")

            # TSKL loss
            # Can be computed only when the buffer is not empty
            loss_tskl_raw = self.tskl_loss(t_logits=buf_logits, s_logits=buf_outputs)
            loss_tskl = self.alpha * (self.tau ** 2) * loss_tskl_raw
            loss += loss_tskl

            # Re-transpose to the shape [B, T, K] for compatibility with sdtw
            buf_logits = buf_logits.transpose(0, 1).contiguous()
            # The logits are transposed to the shape [B, T, K]
            buf_outputs = buf_outputs.transpose(0, 1).contiguous()

            # SDTW loss
            loss_sdtw_raw = self.sdtw_loss(buf_outputs, buf_logits).sum()
            loss_sdtw = self.beta * loss_sdtw_raw
            loss += loss_sdtw

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, logits=s_logits.data)

        return loss.item()
