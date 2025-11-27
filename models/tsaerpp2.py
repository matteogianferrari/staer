import torch
import torchvision.transforms as transforms
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import CELoss
from models.spiking_er.soft_dtw import SoftDTW


class Tsaerpp2(ContinualModel):
    """Continual learning via Experience Replay for SNN with temporal alignment."""
    NAME = 'tsaerpp2'
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

        parser.add_argument('--beta', type=float, default=1e-4,
                            help='Hyperparameter that balances the SDTW term of the loss.')

        parser.add_argument('--sdtw_gamma', type=float, default=1.0,
                            help='Native Hyperparameter of SDTW.')

        parser.add_argument('--sdtw_norm', type=bool, default=True,
                            help='Native Hyperparameter of SDTW on data normalization.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The TSER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Tsaerpp2, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.brain_buffer = Buffer(self.args.buffer_size)

        self.T = args.T
        self.tau = args.tau
        self.beta = args.beta
        self.sdtw_gamma = args.sdtw_gamma
        self.sdtw_norm = args.sdtw_norm
        self.theta = args.theta

        self.ce_loss = CELoss()
        self.sdtw_loss = SoftDTW(gamma=self.sdtw_gamma, normalize=self.sdtw_norm)

        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

        self.brain_buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T*2)
        ])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        TSER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        self.opt.zero_grad()

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
        inputs = inputs.transpose(0, 1).contiguous()

        # The model processes the inputs, s_logits.shape: [T, B, K]
        s_logits = self.net(inputs)

        # The input is replicated to create the logits over 2*T, brain_logits.shape: [2*T, B, K]
        brain_input = inputs.repeat_interleave(2, dim=0)
        brain_logits = self.net(brain_input)

        # Computes the CE (can be always computed, even when the buffer is empty)
        loss_ce_raw = self.ce_loss(s_logits=s_logits, targets=labels)

        if not self.buffer.is_empty():
            # Retrieves a batch of inputs and their logits from the buffer
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )

            # The buffer inputs are transposed to the shape [T, B, C, H, W] for compatibility
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()

            # The model processes the buffer inputs, buf_outputs.shape: [T, B, K]
            buf_outputs = self.net(buf_inputs)

            # Retrieves a batch of extended logits from the special buffer
            _, _, brain_buf_logits = self.brain_buffer.get_data(
                self.args.minibatch_size, transform=self.brain_buffer_transform, device=self.device
            )

            # The model output is transposed to the shape [B, T, K]
            buf_outputs = buf_outputs.transpose(0, 1).contiguous()

            # Computes the Soft-DTW loss between [B, T, K] and [B, 2*T, K], and applies the scaling factor
            loss_sdtw_raw = self.sdtw_loss(buf_outputs, brain_buf_logits).mean(dim=0)
            loss_sdtw = self.beta * loss_sdtw_raw

            # Computes the total loss
            loss = loss_ce_raw + loss_sdtw
        else:
            loss = loss_ce_raw

        loss.backward()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=s_logits.data)

        # T interpolation
        self.brain_buffer.add_data(examples=not_aug_inputs, labels=labels, logits=brain_logits.data)

        return loss.item()
