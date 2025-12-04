import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss, KLLoss
from models.spiking_er.divergence import SoftDTWDivergence


class Taer(ContinualModel):
    """Temporal Alignment with Experience Replay for Continual learning in Spiking Neural Networks (TAER)."""
    NAME = 'taer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the TSAER model.

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

        parser.add_argument('--tau', type=int, default=5,
                            help='Hyperparameter that regulates the temperature of the softmax in KL.')

        parser.add_argument(
            '--alpha', type=float, default=1e-2,
            help='Hyperparameter that balances the KL term of the loss.'
        )

        parser.add_argument(
            '--beta', type=float, default=1e-3,
            help='Hyperparameter that balances the Soft-DTW term of the loss.'
        )

        parser.add_argument(
            '--sdtw_gamma', type=float, default=1.0,
            help='Native Hyperparameter of Soft-DTW.'
        )

        parser.add_argument(
            '--sdtw_norm', type=int, default=1,
            help='Native Hyperparameter of Soft-DTW on normalization.'
        )

        # 'halve' makes the temporal dimension for the Soft-DTW past logits the half of the current T, requires T>=2
        # 'same' makes the temporal dimension for the Soft-DTW past logits the same of the current T
        # 'double' makes the temporal dimension for the Soft-DTW past logits double the current T
        parser.add_argument(
            '--sdtw_T', type=str, default='same',
            help='Temporal dimension for the past logits for alignment. Select between [halve, same, double].'
        )

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The STAER model maintains a buffer of previously seen examples and
        uses them to align the past logits (viewed as time-series) with the current logits, using the Soft-DTW.
        """
        super(Taer, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Soft-DTW temporal factors
        sdtw_factors = {'halve': 0.5, 'same': 1.0, 'double': 2.0}

        # Model's specific args
        # Temporal dimension
        self.T = args.T
        self.sdtw_T = int(round(self.T * sdtw_factors[args.sdtw_T]))

        # Temporal separation
        self.temp_sep = args.temp_sep

        # TSKL
        self.tau = args.tau
        self.alpha = args.alpha

        # Soft-DTW
        self.beta = args.beta
        self.sdtw_gamma = args.sdtw_gamma
        self.sdtw_norm = args.sdtw_norm

        # Creates the loss with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            self.ce_loss = TSCELoss()
        else:
            self.ce_loss = CELoss()

        # Creates the Soft-DTW loss
        self.sdtw_loss = SoftDTWDivergence(gamma=self.sdtw_gamma, normalize=self.sdtw_norm)

        # Creates the TSKL loss
        self.kl_loss = KLLoss(tau=self.tau)

        # Creates the buffer and its transforms
        self.kl_buffer = Buffer(self.args.buffer_size)
        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

        # Creates the buffer for Soft-DTW logits
        self.sdtw_buffer = Buffer(self.args.buffer_size)

    def _build_sdtw_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Interpolate 'logits' along the temporal dimension from T to sdtw_T.
        Works for logits shaped either [T, B, K].
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits, got shape {logits.shape}")

        # 'same' case
        if self.sdtw_T == self.T:
            return logits
        else:
            # Changes the order from [T, B, K] to [B, K, T]
            x = logits.permute(1, 2, 0)
            x = F.interpolate(x, size=self.sdtw_T, mode="linear", align_corners=False)

            # Changes the order from [B, K, T] to [T, B, K]
            return x.permute(2, 0, 1)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        STAER trains on the current task using the data provided, but also aligns the current logits with the
        past logits using the Soft-DTW term.
        """
        self.opt.zero_grad()

        B = inputs.shape[0]
        sdtw_inputs = inputs

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        inputs = inputs.transpose(0, 1).contiguous()

        # The model processes the data
        tser_logits = self.net(inputs)

        # CE loss computation with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            tsce_loss_raw = self.ce_loss(tser_logits, labels)
            tsce_loss = (1 - self.alpha) * tsce_loss_raw
        else:
            ce_loss_raw = self.ce_loss(tser_logits, labels)
            ce_loss = (1 - self.alpha) * ce_loss_raw

        # Temporal separation with KL
        kl_loss_raw = 0
        kl_loss = 0
        if not self.kl_buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of past examples
            # and their train of logits, output of the past SNN (old parameters related to the past task)
            # Applies the transforms to the data
            past_inputs, past_logits_past_inputs = self.kl_buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )

            # The past examples are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
            past_inputs = past_inputs.transpose(0, 1).contiguous()

            # The
            past_logits_past_inputs = past_logits_past_inputs.transpose(0, 1).contiguous()

            # The current model processes the past examples
            curr_logits_past_inputs = self.net(past_inputs)

            # KL loss computation between past logits (computed by SNN with previous parameters) related to the past
            # examples, and the current logits (computed by current SNN parameters) related to the past examples
            kl_loss_raw = self.kl_loss(t_logits=past_logits_past_inputs, s_logits=curr_logits_past_inputs)
            kl_loss = self.alpha * (self.tau ** 2) * kl_loss_raw

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        sdtw_inputs = sdtw_inputs.transpose(0, 1).contiguous()

        # Creates the logits for the Soft-DTW by interpolating the original temporal dimension
        sdtw_logits = self.net(sdtw_inputs)
        sdtw_logits = self._build_sdtw_logits(sdtw_logits)

        # Temporal alignment with Soft-DTW
        sdtw_loss_raw = 0
        sdtw_loss = 0
        if not self.sdtw_buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of logits
            _, past_sdtw_logits = self.sdtw_buffer.get_data(self.args.minibatch_size, device=self.device)

            # The current logits are transposed to the shape [B, T, K] for Soft-DTW compatibility
            sdtw_logits = sdtw_logits.transpose(0, 1).contiguous()

            if sdtw_logits.shape[0] == past_sdtw_logits.shape[0]:
                # Soft-DTW loss computation between past logits and current logits
                sdtw_loss_raw = self.sdtw_loss(sdtw_logits, past_sdtw_logits)
                sdtw_loss = self.beta * sdtw_loss_raw

            # The current logits are transposed back to the shape [T, B, K]
            sdtw_logits = sdtw_logits.transpose(0, 1).contiguous()

        if self.temp_sep:
            loss = tsce_loss + kl_loss + sdtw_loss
        else:
            loss = ce_loss + kl_loss + sdtw_loss

        # Backprop
        loss.backward()
        self.opt.step()

        # Adds to the KL buffer the current non-augmented data and their logits
        self.kl_buffer.add_data(examples=not_aug_inputs, logits=tser_logits.data)

        # Adds to the Soft-DTW buffer the current non-augmented data and their logits
        self.sdtw_buffer.add_data(examples=not_aug_inputs, logits=sdtw_logits.data)

        return loss.item()
