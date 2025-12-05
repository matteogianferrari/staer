import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss
from models.spiking_er.divergence import SoftDTWDivergence


class Staer(ContinualModel):
    """Spiking Neural Networks with Temporal Alignment and Experience Replay for Continual learning (STAER)."""
    NAME = 'staer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the STAER model.

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

        parser.add_argument(
            '--beta', type=float, default=1e-2,
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
        uses them to align the past outputs (viewed as time-series, the output is generated with the old parameters
        of the SNN w.r.t. the current training process) with the outputs of the SNN (viewed as time-series, the output
        is generated using the same examples in the buffer, but using the updated parameters of the current SNN),
        using the Soft-DTW.
        """
        super(Staer, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Soft-DTW temporal factors
        sdtw_factors = {'halve': 0.5, 'same': 1.0, 'double': 2.0}

        # Model's specific args
        # Temporal dimension
        self.T = args.T
        self.sdtw_T = int(round(self.T * sdtw_factors[args.sdtw_T]))

        # Temporal separation
        self.temp_sep = args.temp_sep

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

        # Creates the buffer and its transforms
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_transform = transforms.Compose([
            dataset.get_transform(),
            StaticEncoding(T=self.T)
        ])

        # Creates the buffer for Soft-DTW logits
        self.sdtw_buffer = Buffer(self.args.buffer_size)

    def _build_sdtw_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Interpolate 'outputs' along the temporal dimension from T to sdtw_T.
        Works for logits shaped either [T, B, K].
        """
        if outputs.dim() != 3:
            raise ValueError(f"Expected 3D outputs, got shape {outputs.shape}")

        # 'same' case
        if self.sdtw_T == self.T:
            return outputs
        else:
            # Changes the order from [T, B, K] to [B, K, T]
            x = outputs.permute(1, 2, 0)
            x = F.interpolate(x, size=self.sdtw_T, mode="linear", align_corners=False)

            # Changes the order from [B, K, T] to [T, B, K]
            return x.permute(2, 0, 1)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        STAER trains on the current task using the data provided, but also aligns the output of the examples in the
        buffer, using the current SNN, with the past outputs of the examples, generated using the old parameters
        of the SNN, using the Soft-DTW term.
        """
        self.opt.zero_grad()

        B = inputs.shape[0]
        sdtw_inputs = inputs
        sdtw_labels = labels

        # SER
        if not self.buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of data and their labels
            # Applies the transforms to the data
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # Augments the current batch with the mini-batch of data from the buffer
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        inputs = inputs.transpose(0, 1).contiguous()

        # The model processes the data
        outputs = self.net(inputs)

        # CE loss computation with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            tsce_loss_raw = self.ce_loss(outputs, labels)
        else:
            ce_loss_raw = self.ce_loss(outputs, labels)

        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        sdtw_inputs = sdtw_inputs.transpose(0, 1).contiguous()

        # Creates the outputs for the Soft-DTW by interpolating the original temporal dimension
        # sdtw_outputs.shape: [B, T, K]
        sdtw_outputs = self.net(sdtw_inputs)
        sdtw_outputs = self._build_sdtw_logits(sdtw_outputs)

        # Temporal alignment with Soft-DTW
        sdtw_loss_raw = 0
        sdtw_loss = 0
        if not self.sdtw_buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of logits
            sdtw_buf_inputs, past_sdtw_outputs = self.sdtw_buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # The buffer examples are transposed to the shape [T, B, C, H, W] for compatibility
            sdtw_buf_inputs = sdtw_buf_inputs.transpose(0, 1).contiguous()

            # Creates the outputs for the buffer examples using the updated parameters of the SNN
            sdtw_buf_outputs = self.net(sdtw_buf_inputs)
            sdtw_buf_outputs = self._build_sdtw_logits(sdtw_buf_outputs)
            sdtw_buf_outputs = sdtw_buf_outputs.transpose(0, 1).contiguous()

            # Soft-DTW loss computation between past logits and current logits
            sdtw_loss_raw = self.sdtw_loss(sdtw_buf_outputs, past_sdtw_outputs)
            sdtw_loss = self.beta * sdtw_loss_raw

        if self.temp_sep:
            loss = tsce_loss_raw + sdtw_loss
        else:
            loss = ce_loss_raw + sdtw_loss

        # Backprop
        loss.backward()
        self.opt.step()

        # Adds to the buffer the current non-augmented data and their labels
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:B])

        # Adds to the buffer the current non-augmented data and their logits for Soft-DTW
        self.sdtw_buffer.add_data(examples=not_aug_inputs, logits=sdtw_outputs.data)

        return loss.item()
