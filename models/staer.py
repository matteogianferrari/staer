import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss
# from models.spiking_er.soft_dtw import SoftDTW
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
        uses them to align the past logits (viewed as time-series) with the current logits, using the Soft-DTW.
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
        # self.sdtw_loss = SoftDTW(gamma=self.sdtw_gamma, normalize=self.sdtw_norm)
        self.sdtw_loss = SoftDTWDivergence(gamma=self.sdtw_gamma, normalize=self.sdtw_norm)

        # Creates the buffer and its transforms
        self.buffer = Buffer(self.args.buffer_size)
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
        # with open("err_log.txt", "a", encoding="utf-8") as f:

        self.opt.zero_grad()

        B = inputs.shape[0]
        sdtw_inputs = inputs

        # SER
        if not self.buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of data and their labels
            # Applies the transforms to the data
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)

            # Augments the current batch with the mini-batch of data from the buffer
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # f.write(f"inputs_pre:{inputs.shape}\n")
        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        inputs = inputs.transpose(0, 1).contiguous()
        # f.write(f"inputs:{inputs.shape}\n")

        # The model processes the data
        ser_logits = self.net(inputs)
        # f.write(f"ser_logits:{ser_logits.shape}\n")

        # CE loss computation with or without temporal separation based on the 'temp_sep' arg
        if self.temp_sep:
            tsce_loss_raw = self.ce_loss(ser_logits, labels)
        else:
            ce_loss_raw = self.ce_loss(ser_logits, labels)

        # f.write(f"sdtw_inputs_pre:{sdtw_inputs.shape}\n")
        # The inputs are transposed to the shape [T, B, C, H, W] for compatibility with SNN model
        sdtw_inputs = sdtw_inputs.transpose(0, 1).contiguous()
        # f.write(f"sdtw_inputs:{sdtw_inputs.shape}\n")

        # Creates the logits for the Soft-DTW by interpolating the original temporal dimension
        sdtw_logits = self.net(sdtw_inputs)
        # f.write(f"sdtw_logits_pre:{sdtw_logits.shape}\n")
        sdtw_logits = self._build_sdtw_logits(sdtw_logits)
        # f.write(f"sdtw_logits:{sdtw_logits.shape}\n")

        # Temporal alignment with Soft-DTW
        sdtw_loss_raw = 0
        sdtw_loss = 0
        if not self.sdtw_buffer.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of logits
            _, past_sdtw_logits = self.sdtw_buffer.get_data(self.args.minibatch_size, device=self.device)

            # f.write(f"sdtw_logits_pre:{sdtw_logits.shape}\n")
            # The current logits are transposed to the shape [B, T, K] for Soft-DTW compatibility
            sdtw_logits = sdtw_logits.transpose(0, 1).contiguous()
            # f.write(f"sdtw_logits:{sdtw_logits.shape}\n")
            # f.write(f"past_sdtw_logits:{past_sdtw_logits.shape}\n")

            if sdtw_logits.shape[0] == past_sdtw_logits.shape[0]:
                # Soft-DTW loss computation between past logits and current logits
                # sdtw_loss_raw = self.sdtw_loss(sdtw_logits, past_sdtw_logits).mean(dim=0)
                sdtw_loss_raw = self.sdtw_loss(sdtw_logits, past_sdtw_logits)
                sdtw_loss = self.beta * sdtw_loss_raw

            # f.write(f"sdtw_logits_pre:{sdtw_logits.shape}\n")
            # The current logits are transposed back to the shape [T, B, K]
            sdtw_logits = sdtw_logits.transpose(0, 1).contiguous()
            # f.write(f"sdtw_logits:{sdtw_logits.shape}\n")

        if self.temp_sep:
            loss = tsce_loss_raw + sdtw_loss
        else:
            loss = ce_loss_raw + sdtw_loss

        # Backprop
        loss.backward()
        self.opt.step()

        # loss.backward(retain_graph=True)
        # if not self.buffer.is_empty() and not self.sdtw_buffer.is_empty():
        #     shared_params = [p for p in self.net.parameters() if p.requires_grad]
        #
        #     # loss1 gradients
        #     if self.temp_sep:
        #         g1 = torch.autograd.grad(tsce_loss_raw, shared_params, retain_graph=True, create_graph=False)
        #     else:
        #         g1 = torch.autograd.grad(ce_loss_raw, shared_params, retain_graph=True, create_graph=False)
        #
        #     # loss2 gradients
        #     g2 = torch.autograd.grad(ce_loss_raw, shared_params, retain_graph=True, create_graph=False)
        #
        #     # Compute gradient norm for each loss term (over all shared params)
        #     g1_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g1))).item()
        #     g2_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g2))).item()
        #
        #     with open("grad_norm_staer.txt", "a", encoding="utf-8") as f:
        #         f.write(f"{g1_norm};{g2_norm}\n")

        # Adds to the buffer the current non-augmented data and their labels
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:B])

        # Adds to the buffer the current non-augmented data and their logits for Soft-DTW
        self.sdtw_buffer.add_data(examples=not_aug_inputs, logits=sdtw_logits.data)

        return loss.item()
