import random
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, CELoss
from models.spiking_er.divergence import SoftDTWDivergence


class Staer2(ContinualModel):
    """Spiking Neural Networks with Temporal Alignment and Experience Replay for Continual learning (STAER)."""
    NAME = 'staer2'
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

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The STAER model maintains a buffer of previously seen examples and
        uses them to align the past outputs (viewed as time-series, the output is generated with the old parameters
        of the SNN w.r.t. the current training process) with the outputs of the SNN (viewed as time-series, the output
        is generated using the same examples in the buffer, but using the updated parameters of the current SNN),
        using the Soft-DTW.
        """
        super(Staer2, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Model's specific args
        # Temporal dimension
        self.T = args.T

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

        # Creates the buffers for Soft-DTW outputs, one with halve the temporal dimension, one with the same, and
        # one with double the temporal dimension
        self.sdtw_buffer1 = Buffer(self.args.buffer_size)
        self.sdtw_buffer2 = Buffer(self.args.buffer_size)
        self.sdtw_buffer3 = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        STAER trains on the current task using the data provided, but also aligns the output of the examples in the
        buffer, using the current SNN, with the past outputs of the examples, generated using the old parameters
        of the SNN, using the Soft-DTW term.
        """
        self.opt.zero_grad()

        # not_aug_inputs.shape: [B, C, H, W]
        # inputs.shape: [B, T, C, H, W]
        # labels.shape: [B]
        B = inputs.shape[0]

        # Creates the inputs for the Soft-DTW outputs, one with halve the temporal dimension, one with the same, and
        # one with double the temporal dimension
        # sdtw_inputs.shape: [B, T/2, C, H, W]
        sdtw_inputs1 = inputs[:, : self.T // 2, :]
        # sdtw_inputs2.shape: [B, T, C, H, W]
        sdtw_inputs2 = inputs
        # sdtw_inputs3.shape: [B, 2T, C, H, W]
        sdtw_inputs3 = torch.cat([inputs, inputs], dim=1)

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
        else:
            ce_loss_raw = self.ce_loss(outputs, labels)

        # The Soft-DTW inputs are transposed to the shape [T/2, B, C, H, W] or [T, B, C, H, W] or
        # [2T, B, C, H, W] for compatibility with SNN model
        # sdtw_inputs1.shape: [T/2, B, C, H, W]
        sdtw_inputs1 = sdtw_inputs1.transpose(0, 1).contiguous()
        # sdtw_inputs2.shape: [T, B, C, H, W]
        sdtw_inputs2 = sdtw_inputs2.transpose(0, 1).contiguous()
        # sdtw_inputs3.shape: [2T, B, C, H, W]
        sdtw_inputs3 = sdtw_inputs3.transpose(0, 1).contiguous()

        # Creates the outputs for the Soft-DTW, one with halve the temporal dimension, one with the same, and
        # one with double the temporal dimension
        # sdtw_outputs1.shape: [T/2, B, K]
        sdtw_outputs1 = self.net(sdtw_inputs1)
        # sdtw_outputs2.shape: [T, B, K]
        sdtw_outputs2 = self.net(sdtw_inputs2)
        # sdtw_outputs3.shape: [2T, B, K]
        sdtw_outputs3 = self.net(sdtw_inputs3)

        # Temporal alignment with Soft-DTW
        sdtw_loss_raw = 0
        sdtw_loss = 0

        # Doesn't care which buffer is in the condition, all 3 are kept in synch
        if not self.sdtw_buffer1.is_empty():
            # Retrieves from the buffer a mini-batch of size 'minibatch_size' of data and their outputs computed
            # with the old parameters of the SNN w.r.t. the current training process, one with halve the temporal
            # dimension, one with the same, and one with double the temporal dimension
            # Applies the transforms to the data making it equivalent to the temporal dimension used for the SER part
            # sdtw_buf_inputs.shape: [B, T, C, H, W]
            # past_sdtw_outputs1.shape: [B, T/2, K]
            sdtw_buf_inputs, past_sdtw_outputs1 = self.sdtw_buffer1.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device)
            # past_sdtw_outputs2.shape: [B, T, K]
            _, past_sdtw_outputs2 = self.sdtw_buffer2.get_data(self.args.minibatch_size, device=self.device)
            # past_sdtw_outputs3.shape: [B, 2T, K]
            _, past_sdtw_outputs3 = self.sdtw_buffer3.get_data(self.args.minibatch_size, device=self.device)

            # The buffer examples are transposed to the shape [T, B, C, H, W] for compatibility
            # sdtw_buf_inputs.shape: [T, B, C, H, W]
            sdtw_buf_inputs = sdtw_buf_inputs.transpose(0, 1).contiguous()

            # Creates the outputs for the buffer examples using the updated parameters of the SNN
            # sdtw_buf_outputs.shape: [T, B, K]
            sdtw_buf_outputs = self.net(sdtw_buf_inputs)

            # The outputs computed with the updated parameters of the SNN are transposed to [B, T, K]
            # for compatibility with Soft-DTW
            # sdtw_buf_outputs.shape: [B, T, K]
            sdtw_buf_outputs = sdtw_buf_outputs.transpose(0, 1).contiguous()

            # Soft-DTW loss computation between past outputs and current outputs
            sdtw1 = self.sdtw_loss(sdtw_buf_outputs, past_sdtw_outputs1)
            sdtw2 = self.sdtw_loss(sdtw_buf_outputs, past_sdtw_outputs2)
            sdtw3 = self.sdtw_loss(sdtw_buf_outputs, past_sdtw_outputs3)

            # OPTIONAL: Normalize the values
            sdtw1 /= (sdtw1 + sdtw2 + sdtw3)
            sdtw2 /= (sdtw1 + sdtw2 + sdtw3)
            sdtw3 /= (sdtw1 + sdtw2 + sdtw3)

            # Assigns randomly the value of the Soft-DTW between:
            # - [B, T, K] and [B, T/2, K]
            # - [B, T, K] and [B, T, K]
            # - [B, T, K] and [B, 2T, K]
            sdtw_loss_raw = (sdtw1 + sdtw2 + sdtw3) / 3
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

        # To add the outputs of the network of shape [T/2, B, K] or [T, B, K] or [2T, B, K] to the buffers,
        # their shape must be transposed to [B, T/2, K] or [B, T, K] or [B, 2T, K] for compatibility with Mammoth
        sdtw_outputs1 = sdtw_outputs1.transpose(0, 1).contiguous()
        sdtw_outputs2 = sdtw_outputs2.transpose(0, 1).contiguous()
        sdtw_outputs3 = sdtw_outputs3.transpose(0, 1).contiguous()

        # Adds to the buffer the current non-augmented data and their outputs for Soft-DTW
        self.sdtw_buffer1.add_data(examples=not_aug_inputs, logits=sdtw_outputs1.data)
        self.sdtw_buffer2.add_data(examples=not_aug_inputs, logits=sdtw_outputs2.data)
        self.sdtw_buffer3.add_data(examples=not_aug_inputs, logits=sdtw_outputs3.data)

        return loss.item()
