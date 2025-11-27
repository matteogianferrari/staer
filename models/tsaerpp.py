import torch
import torchvision.transforms as transforms
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets.transforms.static_encoding import StaticEncoding
from models.spiking_er.losses import TSCELoss, TSKLLoss
from models.spiking_er.soft_dtw import SoftDTW


class Tsaerpp(ContinualModel):
    """Continual learning via Experience Replay for SNN with temporal alignment."""
    NAME = 'tsaerpp'
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

        parser.add_argument('--alpha', type=float, default=0.18,
                            help='Hyperparameter that balances the KL term of the loss.')

        parser.add_argument('--beta', type=float, default=1e-4,
                            help='Hyperparameter that balances the SDTW term of the loss.')

        parser.add_argument('--sdtw_gamma', type=float, default=1.0,
                            help='Native Hyperparameter of SDTW.')

        parser.add_argument('--sdtw_norm', type=bool, default=True,
                            help='Native Hyperparameter of SDTW on data normalization.')

        parser.add_argument('--theta', type=float, default=0.14,
                            help='Hyperparameter to balance CE for reharsal.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The TSER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Tsaerpp, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

        self.T = args.T
        self.tau = args.tau
        self.alpha = args.alpha
        self.beta = args.beta
        self.sdtw_gamma = args.sdtw_gamma
        self.sdtw_norm = args.sdtw_norm
        self.theta = args.theta

        self.tsce_loss = TSCELoss()
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

        # TSCE loss
        # Can be always computed, even when the buffer is empty
        loss_tsce_raw = self.tsce_loss(s_logits=s_logits, targets=labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )
            
            # The inputs are transposed to the shape [T, B, C, H, W] for compatibility
            buf_inputs = buf_inputs.transpose(0, 1).contiguous()
            # The logits are transposed to the shape [T, B, K]
            buf_logits = buf_logits.transpose(0, 1).contiguous()

            buf_outputs = self.net(buf_inputs)

            # TSKL loss
            # Can be computed only when the buffer is not empty
            loss_tskl_raw = self.tskl_loss(t_logits=buf_logits, s_logits=buf_outputs)
            loss_tskl = self.alpha * (self.tau ** 2) * loss_tskl_raw

            # Re-transpose to the shape [B, T, K] for compatibility with sdtw
            buf_logits = buf_logits.transpose(0, 1).contiguous()
            # The logits are transposed to the shape [B, T, K]
            buf_outputs = buf_outputs.transpose(0, 1).contiguous()

            # SDTW loss
            loss_sdtw_raw = self.sdtw_loss(buf_outputs, buf_logits).mean(dim=0)
            loss_sdtw = self.beta * loss_sdtw_raw

            buf_x2, buf_y2, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.buffer_transform, device=self.device
            )

            buf_x2 = buf_x2.transpose(0, 1).contiguous()

            buf_out2 = self.net(buf_x2)
            loss_tsce_buf_raw = self.tsce_loss(s_logits=buf_out2, targets=buf_y2)
            loss_tsce_buf = self.args.theta * loss_tsce_buf_raw

            loss = loss_tsce_raw + loss_tskl + loss_sdtw + loss_tsce_buf
        else:
            loss = loss_tsce_raw

        loss.backward()

        # loss.backward(retain_graph=True)
        # if not self.buffer.is_empty():
        #     shared_params = [p for p in self.net.parameters() if p.requires_grad]
        #
        #     # loss1 gradients
        #     g1 = torch.autograd.grad(loss_ce_raw, shared_params, retain_graph=True, create_graph=False)
        #     # loss2 gradients
        #     g2 = torch.autograd.grad(loss_tskl_raw, shared_params, retain_graph=True, create_graph=False)
        #     # loss3 gradients
        #     g3 = torch.autograd.grad(loss_sdtw_raw, shared_params, retain_graph=True, create_graph=False)
        #     # loss4 gradients
        #     g4 = torch.autograd.grad(loss_ce_buf_raw, shared_params, retain_graph=True, create_graph=False)
        #
        #     # Compute gradient norm for each loss term (over all shared params)
        #     g1_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g1))).item()
        #     g2_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g2))).item()
        #     g3_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g3))).item()
        #     g4_norm = torch.sqrt(sum((gi.norm() ** 2 for gi in g4))).item()
        #
        #     with open("grad_norm.txt", "a", encoding="utf-8") as f:
        #         f.write(f"{g1_norm};{g2_norm}; {g3_norm}; {g4_norm}\n")

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=s_logits.data)

        return loss.item()
