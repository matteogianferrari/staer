import torch
import torch.nn as nn


class StaticEncoding:
    """PyTorch transform operation to encode an input image.

    This transformation replicates a static input image across all time steps.

    Attributes:
        T: Number of time steps.
        mem_efficient: Flag that selects the replication process, true for memory efficient, false otherwise.
    """

    def __init__(self, T: int, mem_efficient: bool = True) -> None:
        """Initializes the StaticEncoding.

        Args:
            T: The number of time steps to replicate the image.
            mem_efficient: Flag to select the replication process.
        """
        self.T = T
        self.mem_efficient = mem_efficient

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats the input tensor 'x' for every time step.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: A 5D tensor with shape [T, B, C, H, W] representing the encoded input over time.
        """
        if self.mem_efficient:
            return x.unsqueeze(0).expand(self.T, *x.shape)
        else:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
