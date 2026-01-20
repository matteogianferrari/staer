import torch


class StaticEncoding:
    """Replicate a static input tensor across a time dimension.

    Given an input tensor `x` of shape [B, C, H, W], this transform returns a
    time-expanded tensor of shape [T, B, C, H, W] where each time step contains
    the same input.

    If `mem_efficient=True`, the output is created via `unsqueeze` + `expand` and is a view with shared
    storage across the time dimension (no new memory for the expanded dimension). If `mem_efficient=False`,
    the output is created via `repeat` and is a new tensor (copies data), which uses more memory but
    is safe to modify.

    Attributes:
        T: Number of time steps.
        mem_efficient: Flag for selecting the desired replication method.
    """

    def __init__(self, T: int, mem_efficient: bool = True) -> None:
        """Initialize the StaticEncoding object.

        Args:
            T: The number of time steps to replicate the image.
            mem_efficient: Flag to select the replication process.
        """
        self.T = T
        self.mem_efficient = mem_efficient

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tensor of shape [T, B, C, H, W] on the same device and with the same dtype as `x`.
        """
        if self.mem_efficient:
            return x.unsqueeze(0).expand(self.T, *x.shape)
        else:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
