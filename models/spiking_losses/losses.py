import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSCELoss(nn.Module):
    """Temporal Separation Cross Entropy (TSCE) loss function.

    This criterion extends the classic cross-entropy loss to sequences over the temporal dimension T.
    All time steps share the same ground-truth label.

    Legends:
        T: Time steps.
        B: Batch-size.
        K: Number of classes.
    """

    def __init__(self) -> None:
        """Initializes the TSCELoss.
        """
        super(TSCELoss, self).__init__()

    def forward(self, s_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the TSCE loss.

        Args:
            s_logits: Tensor containing the student logits of shape [T, B, K].
            targets: Tensor containing the ground-truth class indices of shape [B].

        Returns:
            torch.Tensor: A scalar tensor containing the TSCE loss value.
        """
        # Retrieves the shape of the student logits
        T, B, K = s_logits.shape

        # Reshapes the tensor to perform CE in a vectorized way, s_logits.shape: [T * B, K]
        s_logits = s_logits.reshape(-1, K)

        # Repeats the targets to match the vectorized logits, targets.shape: [T * B]
        targets = targets.repeat(T)

        # Computes the CE loss over T and over B
        loss_val = F.cross_entropy(s_logits, targets, reduction='mean')

        return loss_val


class CELoss(nn.Module):
    """Cross Entropy (CE) loss function.

    Common Cross-Entropy loss wrapped with the averaging over the temporal dimension.

    Legends:
    T: Time steps.
    B: Batch-size.
    K: Number of classes.
    """
    def __init__(self) -> None:
        """Initializes the CELoss.
        """
        super(CELoss, self).__init__()

    def forward(self, s_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the CE loss.

        Args:
            s_logits: Tensor containing the student logits of shape [T, B, K].
            targets: Tensor containing the ground-truth class indices of shape [B].

        Returns:
            torch.Tensor: A scalar tensor containing the CE loss value.
        """
        # Averages the logits over the time dimension T
        avg_logits = s_logits.mean(dim=0)

        # Computes the CE loss over B
        loss_val = F.cross_entropy(avg_logits, targets, reduction='mean')

        return loss_val


class MSELoss(nn.Module):
    """Mean Square Error (MSE) loss function.

    Common MSE loss wrapped with the averaging over the temporal dimension.

    Legends:
    T: Time steps.
    B: Batch-size.
    K: Number of classes.
    """
    def __init__(self) -> None:
        """Initializes the MSELoss.
        """
        super(MSELoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the MSE loss.

        Args:
            x: Tensor of shape [T, B, K].
            y: Tensor of shape [T, B, K].

        Returns:
            torch.Tensor: A scalar tensor containing the MSE loss value.
        """
        # Averages the logits over the time dimension T
        x = x.mean(dim=0)
        y = y.mean(dim=0)

        # Computes the MSE loss over B
        loss_val = F.mse_loss(x, y)

        return loss_val
