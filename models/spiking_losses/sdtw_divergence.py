import torch
from torch import nn
from torch.autograd import Function


class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, C, gamma):

        if C.dim() == 2:
            # Allow unbatched cost as convenience
            C = C.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, m, n = C.shape
        device, dtype = C.device, C.dtype

        gamma_val = float(gamma.item())
        if gamma_val <= 0.0:
            raise ValueError("gamma must be > 0 for differentiable soft-DTW.")

        # DP table V with extra row/column for boundary conditions
        # V[:, 0, :] = +inf, V[:, :, 0] = +inf, V[:, 0, 0] = 0
        V = torch.full((B, m + 1, n + 1), float("inf"), device=device, dtype=dtype)
        V[:, 0, 0] = 0.0

        # Transition probabilities P (for gradient of min_γ) on inner grid [1..m, 1..n]
        P = torch.zeros((B, m, n, 3), device=device, dtype=dtype)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # previous accumulated costs
                v_left = V[:, i, j - 1]  # Vi,j-1
                v_diag = V[:, i - 1, j - 1]  # Vi-1,j-1
                v_up = V[:, i - 1, j]  # Vi-1,j
                prev = torch.stack((v_left, v_diag, v_up), dim=-1)  # (B, 3)

                # soft minimum: min_γ(x) = -γ log Σ_k exp(-x_k/γ)
                scaled = -prev / gamma_val  # (B, 3)
                minv = -gamma_val * torch.logsumexp(scaled, dim=-1)  # (B,)

                # gradient wrt (v_left, v_diag, v_up):
                # ∂min_γ / ∂x_k = softmax(-x/γ)_k
                probs = torch.softmax(scaled, dim=-1)  # (B, 3)

                V[:, i, j] = C[:, i - 1, j - 1] + minv
                P[:, i - 1, j - 1, :] = probs

        sdtw = V[:, m, n]  # (B,)

        # Save for backward
        ctx.save_for_backward(P)
        ctx.m = m
        ctx.n = n
        ctx.squeezed = squeezed

        if squeezed:
            return sdtw[0]
        return sdtw

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backprop through sdtw_γ w.r.t C using Algorithm 2: expected alignment E_γ(C).
        grad_output: (B,) or scalar if unbatched input
        returns:
          grad_C:   (B, m, n)
          grad_γ:   None  (we don't differentiate w.r.t gamma)
        """
        (P,) = ctx.saved_tensors
        m, n = ctx.m, ctx.n
        squeezed = ctx.squeezed

        if squeezed:
            grad_output = grad_output.unsqueeze(0)

        B = P.shape[0]
        device, dtype = P.device, P.dtype

        # Extend P with a border (size m+2, n+2) for convenience, as in Alg. 2
        P_ext = torch.zeros((B, m + 2, n + 2, 3), device=device, dtype=dtype)
        P_ext[:, 1:m + 1, 1:n + 1, :] = P
        # Special value at (m+1, n+1): (0, 1, 0)
        P_ext[:, m + 1, n + 1, 1] = 1.0

        # DP for expected alignment E
        E = torch.zeros((B, m + 2, n + 2), device=device, dtype=dtype)
        E[:, m + 1, n + 1] = 1.0

        for j in range(n, 0, -1):
            for i in range(m, 0, -1):
                E[:, i, j] = (
                        P_ext[:, i, j + 1, 0] * E[:, i, j + 1] +  # from left
                        P_ext[:, i + 1, j + 1, 1] * E[:, i + 1, j + 1] +  # from diag
                        P_ext[:, i + 1, j, 2] * E[:, i + 1, j]  # from up
                )

        E_main = E[:, 1:m + 1, 1:n + 1]  # (B, m, n)

        grad_output = grad_output.view(B, 1, 1)
        grad_C = grad_output * E_main  # chain rule: dL/dC = dL/dsdtw * d sdtw/dC

        if squeezed:
            grad_C = grad_C[0]

        # No gradient for gamma
        return grad_C, None


class SoftDTWDivergence(nn.Module):
    """Computes the Soft-DTW divergence between two batches of time series.

    The Soft-DTW divergence is defined in the following paper:
    https://arxiv.org/pdf/2010.08354

    Attributes:
        gamma: Positive smoothing parameter. Smaller values approach classic DTW (less smooth),
            larger values produce smoother alignment costs.
        normalize: If True, divides the per-pair divergence by a simple length-based factor
            (the mean of the two sequence lengths).
        reduction: Specifies the reduction applied to the output:
            - "none": returns a tensor of shape [B] (per-batch divergences)
            - "mean": returns a scalar mean over the batch
            - "sum":  returns a scalar sum over the batch
    """
    def __init__(self, gamma: float = 1.0, reduction: str = "mean", normalize: bool = False) -> None:
        """Initializes the SoftDTWDivergence.

        Args:
            gamma: Soft-DTW smoothing parameter. Must be strictly positive.
            reduction: Output reduction mode: "none", "mean", or "sum".
            normalize: If True, applies simple length-based normalization to the per-batch divergence values.
        """
        super().__init__()

        # Attributes
        self.gamma = gamma
        self.normalize = normalize

        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    @staticmethod
    def _pairwise_sq_euclidean(A: torch.Tensor, B: torch.Tensor):
        """Computes the squared Euclidean distance between A and B.

        This function computes the per-batch pairwise squared Euclidean distances.

        Args:
            A: Tensor that represents a batch of time series with shape [B, T1, K].
            B: Tensor that represents a batch of time series with shape [B, T1, K].

        Returns:
            A tensor representing the per-batch cost matrix with shape [B, T1, T2].
        """
        # Broadcast the operation to [B, T1, T2, K]
        diff = A.unsqueeze(2) - B.unsqueeze(1)

        # Performs the squared Euclidean distance
        cost_matrix = 0.5 * (diff * diff).sum(dim=-1)

        return cost_matrix

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Computes the per-batch Soft-DTW divergence between 2 batches of time series.

        Args:
             X: Tensor representing a batch of time series with shape [B, T1, K].
             Y: Tensor representing a batch of time series with shape [B, T2, K].

        Returns:
            The mean cost if reduction is 'mean', else the sum cost if reduction is 'sum',
            else the tensor of per-batch costs.
        """
        # Sanity checks
        assert X.dim() == 3 and Y.dim() == 3, "X and Y must be (B, T, K)"
        Bx, Tx, Kx = X.shape
        By, Ty, Ky = Y.shape
        assert Bx == By, "Batch sizes of X and Y must match"
        assert Kx == Ky, "Feature dimensions of X and Y must match"

        # Creates a tensor from gamma, the same dtype and on the same device of X
        gamma = X.new_tensor(self.gamma)

        # Computes the cost matrix C between X and Y, tensor shape: [B, Tx, Ty]
        C_xy = self._pairwise_sq_euclidean(X, Y)

        s_xy = _SoftDTW.apply(C_xy, gamma)

        # Computes the cost matrix C between X and X, tensor shape: [B, Tx, Tx]
        C_xx = self._pairwise_sq_euclidean(X, X)

        s_xx = _SoftDTW.apply(C_xx, gamma)  # (B,)

        # Computes the cost matrix C between Y and Y, tensor shape: [B, Ty, Ty]
        C_yy = self._pairwise_sq_euclidean(Y, Y)

        s_yy = _SoftDTW.apply(C_yy, gamma)  # (B,)

        # Computes the Soft-DTW divergence
        D = s_xy - 0.5 * s_xx - 0.5 * s_yy  # (B,)

        # Normalizes the results
        if self.normalize:
            # simple length-based normalization
            mean_len = 0.5 * (Tx + Ty)
            D = D / mean_len

        # Performs the reduction based on the argument
        if self.reduction == "mean":
            return D.mean()
        elif self.reduction == "sum":
            return D.sum()
        else:
            return D
