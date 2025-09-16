import torch
import torch.nn as nn
import torch.nn.functional as F


class TSCELoss(nn.Module):
    """Temporal Separation Cross Entropy (TSCE) loss function.

    This criterion extends the classic cross-entropy loss to sequences over the temporal dimension T.
    All time steps share the same ground-truth label, as intended in the paper.

    Legends:
        T: Time steps.
        B: Batch-size.
        K: Number of classes.

    The math equation that represents the TSCE loss function is:
    $$
    L_{CE} = \frac{1}{B}\sum_{i=1}^B{\frac{1}{T}\sum_{t=1}^T{CE(Q^{stu}(t), y_{true})}}
    $$
    Where $Q^{stu}(t)$ is the probability vector at time $t$, computed by:
    $$
    Q^{stu}(t) = \frac{exp(Z_i(t) / \tau)}{\sum_{j=1}^K{exp(Z_j(t) / \tau)}}
    $$
    Where $Z$ is the vector containing the student logits at time $t$.

    Attributes:
        tau: Hyperparameter that controls the temperature for the softmax.
    """

    def __init__(self, tau: float = 1.0) -> None:
        """Initializes the TSCELoss.

        Args:
            tau: Temperature parameter for softening of logits.
        """
        super(TSCELoss, self).__init__()

        self.tau = tau

    def forward(self, s_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the TSCE loss.

        Args:
            s_logits: Tensor containing the student logits of shape [T, B, K].
            targets: Tensor containing the ground-truth class indices of shape [B] with values in [0, K − 1].

        Returns:
            torch.Tensor: A scalar tensor containing the TSCE loss value.
        """
        # Retrieves the shape of the student logits
        T, B, K = s_logits.shape

        # Scales the logits by the temperature, s_logits.shape: [T, B, K]
        s_logits = s_logits / self.tau

        # Reshapes the tensor to perform CE in a vectorized way, s_logits.shape: [T * B, K]
        s_logits = s_logits.reshape(-1, K)

        # Repeats the targets to match the vectorized logits, targets.shape: [T * B]
        targets = targets.repeat(T)

        # Computes the CE loss over T and over B
        loss_val = F.cross_entropy(s_logits, targets, reduction='mean')

        return loss_val


class TSKLLoss(nn.Module):
    """Temporal Separation Kullback–Leibler Divergence (TSKL) loss function.

    The TSKL loss aligns the student distribution for each time step with the teacher distribution,

    Legends:
        T: Time steps.
        B: Batch-size.
        K: Number of classes.

    The math equation that represents the TSKL loss function is:
    $$
    L_{KL} = \frac{1}{B}\sum_{i=1}^B{\frac{1}{T}\sum_{t=1}^T{KL(Q^{tea} || Q^{stu}(t))}}
    $$
    Where $Q^{stu}(t)$ is the probability vector at time $t$, computed by ($Q^{tea}$ is computed the same way):
    $$
    Q^{stu}(t) = \frac{exp(Z_i(t) / \tau)}{\sum_{j=1}^K{exp(Z_j(t) / \tau)}}
    $$
    Where $Z$ is the vector containing the student logits at time $t$ (same for the teacher but without the time).

    Attributes:
        tau: Hyperparameter that controls the temperature for the softmax.
    """

    def __init__(self, tau: float = 1.0) -> None:
        """Initializes the TSKLLoss.

        Args:
            tau: Temperature parameter for softening of logits.
        """
        super(TSKLLoss, self).__init__()

        self.tau = tau

    def forward(self, t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
        """Computes the TSKL loss.

        Args:
            t_logits: Tensor containing the teacher logits of shape [B, K].
            s_logits: Tensor containing the student logits of shape [T, B, K].

        Returns:
            torch.Tensor: A scalar tensor containing the TSKL loss value.
        """
        # Retrieves the shape of the student logits
        T, B, K = s_logits.shape

        # Computes the log probabilities with the scaled student logits (by temperature), s_log_prob.shape: [T, B, K]
        s_log_prob = F.log_softmax(s_logits / self.tau, dim=-1)

        # Computes the log probabilities with the scaled teacher logits (by temperature), t_log_prob.shape: [B, K]
        t_log_prob = F.log_softmax(t_logits / self.tau, dim=-1)

        # Reshapes the student log probabilities to perform KL in a vectorized way, s_log_prob.shape: [T * B, K]
        s_log_prob = s_log_prob.reshape(-1, K)

        # print(f"s_log_prob.shape: {s_log_prob.shape}")
        # print(f"t_log_prob.shape: {t_log_prob.shape}")
        if t_log_prob.dim() != 3:
            # Repeats the teacher log probabilities to match the student log probabilities, t_log_prob.shape: [T * B, K]
            t_log_prob = t_log_prob.repeat(T, 1)
        else:
            t_log_prob = t_log_prob.reshape(-1, K)

        # Computes the KL divergence between the log probabilities and performs averaging
        loss_val = F.kl_div(input=s_log_prob, target=t_log_prob, log_target=True, reduction='batchmean')

        return loss_val


class EntropyReg(nn.Module):
    """Entropy Regularization (ER) term.

    This regularization term encourages a more uniform output distribution, which can be useful in knowledge
    distillation to keep the student model from becoming over-confident.

    Legends:
        T: Time steps.
        B: Batch-size.
        K: Number of classes.

    The math equation that represents the ER term is:
    $$
    L_{ER} = \frac{1}{B}\sum_{i=1}^B{\frac{1}{T}\sum_{t=1}^T{H(Q^{stu}(t))}}
    $$
    Where $Q^{stu}(t)$ is the probability vector at time $t$, computed by:
    $$
    Q^{stu}(t) = \frac{exp(Z_i(t) / \tau)}{\sum_{j=1}^K{exp(Z_j(t) / \tau)}}
    $$
    Where $Z$ is the vector containing the student logits at time $t$.

    Attributes:
        tau: Hyperparameter that controls the temperature for the softmax.
    """

    def __init__(self, tau: float = 1.0) -> None:
        """Initializes the EntropyReg.

        Args:
            tau: Temperature parameter for softening of logits.
        """
        super(EntropyReg, self).__init__()

        self.tau = tau

    def forward(self, s_logits: torch.Tensor) -> torch.Tensor:
        """Computes the entropy regularizer term.

        Args:
            s_logits: Tensor containing the student logits of shape [T, B, K].

        Returns:
            torch.Tensor: A scalar tensor containing the entropy regularizer value.
        """
        # Computes the probabilities with the scaled logits (by temperature), s_prob.shape: [T, B, K]
        s_prob = F.softmax(s_logits / self.tau, dim=-1)

        # Computes the log probabilities with the scaled logits (by temperature), s_log_prob.shape: [T, B, K]
        s_log_prob = F.log_softmax(s_logits / self.tau, dim=-1)

        # Computes the entropy, entropy.shape: [T, B]
        entropy = -(s_prob * s_log_prob).sum(dim=-1)

        # Performs mean over T and B
        reg_val = entropy.mean()

        return reg_val


class TSERKDLoss(nn.Module):
    """Temporal Separation with Entropy Regularization for Knowledge Distillation (TSERKD) loss function.

    Attributes:
        tau: Hyperparameter that controls the temperature for the softmax.
        alpha: A float representing the weight to balance the 2 loss terms.
        gamma: A float representing the weight for the regularization term.
        ce_loss: The TSCE loss term.
        kl_loss: The TSKL loss term.
        e_reg: The ER regularization term.
    """

    def __init__(self, alpha: float = 1e-1, gamma: float = 1e-3, tau: float = 6) -> None:
        """Initializes the TSERKDLoss.

        Args:
            alpha: Hyperparameter that balances the TSCE and TSKL terms in the loss.
            gamma: Hyperparameter to regulate the entropy regularization term.
            tau: Temperature parameter for softening of logits.
        """
        super(TSERKDLoss, self).__init__()

        # Hyperparameters
        self.tau = tau

        self.alpha = alpha
        self.gamma = gamma

        # Loss terms
        self.ce_loss = TSCELoss(tau=tau)
        self.kl_loss = TSKLLoss(tau=tau)
        self.e_reg = EntropyReg(tau=tau)

    def forward(self, t_logits: torch.Tensor, s_logits: torch.Tensor, targets: torch.Tensor):
        """Computes the TSERKD loss.

        Args:
            t_logits: Tensor containing the teacher logits of shape [B, K].
            s_logits: Tensor containing the student logits of shape [T, B, K].
            targets: Tensor containing the ground-truth class indices of shape [B] with values in [0, K − 1].

        Returns:
            tuple: A 4-tuple containing:
                - total_loss: A scalar tensor containing the total loss value.
                - ce_loss: A scalar tensor containing the TSCE loss value.
                - kl_loss: A scalar tensor containing the TSKL loss value.
                - e_reg: A scalar tensor containing the entropy regularizer value.
        """
        # Computes the TSCE loss term
        ce_loss = self.ce_loss(s_logits=s_logits, targets=targets)

        # Computes the TSKL loss term
        kl_loss = self.kl_loss(t_logits=t_logits, s_logits=s_logits)

        # Computes the ER regularization term
        e_reg = self.e_reg(s_logits=s_logits)

        # Computes the total loss value
        # Multiplies the KL loss value by squared tau to avoid vanishing gradient and slower training
        # (typical operation performed in knowledge distillation)
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * (self.tau ** 2) * kl_loss - self.gamma * e_reg

        return total_loss, ce_loss, kl_loss, e_reg
