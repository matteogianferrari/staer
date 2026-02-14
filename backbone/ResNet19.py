import torch
import torch.nn as nn
from backbone import MammothBackbone, register_backbone


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Pre-configured 2D 3x3 convolution to use in ResNet architectures.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of kernels C_out.
        stride: Stride to apply over the spatial dimensionality.

    Returns:
        nn.Conv2d: A pre-configured 2D 3x3 convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Pre-configured 2D 1x1 convolution to use in ResNet architectures.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of kernels C_out.
        stride: Stride to apply over the spatial dimensionality.

    Returns:
        nn.Conv2d: A pre-configured 2D 1x1 convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class ResNetBlock(nn.Module):
    """ResNet basic block.

    Classic ResNet basic block.

    Attributes:
        conv1: First 3×3 convolution of the main branch.
        bn1: Batch normalization after `conv1`.
        relu: ReLU activation.
        conv2: Second 3×3 convolution of the main branch.
        bn2: Batch normalization after `conv2`.
        shortcuts: Projection shortcut (1×1 conv + BN) or identity if not used.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initializes the ResNetBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride to apply in the first 3x3conv and in the 1x1 projection if needed.
        """
        super(ResNetBlock, self).__init__()

        # Main branch
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # Shortcut branch
        self.shortcuts = None
        if stride != 1 or in_channels != out_channels:
            self.shortcuts = nn.Sequential(
                conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the basic block.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The output of the basic block of shape [B, C, H, W].
        """
        identity = x

        # Main branch
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Shortcut branch
        if self.shortcuts is not None:
            identity = self.shortcuts(identity)

        # Add and final ReLU
        x += identity

        return self.relu(x)


class ResNet19(MammothBackbone):
    """ResNet-19 backbone.

    Attributes:
        start_channels: Tracks the current number of channels while building the residual stages.
        stem: Initial feature extractor.
        block1: Residual stage at 128 channels (3 blocks).
        block2: Residual stage at 256 channels (3 blocks).
        block3: Residual stage at 512 channels (2 blocks).
        avg_pool: Global average pooling to (1, 1).
        mlp: Classification head mapping 512 features to `num_classes`.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initializes the ResNet19.

        Args:
            in_channels: Number of channels in the input tensor.
            num_classes: Number of output classes.
        """
        super(ResNet19, self).__init__()

        self.start_channels = 128

        self.stem = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        # Block 1
        self.block1 = self._make_block(num_blocks=3, out_channels=128)

        # Block 2
        self.block2 = self._make_block(num_blocks=3, out_channels=256)

        # Block 3
        self.block3 = self._make_block(num_blocks=2, out_channels=512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Linear(in_features=512, out_features=num_classes, bias=False)

    def _make_block(self, num_blocks: int, out_channels: int) -> nn.Sequential:
        """Builds a residual stage composed of multiple `ResNetBlock`.

        Args:
            num_blocks: Number of `ResNetBlock` instances to include in the stage.
            out_channels: Output channel width for all blocks in the stage.

        Returns:
            nn.Sequential: A sequential container representing the residual stage.
        """
        layers = []

        for _ in range(num_blocks):
            if self.start_channels != out_channels:
                stride = 2
            else:
                stride = 1

            layers.append(
                ResNetBlock(in_channels=self.start_channels, out_channels=out_channels, stride=stride)
            )

            self.start_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Logits of shape [B, K].
        """
        # Stem layer, x.shape: [B, 128, 32, 32]
        x = self.stem(x)

        # First block, x.shape: [B, 128, 32, 32]
        x = self.block1(x)

        # Second block, x.shape: [B, 256, 16, 16]
        x = self.block2(x)

        # First block, x.shape: [B, 512, 8, 8]
        x = self.block3(x)

        # Global average pool layer, x.shape: [B, 512, 1, 1]
        x = self.avg_pool(x)

        # Reshape of tensor, x.shape [B, 512]
        x = torch.flatten(x, 1)

        # Computes the output logits, x.shape: [B, K]
        x = self.mlp(x)

        return x


@register_backbone("resnet19-mnist")
def resnet19_mnist(num_classes: int) -> ResNet19:
    """Instantiates a ResNet19 network for Sequential MNIST dataset.

    Args:
        num_classes: number of output classes.

    Returns:
        A ResNet19 object.
    """
    return ResNet19(in_channels=1, num_classes=num_classes)


@register_backbone("resnet19-cifar10")
def resnet19_cifar(num_classes: int) -> ResNet19:
    """Instantiates a ResNet19 network for Sequential CIFAR10 dataset.

    Args:
        num_classes: number of output classes.

    Returns:
        A ResNet19 object.
    """
    return ResNet19(in_channels=3, num_classes=num_classes)
