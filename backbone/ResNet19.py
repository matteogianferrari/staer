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
    """Basic building block for ResNet architectures.

    Attributes:
        main_branch:
        shortcut:
        lif:
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initializes the SResNetBlock.

        Args:
            in_channels:
            out_channels:
            stride:
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
        """

        Args:
            x:

        Returns:

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
    """

    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """

        Args:
            in_channels:
            num_classes:
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
        """

        Args:
            num_blocks:
            out_channels:
            beta:

        Returns:

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
        """

        Args:
            x:

        Returns:

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
    """
    """

    return ResNet19(in_channels=1, num_classes=num_classes)


@register_backbone("resnet19-cifar10")
def resnet19_cifar(num_classes: int) -> ResNet19:
    """
    """

    return ResNet19(in_channels=3, num_classes=num_classes)
