"""NAM (Normalisation-based Attention Module) for YOLO integration.

Implements the NAM attention mechanism from:
    NAM: Normalization-based Attention Module (https://arxiv.org/abs/2111.12419)

Uses batch normalisation scaling factors to measure channel importance,
which is more efficient than SE/CBAM while maintaining competitive performance.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NAMChannelAttention(nn.Module):
    """Channel attention using batch normalisation scaling factors.

    The BN gamma parameter naturally indicates channel importance —
    channels with larger gamma contribute more to the output.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn_weight = self.bn.weight.data.abs()
        bn_weight = bn_weight / (bn_weight.sum() + 1e-8)
        out = self.bn(x)
        weight = self.gamma * bn_weight.view(1, -1, 1, 1)
        attention = self.sigmoid(weight)
        return out * attention


class NAMSpatialAttention(nn.Module):
    """Pixel-wise spatial attention using batch normalisation."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        attention = self.sigmoid(out)
        return x * attention


class NAMBlock(nn.Module):
    """Combined NAM attention block (channel + spatial) with residual connection.

    Drop-in module for YOLO architectures. Accepts a single channels argument
    to match the Ultralytics module registration interface.
    """

    def __init__(self, channels: int, use_spatial: bool = True):
        super().__init__()
        self.channel_attention = NAMChannelAttention(channels)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_attention = NAMSpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x)
        if self.use_spatial:
            out = self.spatial_attention(out)
        return out + x  # Residual


def register_nam_modules() -> None:
    """Register NAM modules with Ultralytics so they can be used in model YAMLs.

    Call this before loading a custom YOLO model YAML that references NAMBlock.
    """
    try:
        import ultralytics.nn.modules as modules

        # Register NAMBlock so YOLO YAML parser can find it
        if not hasattr(modules, "NAMBlock"):
            modules.NAMBlock = NAMBlock
            # Also add to the modules __all__ if it exists
            if hasattr(modules, "__all__"):
                modules.__all__.append("NAMBlock")

        # Register in the tasks module where model building happens
        from ultralytics.nn import tasks
        if not hasattr(tasks, "NAMBlock"):
            tasks.NAMBlock = NAMBlock

        logger.info("NAMBlock registered with Ultralytics")

    except ImportError:
        logger.warning(
            "ultralytics not installed — NAM registration skipped. "
            "Install with: pip install ultralytics>=8.3.0"
        )
    except Exception as e:
        logger.warning(f"Failed to register NAM modules: {e}")
