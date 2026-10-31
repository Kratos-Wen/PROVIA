"""Cache causal frozen-DINO patch maps for training-fold procedural events.

Only event start, pre-contact midpoint, and contact frames are decoded.  The
reserved test partition is excluded before any video is opened.  Native patch
maps are stored in float16; no labels or hand tracks are consumed by the
frozen extractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



def _load_backbone(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
    backbone: str = "dinov2_vits14",
):
    if backbone == "dinov2_vits14":
        if not repo.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("local DINOv2 repository or checkpoint is missing")
        model = torch.hub.load(
            str(repo),
            "dinov2_vits14",
            source="local",
            pretrained=False,
            verbose=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"unsupported frozen backbone: {backbone}")
    model.eval().requires_grad_(False).to(device)
    return model


def _load_backbone_from_provenance(
    repo: Path,
    cache: Mapping[str, object],
    device: torch.device,
) -> tuple[str, nn.Module]:
    """Dispatch the frozen encoder exactly as recorded by a fitted checkpoint."""

    backbone = str(cache.get("backbone", ""))
    if backbone != "dinov2_vits14":
        raise ValueError("checkpoint has an unsupported visual backbone")
    checkpoint_value = cache.get("backbone_checkpoint")
    if not isinstance(checkpoint_value, (str, Path)) or not str(checkpoint_value):
        raise ValueError("checkpoint has no frozen-backbone provenance")
    return backbone, _load_backbone(
        repo,
        Path(str(checkpoint_value)),
        device,
        backbone=backbone,
    )


def _preprocess(decoded: np.ndarray, device: torch.device) -> torch.Tensor:
    frames = torch.from_numpy(decoded).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32
    )
    frames = frames.div_(255.0)
    frames = F.interpolate(
        frames,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (frames - mean) / std
