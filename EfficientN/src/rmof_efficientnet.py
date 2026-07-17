"""Model, training loop, and command-line entry point for RMOF-Net experiments.

Dataset loading lives in :mod:`rmof_data`; shared metrics and report figures
live in :mod:`rmof_metrics`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models

from rmof_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    make_loaders,
)
from rmof_metrics import metrics_from_predictions, save_confusion


@dataclass
class ModelConfig:
    token_dim: int = 128
    dropout: float = 0.30
    use_stage3: bool = True
    use_stage4: bool = True
    use_region_tokens: bool = True
    use_color_stats: bool = True
    use_color_texture: bool = True
    fusion: str = "region_attention"  # concat, gate, attention, or colour_residual
    attention_heads: int = 4
    num_classes: int = 3


def preset_config(name: str) -> ModelConfig:
    """Return the requested main-ablation configuration."""
    base = ModelConfig()
    presets = {
        "cnn_baseline": dict(
            use_stage3=False,
            use_region_tokens=False,
            use_color_stats=False,
            use_color_texture=False,
            fusion="concat",
        ),
        "safe_deep_residual": dict(
            use_stage3=False,
            use_stage4=True,
            use_region_tokens=False,
            use_color_stats=False,
            use_color_texture=False,
            fusion="concat",
        ),
        "multiscale": dict(
            use_stage3=True,
            use_region_tokens=False,
            use_color_stats=False,
            use_color_texture=False,
            fusion="concat",
        ),
        "regions": dict(
            use_stage3=True,
            use_region_tokens=True,
            use_color_stats=False,
            use_color_texture=False,
            fusion="concat",
        ),
        "color_stats": dict(
            use_color_stats=True, use_color_texture=False, fusion="concat"
        ),
        "color_texture": dict(
            use_color_stats=True, use_color_texture=True, fusion="concat"
        ),
        "safe_colour_residual": dict(
            use_stage3=False,
            use_stage4=True,
            use_region_tokens=True,
            use_color_stats=True,
            use_color_texture=False,
            fusion="colour_residual",
        ),
        "region_cross_attention": dict(fusion="region_attention"),
        "ordinal_supervision": dict(fusion="region_attention"),
        "fusion_concat": dict(fusion="concat"),
        "fusion_gate": dict(fusion="gate"),
        "fusion_cross_attention": dict(fusion="cross_attention"),
        "fusion_region_attention": dict(fusion="region_attention"),
        "loss_ce": dict(fusion="region_attention"),
        "loss_ce_emd": dict(fusion="region_attention"),
        "loss_ce_score": dict(fusion="region_attention"),
        "loss_ce_emd_score": dict(fusion="region_attention"),
    }
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}")
    return replace(base, **presets[name])


def loss_weights_for_preset(name: str) -> Tuple[float, float]:
    weights = {
        "ordinal_supervision": (0.25, 0.25),
        "loss_ce": (0.0, 0.0),
        "loss_ce_emd": (0.25, 0.0),
        "loss_ce_score": (0.0, 0.25),
        "loss_ce_emd_score": (0.25, 0.25),
    }
    return weights.get(name, (0.0, 0.0))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class SpatialTokenExtractor(nn.Module):
    """One global token and, optionally, four row-major 2x2 region tokens."""

    def __init__(self, channels: int, token_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(channels, token_dim), nn.GELU(), nn.LayerNorm(token_dim)
        )

    def forward(self, feature: torch.Tensor, regions: bool) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(2)
        if regions:
            pooled = torch.cat((pooled, F.adaptive_avg_pool2d(feature, 2).flatten(2)), dim=2)
        tokens = pooled.transpose(1, 2)
        return self.project(tokens)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB-to-HSV conversion for RGB values in [0, 1]."""
    maxc, max_index = rgb.max(dim=1)
    minc = rgb.min(dim=1).values
    delta = maxc - minc
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    safe_delta = delta.clamp_min(1e-6)
    hue = torch.zeros_like(maxc)
    hue = torch.where(max_index == 0, torch.remainder((g - b) / safe_delta, 6.0), hue)
    hue = torch.where(max_index == 1, (b - r) / safe_delta + 2.0, hue)
    hue = torch.where(max_index == 2, (r - g) / safe_delta + 4.0, hue)
    hue = torch.where(delta > 1e-6, hue / 6.0, torch.zeros_like(hue))
    saturation = torch.where(maxc > 1e-6, delta / maxc.clamp_min(1e-6), torch.zeros_like(maxc))
    return torch.stack((hue, saturation, maxc), dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Approximate sRGB to CIE Lab conversion, sufficient for colour statistics."""
    linear_rgb = torch.where(
        rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92
    )
    matrix = rgb.new_tensor(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear_rgb)
    reference_white = rgb.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    xyz = xyz / reference_white
    delta = 6.0 / 29.0
    f_xyz = torch.where(
        xyz > delta**3, xyz.pow(1.0 / 3.0), xyz / (3 * delta**2) + 4.0 / 29.0
    )
    l = 116.0 * f_xyz[:, 1] - 16.0
    a = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])
    return torch.stack((l, a, b), dim=1)


class SoftForegroundMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(-0.03))
        self.log_sharpness = nn.Parameter(torch.tensor(12.0))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        exg = 2.0 * rgb[:, 1:2] - rgb[:, 0:1] - rgb[:, 2:3]
        sharpness = F.softplus(self.log_sharpness)
        return torch.sigmoid(sharpness * (exg - self.threshold))


def weighted_region_statistics(
    features: torch.Tensor, mask: torch.Tensor, regions: bool
) -> torch.Tensor:
    """Return mask-weighted mean and standard deviation per global/2x2 region."""
    def pool(grid: int) -> torch.Tensor:
        denominator = F.adaptive_avg_pool2d(mask, grid).clamp_min(1e-5)
        mean = F.adaptive_avg_pool2d(features * mask, grid) / denominator
        second_moment = F.adaptive_avg_pool2d(features.square() * mask, grid) / denominator
        std = (second_moment - mean.square()).clamp_min(0.0).sqrt()
        return torch.cat((mean, std), dim=1).flatten(2).transpose(1, 2)

    global_stats = pool(1)
    return torch.cat((global_stats, pool(2)), dim=1) if regions else global_stats


class ColourBranch(nn.Module):
    """Soft leaf mask, explicit statistics and a shallow learned colour texture encoder."""

    def __init__(self, token_dim: int, use_stats: bool, use_texture: bool) -> None:
        super().__init__()
        self.use_stats = use_stats
        self.use_texture = use_texture
        self.mask = SoftForegroundMask()
        if use_stats:
            self.stats_project = nn.Sequential(
                nn.Linear(18, token_dim), nn.GELU(), nn.LayerNorm(token_dim)
            )
        if use_texture:
            texture_channels = max(32, token_dim // 2)
            self.texture = nn.Sequential(
                nn.Conv2d(3, texture_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(texture_channels),
                nn.GELU(),
                nn.Conv2d(texture_channels, token_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(token_dim),
                nn.GELU(),
            )
            self.texture_project = nn.Sequential(nn.LayerNorm(token_dim), nn.GELU())
        self.merge = (
            nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
            if use_stats and use_texture
            else None
        )

    def forward(self, rgb: torch.Tensor, regions: bool) -> torch.Tensor:
        mask = self.mask(rgb)
        tokens: List[torch.Tensor] = []
        if self.use_stats:
            hsv = rgb_to_hsv(rgb)
            lab_scale = rgb.new_tensor((100.0, 128.0, 128.0)).view(1, 3, 1, 1)
            lab = rgb_to_lab(rgb) / lab_scale
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            exg = (2.0 * g - r - b) / 2.0
            exgr = (3.0 * g - 2.4 * r - b) / 3.4
            ngrdi = (g - r) / (g + r).clamp_min(1e-5)
            features = torch.cat((hsv, lab, exg, exgr, ngrdi), dim=1)
            tokens.append(self.stats_project(weighted_region_statistics(features, mask, regions)))
        if self.use_texture:
            texture = self.texture(rgb * mask)
            token_grid = 2 if regions else 1
            local = F.adaptive_avg_pool2d(texture, token_grid).flatten(2).transpose(1, 2)
            if regions:
                global_token = F.adaptive_avg_pool2d(texture, 1).flatten(2).transpose(1, 2)
                local = torch.cat((global_token, local), dim=1)
            tokens.append(self.texture_project(local))
        return tokens[0] if len(tokens) == 1 else self.merge(torch.cat(tokens, dim=-1))


class TokenAttentionPool(nn.Module):
    """Learn a token summary, starting from an unbiased uniform average."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.score = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = self.score(self.norm(tokens)).softmax(dim=1)
        return (weights * tokens).sum(dim=1)


class CrossAttention(nn.Module):
    """Cross-attention with optional learnable region-to-region attention bias."""

    def __init__(self, token_dim: int, heads: int, aligned: bool) -> None:
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = token_dim // heads
        self.aligned = aligned
        self.query = nn.Linear(token_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.out = nn.Linear(token_dim, token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_value_norm = nn.LayerNorm(token_dim)
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim), nn.GELU(), nn.Linear(2 * token_dim, token_dim)
        )
        self.region_bias = nn.Parameter(torch.zeros(5, 5)) if aligned else None
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.feed_forward[-1].weight)
        nn.init.zeros_(self.feed_forward[-1].bias)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        query_regions: torch.Tensor,
        key_regions: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, token_dim = query_tokens.shape
        key_count = key_value_tokens.shape[1]
        normalised_query = self.query_norm(query_tokens)
        normalised_key_value = self.key_value_norm(key_value_tokens)
        q = self.query(normalised_query).reshape(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(normalised_key_value).reshape(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(normalised_key_value).reshape(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.aligned:
            bias = self.region_bias[query_regions[:, None], key_regions[None, :]]
            scores = scores + bias.unsqueeze(0).unsqueeze(0)
        attention = scores.softmax(dim=-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, query_count, token_dim)
        fused = query_tokens + self.out(attended)
        return fused + self.feed_forward(self.feed_forward_norm(fused))


class EfficientNetColourFusion(nn.Module):
    """EfficientNet-B0 plus a safely initialized colour-and-region logit residual."""

    def __init__(self, config: ModelConfig, pretrained: bool = False) -> None:
        super().__init__()
        if not (config.use_stage3 or config.use_stage4):
            raise ValueError("At least one CNN stage must be enabled")
        self.config = config
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=config.dropout, inplace=True),
            nn.Linear(backbone.classifier[1].in_features, config.num_classes),
        )
        self.base_classifier = backbone.classifier
        self.stage3_tokens = SpatialTokenExtractor(112, config.token_dim)
        self.stage4_tokens = SpatialTokenExtractor(320, config.token_dim)
        self.deep_pool = TokenAttentionPool(config.token_dim)
        self.colour: Optional[ColourBranch]
        if config.use_color_stats or config.use_color_texture:
            self.colour = ColourBranch(
                config.token_dim, config.use_color_stats, config.use_color_texture
            )
        else:
            self.colour = None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        if self.colour is not None and config.fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(2 * config.token_dim, config.token_dim), nn.Sigmoid())
            nn.init.zeros_(self.gate[0].weight)
            nn.init.constant_(self.gate[0].bias, math.log(0.9 / 0.1))
        else:
            self.gate = None
        if self.colour is not None and config.fusion in {"cross_attention", "region_attention"}:
            self.cross_attention = CrossAttention(
                config.token_dim,
                config.attention_heads,
                aligned=config.fusion == "region_attention",
            )
        else:
            self.cross_attention = None
        self.colour_pool = TokenAttentionPool(config.token_dim) if self.colour is not None else None
        head_dim = 2 * config.token_dim if self.colour is not None and config.fusion == "concat" else config.token_dim
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Dropout(config.dropout), nn.Linear(head_dim, config.num_classes)
        )
        nn.init.zeros_(self.aux_classifier[-1].weight)
        nn.init.zeros_(self.aux_classifier[-1].bias)
        self.score_head = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, 1))
        self.base_frozen = False

    def freeze_base(self) -> None:
        """Freeze the validated EfficientNet path while fitting the logit residual."""
        for module in (self.features, self.base_classifier):
            module.requires_grad_(False)
            module.eval()
        if self.config.fusion == "colour_residual":
            # These deep-token modules are intentionally bypassed by the
            # low-capacity colour-only correction.
            for module in (self.stage3_tokens, self.stage4_tokens, self.deep_pool):
                module.requires_grad_(False)
                module.eval()
        self.base_frozen = True

    def _deep_tokens(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage3 = stage4 = None
        x = images
        for index, block in enumerate(self.features):
            x = block(x)
            if index == 5:
                stage3 = x  # 112 channels, 14x14 for 224x224 input
            elif index == 7:
                stage4 = x  # 320 channels, 7x7 for 224x224 input
        token_sets: List[torch.Tensor] = []
        region_ids: List[torch.Tensor] = []
        base_regions = torch.arange(5 if self.config.use_region_tokens else 1, device=images.device)
        if self.config.use_stage3:
            if stage3 is None:
                raise RuntimeError("EfficientNet stage 3 was not produced")
            token_sets.append(self.stage3_tokens(stage3, self.config.use_region_tokens))
            region_ids.append(base_regions)
        if self.config.use_stage4:
            if stage4 is None:
                raise RuntimeError("EfficientNet stage 4 was not produced")
            token_sets.append(self.stage4_tokens(stage4, self.config.use_region_tokens))
            region_ids.append(base_regions)
        return torch.cat(token_sets, dim=1), torch.cat(region_ids), x

    def forward(self, images: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        deep_tokens, deep_regions, final_features = self._deep_tokens(images)
        base_logits = self.base_classifier(self.avgpool(final_features).flatten(1))
        deep_summary = self.deep_pool(deep_tokens)
        if self.colour is None:
            fused = deep_summary
        else:
            rgb = (images * self.std + self.mean).clamp(0.0, 1.0)
            colour_tokens = self.colour(rgb, self.config.use_region_tokens)
            colour_summary = self.colour_pool(colour_tokens)
            colour_regions = torch.arange(colour_tokens.shape[1], device=images.device)
            if self.config.fusion == "colour_residual":
                fused = colour_summary
            elif self.config.fusion == "concat":
                fused = torch.cat((deep_summary, colour_summary), dim=-1)
            elif self.config.fusion == "gate":
                gate = self.gate(torch.cat((deep_summary, colour_summary), dim=-1))
                fused = gate * deep_summary + (1.0 - gate) * colour_summary
            elif self.config.fusion in {"cross_attention", "region_attention"}:
                fused_tokens = self.cross_attention(
                    deep_tokens, colour_tokens, deep_regions, colour_regions
                )
                fused = self.deep_pool(fused_tokens)
            else:
                raise ValueError(f"Unknown fusion mode: {self.config.fusion}")
        aux_logits = self.aux_classifier(fused)
        return {
            "logits": base_logits + aux_logits,
            "base_logits": base_logits,
            "aux_logits": aux_logits,
            "score": self.score_head(fused),
        }


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        return {"logits": self.classifier(self.encoder(images).flatten(1)), "score": None}


class ClassifierWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        return {"logits": self.model(images), "score": None}


def build_baseline_model(
    model_name: str, config: ModelConfig, pretrained: bool
) -> nn.Module:
    """Build one of the non-RMOF comparison models."""
    if model_name == "simple_cnn":
        return SimpleCNN(config.num_classes)
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=config.dropout, inplace=True),
            nn.Linear(model.classifier[1].in_features, config.num_classes),
        )
        return ClassifierWrapper(model)
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        return ClassifierWrapper(model)
    if model_name == "deit_tiny":
        try:
            import timm
        except ImportError as error:
            raise ImportError("DeiT-Tiny requires `pip install timm`.") from error
        return ClassifierWrapper(
            timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=config.num_classes)
        )
    raise ValueError(f"Unknown baseline model: {model_name}")


def build_model(model_name: str, config: ModelConfig, pretrained: bool) -> nn.Module:
    """Dispatch the RMOF main branch or one isolated baseline branch."""
    if model_name == "rmof_efficientnet":
        return EfficientNetColourFusion(config, pretrained)
    return build_baseline_model(model_name, config, pretrained)


def load_frozen_efficientnet_base(
    model: EfficientNetColourFusion, checkpoint_path: Path
) -> None:
    """Load a ClassifierWrapper EfficientNet checkpoint into the residual model."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    mapped: Dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key.startswith("model.features."):
            mapped["features." + key.removeprefix("model.features.")] = value
        elif key.startswith("model.classifier."):
            mapped["base_classifier." + key.removeprefix("model.classifier.")] = value
        elif key.startswith("features.") or key.startswith("base_classifier."):
            mapped[key] = value
    if not mapped:
        raise ValueError(f"No EfficientNet feature/classifier weights found in {checkpoint_path}")
    incompatible = model.load_state_dict(mapped, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key in mapped]
    if unexpected:
        raise ValueError(f"Unexpected base checkpoint keys: {unexpected}")
    model.freeze_base()


class OrdinalLoss(nn.Module):
    def __init__(self, emd_weight: float, score_weight: float, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.emd_weight = emd_weight
        self.score_weight = score_weight

    def forward(self, output: Dict[str, Optional[torch.Tensor]], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = output["logits"]
        ce = self.cross_entropy(logits, target)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
        emd = (probabilities.cumsum(dim=1) - one_hot.cumsum(dim=1)).square().mean()
        score = output["score"]
        if self.score_weight and score is None:
            raise ValueError("Score regression was selected but this baseline has no score head")
        score_loss = (
            F.smooth_l1_loss(score.squeeze(1), target.float()) if score is not None else ce.new_zeros(())
        )
        total = ce + self.emd_weight * emd + self.score_weight * score_loss
        return {"total": total, "ce": ce, "emd": emd, "score": score_loss}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: OrdinalLoss,
    amp_enabled: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, List[int], List[int], List[str]]:
    model.eval()
    labels: List[int] = []
    predictions: List[int] = []
    names: List[str] = []
    total_loss = 0.0
    total_samples = 0
    elapsed = 0.0
    with torch.inference_mode():
        for images, target, batch_names in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                losses = objective(output, target)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            prediction = output["logits"].argmax(dim=1)
            total_loss += losses["total"].item() * target.shape[0]
            total_samples += target.shape[0]
            labels.extend(target.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            names.extend(batch_names)
    metrics = metrics_from_predictions(labels, predictions)
    metrics["loss"] = total_loss / total_samples
    metrics["inference_ms_per_image"] = 1000.0 * elapsed / total_samples
    return metrics, confusion_matrix(labels, predictions, labels=[0, 1, 2]), labels, predictions, names


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: OrdinalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    if isinstance(model, EfficientNetColourFusion) and model.base_frozen:
        # Frozen BatchNorm statistics and classifier dropout must remain in
        # evaluation mode or the supposedly fixed baseline would still drift.
        model.features.eval()
        model.base_classifier.eval()
    total_loss = 0.0
    labels: List[int] = []
    predictions: List[int] = []
    for images, target, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images)
            losses = objective(output, target)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses["total"].item() * target.shape[0]
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(output["logits"].argmax(dim=1).detach().cpu().tolist())
    metrics = metrics_from_predictions(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Fine-tune ImageNet features conservatively while learning new token heads faster."""
    if isinstance(model, EfficientNetColourFusion) and args.backbone_lr_scale != 1.0:
        backbone = [parameter for parameter in model.features.parameters() if parameter.requires_grad]
        backbone_ids = {id(parameter) for parameter in backbone}
        heads = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        if not backbone:
            return torch.optim.AdamW(
                heads, lr=args.learning_rate, weight_decay=args.weight_decay
            )
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": args.learning_rate * args.backbone_lr_scale},
                {"params": heads, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)


def run_training(args: argparse.Namespace, config: ModelConfig, emd_weight: float, score_weight: float) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, validation_loader, test_loader = make_loaders(args)
    model = build_model(args.model, config, args.pretrained).to(device)
    if args.base_checkpoint is not None:
        if not isinstance(model, EfficientNetColourFusion):
            raise ValueError("--base-checkpoint is only valid for --model rmof_efficientnet")
        load_frozen_efficientnet_base(model, Path(args.base_checkpoint))
    objective = OrdinalLoss(emd_weight, score_weight, args.label_smoothing)
    optimizer = make_optimizer(model, args)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = Path(args.output_dir) / args.experiment_name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    if args.base_checkpoint is not None:
        initial_metrics, _, _, _, _ = evaluate(
            model, validation_loader, device, objective, amp_enabled
        )
        best_f1 = initial_metrics["macro_f1"]
        print(f"epoch 000: frozen baseline val macro-F1={best_f1:.4f}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, device, objective, optimizer, scaler, amp_enabled
        )
        val_metrics, _, _, _, _ = evaluate(
            model, validation_loader, device, objective, amp_enabled
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch:03d}: train loss={train_metrics['loss']:.4f}, "
            f"val macro-F1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break
    model.load_state_dict(best_state)
    test_metrics, cm, labels, predictions, names = evaluate(
        model, test_loader, device, objective, amp_enabled
    )
    test_metrics.update(
        {
            "seed": args.seed,
            "experiment": args.experiment_name,
            "model": args.model,
            "parameters": parameter_count(model),
            "best_validation_macro_f1": best_f1,
            "emd_weight": emd_weight,
            "score_weight": score_weight,
            "base_checkpoint": args.base_checkpoint,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "augmentation": args.augmentation,
                "train_fraction": args.train_fraction,
                "learning_rate": args.learning_rate,
                "backbone_lr_scale": args.backbone_lr_scale,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "dropout": config.dropout,
                "amp": amp_enabled,
                "cache_images": args.cache_images,
            },
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({"filename": names, "label": labels, "prediction": predictions}).to_csv(
        output_dir / "predictions.csv", index=False
    )
    np.save(output_dir / "confusion.npy", cm)
    save_confusion(cm, output_dir / "confusion_matrix.png", f"{args.experiment_name}, seed {args.seed}")
    torch.save(
        {"state_dict": model.state_dict(), "model_config": asdict(config), "metrics": test_metrics},
        output_dir / "best_model.pt",
    )
    print(json.dumps(test_metrics, indent=2))
    return output_dir


def smoke_check(device: torch.device) -> None:
    """Check baseline equivalence, then run the custom forward/loss/backward path."""
    config = preset_config("ordinal_supervision")
    model = EfficientNetColourFusion(config, pretrained=False).to(device)
    images = torch.randn(2, 3, 128, 128, device=device)
    target = torch.tensor([0, 2], device=device)
    model.eval()
    with torch.inference_mode():
        initial_output = model(images)
    if not torch.equal(initial_output["logits"], initial_output["base_logits"]):
        raise RuntimeError("Zero-initialized auxiliary path changed the baseline logits")
    if torch.count_nonzero(initial_output["aux_logits"]):
        raise RuntimeError("Auxiliary logits must be exactly zero at initialization")
    model.train()
    output = model(images)
    assert output["logits"].shape == (2, 3)
    assert output["score"] is not None and output["score"].shape == (2, 1)
    losses = OrdinalLoss(0.25, 0.25)(output, target)
    losses["total"].backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Backward pass did not produce gradients")
    if model.aux_classifier[-1].weight.grad is None:
        raise RuntimeError("Auxiliary classifier did not receive gradients")
    print(
        "smoke check passed: exact baseline initialization, logits=(2,3), "
        "score=(2,1), forward/backward/loss all valid"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="split_40_10_50.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--preset", default="safe_deep_residual", choices=[
        "cnn_baseline", "safe_deep_residual", "multiscale", "regions", "color_stats", "color_texture",
        "safe_colour_residual",
        "region_cross_attention", "ordinal_supervision", "fusion_concat", "fusion_gate",
        "fusion_cross_attention", "fusion_region_attention", "loss_ce", "loss_ce_emd",
        "loss_ce_score", "loss_ce_emd_score",
    ])
    parser.add_argument("--model", default="rmof_efficientnet", choices=[
        "rmof_efficientnet", "simple_cnn", "efficientnet_b0", "resnet18", "deit_tiny"
    ])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--augmentation", choices=["mild", "leaf", "strong"], default="mild")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--token-dim", type=int, default=None,
        help="Optional token width for the auxiliary residual branch.",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Optional validated EfficientNet checkpoint; freezes it and fits only the residual.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emd-weight", type=float, default=None)
    parser.add_argument("--score-weight", type=float, default=None)
    parser.add_argument("--smoke-check", action="store_true")
    for field in ("use_stage3", "use_stage4", "use_region_tokens", "use_color_stats", "use_color_texture"):
        parser.add_argument(f"--{field.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--fusion",
        choices=["concat", "gate", "cross_attention", "region_attention", "colour_residual"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_check:
        smoke_check(torch.device(args.device))
        return
    config = preset_config(args.preset)
    for field in ("use_stage3", "use_stage4", "use_region_tokens", "use_color_stats", "use_color_texture"):
        value = getattr(args, field)
        if value is not None:
            config = replace(config, **{field: value})
    if args.fusion is not None:
        config = replace(config, fusion=args.fusion)
    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError("--dropout must be in [0, 1)")
        config = replace(config, dropout=args.dropout)
    if args.token_dim is not None:
        if args.token_dim <= 0 or args.token_dim % config.attention_heads:
            raise ValueError("--token-dim must be positive and divisible by the attention-head count")
        config = replace(config, token_dim=args.token_dim)
    default_emd, default_score = loss_weights_for_preset(args.preset)
    emd_weight = default_emd if args.emd_weight is None else args.emd_weight
    score_weight = default_score if args.score_weight is None else args.score_weight
    if args.experiment_name is None:
        args.experiment_name = args.preset if args.model == "rmof_efficientnet" else args.model
    run_training(args, config, emd_weight, score_weight)


if __name__ == "__main__":
    main()
