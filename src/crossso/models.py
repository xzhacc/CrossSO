from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossso.data import CLASS_NAMES, image_transform


def _load_legacy_checkpoint(path: Path) -> dict[str, Any]:
    try:
        from numpy._core.multiarray import scalar as numpy_scalar
    except ImportError:
        from numpy.core.multiarray import scalar as numpy_scalar
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    safe_types = [
        (defaultdict, "collections.defaultdict"),
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (ContainerMetadata, "omegaconf.base.ContainerMetadata"),
        (AnyNode, "omegaconf.nodes.AnyNode"),
        (dict, "builtins.dict"),
        (list, "builtins.list"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype(np.float64)),
        (ListConfig, "omegaconf.listconfig.ListConfig"),
        (Any, "typing.Any"),
        (int, "builtins.int"),
        (DictConfig, "omegaconf.dictconfig.DictConfig"),
        (Metadata, "omegaconf.base.Metadata"),
    ]
    with torch.serialization.safe_globals(safe_types):
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    return value


def load_payload(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(checkpoint, device="cpu")
    return _load_legacy_checkpoint(checkpoint)


def checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = load_payload(path)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid state_dict in {path}")
    return state


class DinoBackbone(nn.Module):
    def __init__(self, source: str | Path) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(str(source), local_files_only=True)
        self.model = AutoModel.from_config(config)

    def forward(self, images: torch.Tensor) -> Any:
        return self.model(pixel_values=images)


class PlainConvRouterHead(nn.Module):
    def __init__(self, in_channels: int = 770, dim: int = 512, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class Router(nn.Module):
    """DINOv2-reg-base Router with a convolutional head."""

    def __init__(self, dinov2_source: str | Path) -> None:
        super().__init__()
        self.backbone = DinoBackbone(dinov2_source)
        self.router_head = PlainConvRouterHead()
        self.prefix_tokens = 5

    @staticmethod
    def coordinates(features: torch.Tensor) -> torch.Tensor:
        h, w = features.shape[-2:]
        y = torch.linspace(-1, 1, h, device=features.device, dtype=features.dtype)
        x = torch.linspace(-1, 1, w, device=features.device, dtype=features.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy)).unsqueeze(0).expand(features.size(0), -1, -1, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(images)
        tokens = outputs.last_hidden_state[:, self.prefix_tokens :]
        side = math.isqrt(tokens.size(1))
        if side * side != tokens.size(1):
            raise ValueError(f"Non-square DINO patch token count: {tokens.size(1)}")
        features = tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), side, side)
        features = torch.cat((features, self.coordinates(features)), dim=1)
        logits = self.router_head(features)
        logits = F.interpolate(logits, (10, 10), mode="bilinear", align_corners=False)
        return logits[:, 0]

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        state = checkpoint_state(checkpoint)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        result = self.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Router state mismatch: {result}")


class GRAFT(nn.Module):
    """GRAFT NAIP encoder wrapper."""

    def __init__(self, clip_source: str | Path) -> None:
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(
            str(clip_source), local_files_only=True
        )
        config = self.satellite_image_backbone.config
        self.projector = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.projection_dim, bias=False),
        )
        self.patch_mlp = nn.Identity()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        vision = self.satellite_image_backbone.vision_model
        hidden = vision.pre_layrnorm(vision.embeddings(images))
        last = vision.encoder(inputs_embeds=hidden, output_hidden_states=False).last_hidden_state
        raw_patches = self.projector(self.patch_mlp(last[:, 1:]))
        pooled = vision.post_layernorm(last[:, 0])
        raw_cls = self.satellite_image_backbone.visual_projection(pooled)
        return {
            "normalized_patch_tokens": F.normalize(raw_patches, dim=-1),
            "normalized_cls": F.normalize(raw_cls, dim=-1),
        }

    def load_naip(self, checkpoint: str | Path) -> None:
        state = checkpoint_state(checkpoint)
        state.pop("satellite_image_backbone.vision_model.embeddings.position_ids", None)
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith("position_ids")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"GRAFT state mismatch: missing={missing}, unexpected={result.unexpected_keys}"
            )


class LearnablePosition(nn.Module):
    def __init__(self, count: int, dim: int = 512) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, count, dim))

    def forward(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.pos.to(device=device, dtype=dtype)


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    side = math.isqrt(tokens.size(1))
    if side * side != tokens.size(1):
        raise ValueError(f"Non-square token count: {tokens.size(1)}")
    return tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), side, side)


class TilePooler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=2, stride=2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.conv(tokens_to_grid(tokens)).flatten(2).transpose(1, 2).contiguous()


class TileUpsampleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(512)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _, d = tokens.shape
        grid = tokens.reshape(b * n, 7, 7, d).permute(0, 3, 1, 2)
        output = self.upsample(grid).permute(0, 2, 3, 1).reshape(b, n, 196, d)
        return self.norm(output)


class LRGuidedBlock(nn.Module):
    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(512, 8, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), lr, lr, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class LRGuidedPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LRGuidedBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(512)

    def forward(self, x: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, lr)
        return self.norm(x)


class PredictorCore(nn.Module):
    """Only the trained GRAFT predictor parameters; the GRAFT encoder is shared externally."""

    STATE_PREFIXES = (
        "lr_proj.", "tile_pos_embed.", "sub_pos_embed.", "lr_sub_pos_embed.",
        "mask_token", "predictor.", "hr_pooler.", "upsample_head.",
    )

    def __init__(self) -> None:
        super().__init__()
        self.lr_proj = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512),
            nn.Dropout(0.05),
        )
        self.tile_pos_embed = LearnablePosition(100)
        self.sub_pos_embed = LearnablePosition(49)
        self.lr_sub_pos_embed = LearnablePosition(4)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 512))
        self.predictor = LRGuidedPredictor()
        self.hr_pooler = TilePooler()
        self.upsample_head = TileUpsampleHead()

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        full = checkpoint_state(checkpoint)
        state: dict[str, torch.Tensor] = {}
        for key, value in full.items():
            normalized = key.removeprefix("core.")
            if normalized == "mask_token" or normalized.startswith(self.STATE_PREFIXES):
                state[normalized] = value
        result = self.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Predictor state mismatch: {result}")

    def lr_context_and_summary(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = tokens_to_grid(patch_tokens)
        pooled = F.adaptive_avg_pool2d(grid, (20, 20))
        pooled = pooled.reshape(pooled.size(0), 512, 10, 2, 10, 2)
        pooled = pooled.permute(0, 2, 4, 3, 5, 1).reshape(pooled.size(0), 100, 4, 512)
        pooled = self.lr_proj(pooled)
        summary = pooled.mean(dim=2)
        tile_pos = self.tile_pos_embed(device=pooled.device, dtype=pooled.dtype)
        sub_pos = self.lr_sub_pos_embed(device=pooled.device, dtype=pooled.dtype)
        pooled = pooled + tile_pos[:, :, None] + sub_pos[:, None]
        flat = pooled.reshape(pooled.size(0), 400, 512)
        context = torch.cat((flat.mean(1, keepdim=True), flat), dim=1)
        return context, summary

    def lr_context(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        context, _ = self.lr_context_and_summary(patch_tokens)
        return context

    def forward(
        self,
        selected_pooled: list[dict[int, torch.Tensor]],
        lr_context: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(selected_pooled)
        dtype, device = lr_context.dtype, lr_context.device
        z = self.mask_token.to(dtype=dtype).expand(batch, 100, 49, 512).clone()
        for batch_index, sample in enumerate(selected_pooled):
            for tile_index, value in sample.items():
                z[batch_index, int(tile_index)] = value.to(device=device, dtype=dtype)
        tile_pos = self.tile_pos_embed(device=device, dtype=dtype)
        sub_pos = self.sub_pos_embed(device=device, dtype=dtype)
        z = z + tile_pos[:, :, None] + sub_pos[:, None]
        latent = self.predictor(z.reshape(batch, 4900, 512), lr_context)
        return self.upsample_head(latent.reshape(batch, 100, 49, 512))


class ClasswisePcPolicy(nn.Module):
    """Persistent, non-trainable per-class HR acquisition counts."""

    def __init__(
        self,
        class_names: Sequence[str],
        values: Sequence[int],
        *,
        default_pc: int = 12,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in class_names)
        counts = tuple(int(value) for value in values)
        if names != tuple(CLASS_NAMES):
            raise ValueError("Pc policy class order must exactly match CLASS_NAMES")
        if len(counts) != len(names):
            raise ValueError(f"Expected {len(names)} Pc values, got {len(counts)}")
        if any(value < 0 or value > 100 for value in counts):
            raise ValueError("Every Pc value must be in [0, 100]")
        if not 0 <= int(default_pc) <= 100:
            raise ValueError("default_pc must be in [0, 100]")
        self.class_names = names
        self.default_pc = int(default_pc)
        self.register_buffer(
            "values",
            torch.tensor(counts, dtype=torch.int64),
            persistent=True,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ClasswisePcPolicy":
        from safetensors import safe_open

        source = Path(path).expanduser().resolve()
        with safe_open(source, framework="pt", device="cpu") as handle:
            serialized = (handle.metadata() or {}).get("pc_policy")
        if serialized is None:
            raise ValueError(f"Predictor is missing Pc policy metadata: {source}")
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise ValueError(f"Pc policy must be a JSON object: {source}")
        rows = payload.get("per_class")
        if not isinstance(rows, list):
            raise ValueError("Pc policy requires a per_class list")
        names = [row.get("name") for row in rows]
        values = [row.get("Pc") for row in rows]
        policy = cls(
            names,
            values,
            default_pc=int(payload.get("default_pc", 12)),
        )
        return policy

    @property
    def total_pc(self) -> int:
        return int(self.values.sum().item())

    @property
    def obr(self) -> float:
        return self.total_pc / (100.0 * len(self.class_names))

    def resolve(self, query_class: int | str | None) -> int:
        if isinstance(query_class, int):
            if not 0 <= query_class < len(self.class_names):
                raise IndexError(query_class)
            return int(self.values[query_class].item())
        if isinstance(query_class, str) and query_class in self.class_names:
            return int(self.values[self.class_names.index(query_class)].item())
        return self.default_pc

    def select(self, router_probs: torch.Tensor, query_class: int | str | None) -> list[int]:
        scores = router_probs.flatten()
        if scores.numel() != 100:
            raise ValueError(f"Router must emit 100 probabilities, got {scores.numel()}")
        pc = self.resolve(query_class)
        if pc == 0:
            return []
        return [int(index) for index in scores.argsort(descending=True)[:pc].tolist()]

    def combine_class_scores(
        self,
        scores_by_pc: torch.Tensor,
        counts: Sequence[int],
    ) -> torch.Tensor:
        """Select each fixed class score from the row for its configured Pc."""
        if scores_by_pc.ndim != 2 or scores_by_pc.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected [num_counts, {len(self.class_names)}] scores, "
                f"got {tuple(scores_by_pc.shape)}"
            )
        lookup = {int(value): index for index, value in enumerate(counts)}
        missing = sorted(set(int(value) for value in self.values.tolist()) - set(lookup))
        if missing:
            raise ValueError(f"Missing Pc score rows: {missing}")
        return torch.stack(
            [scores_by_pc[lookup[int(pc)], index] for index, pc in enumerate(self.values.tolist())]
        )


@dataclass
class GraftAnchorFusionConfig:
    graft_naip_checkpoint: Path
    predictor_checkpoint: Path
    predictor_pool: str
    normalization: str
    weight: float
    input_group: str
    output_group: str


def load_graft_anchor_fusion_config(
    raw: Mapping[str, Any] | None,
    *,
    root: Path,
) -> GraftAnchorFusionConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("cross_scale_fusion must be a mapping")
    def resolve(name: str) -> Path:
        if name not in raw:
            raise ValueError(f"cross_scale_fusion requires {name}")
        value = Path(str(raw[name])).expanduser()
        return value if value.is_absolute() else (root / value).resolve()

    predictor_pool = str(raw.get("predictor_pool", "raw_feature_max"))
    if predictor_pool not in {
        "patch_score_max",
        "raw_feature_max",
        "raw_feature_mean",
        "raw_feature_mean_max",
    }:
        raise ValueError(f"Unsupported GRAFT anchor predictor_pool: {predictor_pool}")
    normalization = str(raw.get("normalization", "row_zscore"))
    if normalization not in {"none", "row_zscore", "row_l2"}:
        raise ValueError(f"Unsupported GRAFT anchor normalization: {normalization}")
    weight = float(raw.get("weight", 0.0))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("cross_scale_fusion.weight must be in [0, 1]")
    return GraftAnchorFusionConfig(
        graft_naip_checkpoint=resolve("graft_naip_checkpoint"),
        predictor_checkpoint=resolve("predictor_checkpoint"),
        predictor_pool=predictor_pool,
        normalization=normalization,
        weight=weight,
        input_group=str(raw.get("input_group", "graft")),
        output_group=str(raw.get("output_group", "graft_anchor_fused")),
    )


def normalize_class_scores(
    scores: torch.Tensor,
    method: str,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    values = scores.float()
    if method == "none":
        return values
    if method == "row_l2":
        return F.normalize(values, dim=-1, eps=eps)
    if method == "row_zscore":
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True).clamp_min(eps)
        return (values - mean) / std
    raise ValueError(f"Unsupported class-score normalization: {method}")


def fuse_graft_anchor_scores(
    graft_scores: torch.Tensor,
    predictor_scores: torch.Tensor,
    *,
    weight: float,
    normalization: str,
) -> torch.Tensor:
    if graft_scores.shape != predictor_scores.shape:
        raise ValueError(
            "GRAFT anchor score shapes differ: "
            f"{tuple(graft_scores.shape)} != {tuple(predictor_scores.shape)}"
        )
    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValueError("GRAFT anchor fusion weight must be in [0, 1]")
    graft = normalize_class_scores(graft_scores, normalization)
    predicted = normalize_class_scores(predictor_scores, normalization)
    return (1.0 - value) * graft + value * predicted


class GraftZeroHRCompletion(nn.Module):
    """GRAFT completion branch used with an empty HR observation set."""

    def __init__(
        self,
        *,
        clip_source: str | Path,
        graft_naip_checkpoint: str | Path,
        predictor_checkpoint: str | Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.graft = GRAFT(clip_source).to(device)
        self.graft.load_naip(graft_naip_checkpoint)
        self.predictor = PredictorCore().to(device)
        self.predictor.load_checkpoint(predictor_checkpoint)
        self.device = device
        self.transform = image_transform(224)
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def completed_features(self, images: torch.Tensor) -> torch.Tensor:
        value = images.to(self.device, non_blocking=True)
        autocast = self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast,
        ):
            patches = self.graft(value)["normalized_patch_tokens"]
            context = self.predictor.lr_context(patches)
            predicted = self.predictor([{} for _ in range(value.size(0))], context)
        return predicted.float()

    @staticmethod
    def class_scores(
        predicted: torch.Tensor,
        text_features: torch.Tensor,
        *,
        pool: str,
    ) -> torch.Tensor:
        batch = predicted.size(0)
        flat = predicted.reshape(batch, -1, predicted.size(-1)).float()
        text = F.normalize(text_features.float(), dim=-1)
        if pool == "patch_score_max":
            patches = F.normalize(flat, dim=-1)
            return (patches @ text.t()).max(dim=1).values
        if pool == "raw_feature_max":
            pooled = flat.max(dim=1).values
        elif pool == "raw_feature_mean":
            pooled = flat.mean(dim=1)
        elif pool == "raw_feature_mean_max":
            pooled = 0.5 * (flat.mean(dim=1) + flat.max(dim=1).values)
        else:
            raise ValueError(f"Unsupported GRAFT predictor pool: {pool}")
        return F.normalize(pooled, dim=-1) @ text.t()


__all__ = [
    "GraftAnchorFusionConfig",
    "GraftZeroHRCompletion",
    "fuse_graft_anchor_scores",
    "load_graft_anchor_fusion_config",
    "normalize_class_scores",
]
