"""Zero-shot cross-scale evaluation utilities."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from crossso.data import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, labels_from_row, resolve_path
from crossso.metrics import (
    multilabel_average_precision,
    retrieval_map_at_ks,
)
from crossso.models import ClasswisePcPolicy, Router


PROMPT_NAMES = tuple(
    "amfootball" if index == 2 else name
    for index, name in enumerate(CLASS_NAMES)
)


def extrapolate_hr_residual_scores(
    observed_scores: torch.Tensor,
    zero_hr_scores: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    """Apply s_final = s_Pc + gain * (s_Pc - s_0HR)."""
    return observed_scores + float(gain) * (observed_scores - zero_hr_scores)


@dataclass
class ZeroShotConfig:
    data_root: Path
    checkpoint: Path
    router_checkpoint: Path
    dinov2_source: Path
    clip_text_source: Path
    image_size: int
    hidden_dim: int
    tile_pool_k: int
    lr_tile_k: int
    layers: int
    head_dim: int
    mlp_ratio: float
    dropout: float
    router_image_size: int
    prompt_template: str
    hr_residual_gain: float

def _resolved(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_zero_shot_config(path: str | Path) -> ZeroShotConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Zero-shot config must be a mapping: {config_path}")
    base = config_path.parent.parent
    paths = raw["paths"]
    model = raw["model"]
    router = raw["router"]
    protocol = raw["protocol"]
    scoring = raw["scoring"]
    return ZeroShotConfig(
        data_root=_resolved(base, paths["data_root"]),
        checkpoint=_resolved(base, paths["checkpoint"]),
        router_checkpoint=_resolved(base, paths["checkpoint_router"]),
        dinov2_source=_resolved(base, paths["dinov2_source"]),
        clip_text_source=_resolved(base, paths["clip_text_source"]),
        image_size=int(model["image_size"]),
        hidden_dim=int(model["hidden_dim"]),
        tile_pool_k=int(model["tile_pool_k"]),
        lr_tile_k=int(model["lr_tile_k"]),
        layers=int(model["layers"]),
        head_dim=int(model["head_dim"]),
        mlp_ratio=float(model["mlp_ratio"]),
        dropout=float(model["dropout"]),
        router_image_size=int(router["image_size"]),
        prompt_template=str(protocol["prompt_template"]),
        hr_residual_gain=float(scoring["hr_residual_gain"]),
    )


def _lr_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _hr_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _router_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FileHRTileProvider:
    """Load only the Router-selected HR tiles."""

    def __init__(self, paths: Sequence[str | Path], image_size: int = 260) -> None:
        if len(paths) != 100:
            raise ValueError("The HR provider requires exactly 100 tile paths")
        self.paths = [Path(path) for path in paths]
        self.transform = _hr_transform(image_size)
        self.loaded_indices: list[int] = []

    def load(self, indices: Iterable[int]) -> dict[int, torch.Tensor]:
        selected = [int(index) for index in indices]
        if len(set(selected)) != len(selected) or any(
            not 0 <= index < 100 for index in selected
        ):
            raise ValueError(f"Invalid HR indices: {selected}")
        self.loaded_indices.extend(selected)
        output: dict[int, torch.Tensor] = {}
        for index in selected:
            with Image.open(self.paths[index]) as source:
                output[index] = self.transform(source.convert("RGB"))
        return output


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a safetensors checkpoint: {path}")
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def _substate(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    marker = prefix + "."
    return {
        key[len(marker) :]: value
        for key, value in state.items()
        if key.startswith(marker)
    }


def _load_module(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    selected = _substate(state, prefix)
    if not selected:
        raise KeyError(f"Checkpoint has no {prefix} parameters")
    result = module.load_state_dict(selected, strict=False)
    missing = [
        key for key in result.missing_keys if not key.endswith("position_ids")
    ]
    unexpected = [
        key for key in result.unexpected_keys if not key.endswith("position_ids")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"{prefix} state mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def _build_dino(source: Path) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(str(source), local_files_only=True)


class LearnablePosition(nn.Module):
    def __init__(self, count: int, dim: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, count, dim))

    def forward(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.pos.to(device=device, dtype=dtype)


class LRGuidedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        lr_context: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm1(value)
        value = value + self.self_attn(
            query, query, query, need_weights=False
        )[0]
        query = self.norm2(value)
        value = value + self.cross_attn(
            query,
            lr_context,
            lr_context,
            need_weights=False,
        )[0]
        return value + self.mlp(self.norm3(value))


class LRGuidedPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LRGuidedBlock(dim, heads, mlp_ratio, dropout)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        value: torch.Tensor,
        lr_context: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, lr_context)
        return self.norm(value)


@dataclass
class PredictionResult:
    router_scores: torch.Tensor
    selected_indices: list[int]
    used_hr_ratio: float
    class_scores: torch.Tensor
    tile_scores: torch.Tensor
    predicted_features: torch.Tensor | None


class ZeroShotCrossSO(nn.Module):
    """Router and observation-guided vision-language Predictor."""

    def __init__(
        self,
        config: ZeroShotConfig,
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        self.config = config
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.pc_policy = ClasswisePcPolicy.from_checkpoint(config.checkpoint)
        self.encoder = _build_dino(config.dinov2_source)
        self.lr_encoder = _build_dino(config.dinov2_source)
        heads = config.hidden_dim // config.head_dim
        self.predictor = LRGuidedPredictor(
            config.hidden_dim,
            heads,
            config.layers,
            config.mlp_ratio,
            config.dropout,
        )
        self.lr_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.tile_pos_embed = LearnablePosition(100, config.hidden_dim)
        self.sub_pos_embed = LearnablePosition(
            config.tile_pool_k**2,
            config.hidden_dim,
        )
        self.lr_sub_pos_embed = LearnablePosition(
            config.lr_tile_k**2,
            config.hidden_dim,
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, config.hidden_dim)
        )
        self._router: Router | None = None

        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.clip_text_source),
            local_files_only=True,
        )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            str(config.clip_text_source),
            local_files_only=True,
        )
        self.siglip_visual_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 512),
        )
        self.siglip_logit_scale = nn.Parameter(torch.tensor(2.6593))
        self.siglip_logit_bias = nn.Parameter(torch.tensor(0.0))
        self._load_checkpoint(config.checkpoint)
        self.to(self.device).eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.lr_transform = _lr_transform(config.image_size)
        self.router_transform = _router_transform(config.router_image_size)
        self.hr_transform = _hr_transform(config.image_size)
        self._class_text: torch.Tensor | None = None

    def _load_checkpoint(self, path: Path) -> None:
        payload = _load_payload(path)
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise TypeError(f"Invalid state_dict: {path}")
        _load_module(self.predictor, state, "predictor")
        _load_module(self.lr_proj, state, "lr_proj")
        _load_module(self.tile_pos_embed, state, "tile_pos_embed")
        _load_module(self.sub_pos_embed, state, "sub_pos_embed")
        _load_module(self.lr_sub_pos_embed, state, "lr_sub_pos_embed")
        _load_module(self.siglip_visual_proj, state, "siglip_visual_proj")
        self.mask_token.data.copy_(state["mask_token"])
        self.siglip_logit_scale.data.copy_(state["siglip_logit_scale"])
        self.siglip_logit_bias.data.copy_(state["siglip_logit_bias"])

    def _prepare(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        transform: transforms.Compose,
    ) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                value = transform(source.convert("RGB"))
        elif isinstance(image, Image.Image):
            value = transform(image.convert("RGB"))
        elif torch.is_tensor(image):
            value = image
            if value.ndim == 4:
                if value.size(0) != 1:
                    raise ValueError("predict accepts one LR image")
                return value.to(self.device)
        else:
            raise TypeError(f"Unsupported LR image type: {type(image)!r}")
        return value.unsqueeze(0).to(self.device)

    def _prepare_lr(
        self,
        image: str | Path | Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        return self._prepare(image, self.lr_transform)

    def _prepare_router_lr(
        self,
        image: str | Path | Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        return self._prepare(image, self.router_transform)

    @torch.inference_mode()
    def encode_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        output = self.text_encoder(**tokens, return_dict=True)
        return F.normalize(output.text_embeds.float(), dim=-1)

    def class_prompts(self) -> list[str]:
        return [
            self.config.prompt_template.format(labels=name)
            for name in PROMPT_NAMES
        ]

    @torch.inference_mode()
    def class_text_features(self) -> torch.Tensor:
        if self._class_text is None or self._class_text.device != self.device:
            self._class_text = self.encode_texts(self.class_prompts())
        return self._class_text

    @torch.inference_mode()
    def _lr_context(
        self,
        lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.lr_encoder(
            pixel_values=lr,
            return_dict=True,
        ).last_hidden_state
        register_count = int(
            getattr(self.lr_encoder.config, "num_register_tokens", 0)
        )
        patches = output[:, 1 + register_count :].float()
        side = math.isqrt(patches.size(1))
        grid = patches.transpose(1, 2).reshape(
            lr.size(0),
            patches.size(-1),
            side,
            side,
        )
        k = self.config.lr_tile_k
        pooled = F.adaptive_avg_pool2d(grid, (10 * k, 10 * k))
        pooled = pooled.reshape(
            lr.size(0),
            self.config.hidden_dim,
            10,
            k,
            10,
            k,
        )
        pooled = pooled.permute(0, 2, 4, 3, 5, 1).reshape(
            lr.size(0),
            100,
            k * k,
            -1,
        )
        projected = self.lr_proj(pooled)
        summary = projected.mean(dim=2)
        tile_position = self.tile_pos_embed(
            device=projected.device,
            dtype=projected.dtype,
        )
        sub_position = self.lr_sub_pos_embed(
            device=projected.device,
            dtype=projected.dtype,
        )
        projected = projected + tile_position[:, :, None] + sub_position[:, None]
        context = projected.reshape(lr.size(0), 100 * k * k, -1)
        return (
            torch.cat((context.mean(dim=1, keepdim=True), context), dim=1),
            summary,
        )

    @torch.inference_mode()
    def route(self, lr: torch.Tensor) -> torch.Tensor:
        if self._router is None:
            router = Router(self.config.dinov2_source)
            router.load_checkpoint(self.config.router_checkpoint)
            router.to(self.device).eval()
            for parameter in router.parameters():
                parameter.requires_grad = False
            self._router = router
        return self._router(lr).sigmoid()[0].flatten()

    @torch.inference_mode()
    def _encode_hr(self, images: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            pixel_values=images,
            return_dict=True,
        ).last_hidden_state
        register_count = int(
            getattr(self.encoder.config, "num_register_tokens", 0)
        )
        patches = output[:, 1 + register_count :].float()
        side = math.isqrt(patches.size(1))
        grid = patches.transpose(1, 2).reshape(
            images.size(0),
            patches.size(-1),
            side,
            side,
        )
        return F.adaptive_avg_pool2d(
            grid,
            (self.config.tile_pool_k, self.config.tile_pool_k),
        ).flatten(2).transpose(1, 2)

    @torch.inference_mode()
    def encode_selected_hr(
        self,
        provider: FileHRTileProvider,
        selected: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        ordered = [int(index) for index in selected]
        raw = provider.load(ordered)
        if not ordered:
            return {}
        images = torch.stack([raw[index] for index in ordered]).to(self.device)
        chunks = [
            self._encode_hr(images[start : start + 16])
            for start in range(0, len(ordered), 16)
        ]
        encoded = torch.cat(chunks)
        return {
            index: encoded[offset]
            for offset, index in enumerate(ordered)
        }

    @torch.inference_mode()
    def complete_encoded_batch(
        self,
        lr_context: torch.Tensor,
        encoded: dict[int, torch.Tensor],
        selections: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        batch = len(selections)
        if batch < 1:
            raise ValueError("At least one selection is required")
        shape = (
            batch,
            100,
            self.config.tile_pool_k**2,
            self.config.hidden_dim,
        )
        value = self.mask_token.expand(shape).clone()
        for batch_index, selected in enumerate(selections):
            if selected:
                indices = torch.tensor(selected, device=self.device)
                features = torch.stack(
                    [encoded[int(index)] for index in selected]
                )
                value[batch_index, indices] = features.to(value.dtype)
        tile_position = self.tile_pos_embed(
            device=value.device,
            dtype=value.dtype,
        )
        sub_position = self.sub_pos_embed(
            device=value.device,
            dtype=value.dtype,
        )
        value = value + tile_position[:, :, None] + sub_position[:, None]
        context = lr_context.expand(batch, -1, -1)
        predicted = self.predictor(
            value.reshape(batch, -1, self.config.hidden_dim),
            context,
        )
        return predicted.reshape(*shape)

    @torch.inference_mode()
    def score(
        self,
        predicted: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = predicted.size(0)
        projected = self.siglip_visual_proj(
            predicted.reshape(-1, predicted.size(-1)).float()
        )
        projected = F.normalize(projected, dim=-1)
        patch_scores = projected @ text.t()
        patch_scores = patch_scores.reshape(
            batch,
            100,
            self.config.tile_pool_k**2,
            text.size(0),
        )
        scale = self.siglip_logit_scale.clamp(max=math.log(100)).exp()
        patch_scores = patch_scores * scale + self.siglip_logit_bias
        tile_scores = patch_scores.max(dim=2).values
        return tile_scores.max(dim=1).values, tile_scores

    @torch.inference_mode()
    def score_lr_summary(
        self,
        lr_summary: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if lr_summary.ndim != 3 or lr_summary.size(1) != 100:
            raise ValueError(
                f"Expected LR summary [batch,100,dim], "
                f"got {tuple(lr_summary.shape)}"
            )
        projected = self.siglip_visual_proj(lr_summary.float())
        projected = F.normalize(projected, dim=-1)
        tile_scores = projected @ text.t()
        scale = self.siglip_logit_scale.clamp(max=math.log(100)).exp()
        tile_scores = tile_scores * scale + self.siglip_logit_bias
        pooled = {"mean": tile_scores.mean(dim=1)}
        return pooled, tile_scores

    def apply_hr_residual_gain(
        self,
        observed_scores: torch.Tensor,
        zero_hr_scores: torch.Tensor,
    ) -> torch.Tensor:
        return extrapolate_hr_residual_scores(
            observed_scores,
            zero_hr_scores,
            self.config.hr_residual_gain,
        )

    @torch.inference_mode()
    def predict_pc_query(
        self,
        lr_image: str | Path | Image.Image | torch.Tensor,
        hr_provider: FileHRTileProvider,
        query_class: int | str | None,
        *,
        texts: Sequence[str] | None = None,
        return_features: bool = False,
    ) -> PredictionResult:
        lr = self._prepare_lr(lr_image)
        lr_context, _ = self._lr_context(lr)
        router_scores = self.route(self._prepare_router_lr(lr_image))
        selected = self.pc_policy.select(router_scores, query_class)
        encoded = self.encode_selected_hr(hr_provider, selected)
        predicted_pair = self.complete_encoded_batch(
            lr_context,
            encoded,
            [selected, []],
        )
        text = (
            self.class_text_features()
            if texts is None
            else self.encode_texts(texts)
        )
        class_score_pair, tile_score_pair = self.score(predicted_pair, text)
        return PredictionResult(
            router_scores=router_scores.float().cpu(),
            selected_indices=selected,
            used_hr_ratio=len(selected) / 100.0,
            class_scores=self.apply_hr_residual_gain(
                class_score_pair[0],
                class_score_pair[1],
            ).float().cpu(),
            tile_scores=self.apply_hr_residual_gain(
                tile_score_pair[0],
                tile_score_pair[1],
            ).float().cpu(),
            predicted_features=(
                predicted_pair[0].float().cpu()
                if return_features
                else None
            ),
        )

    @torch.inference_mode()
    def predict_classwise_pc_score_components(
        self,
        lr_image: str | Path | Image.Image | torch.Tensor,
        hr_provider: FileHRTileProvider,
        *,
        count_batch_size: int = 6,
    ) -> dict[str, torch.Tensor | int]:
        """Score all classes while encoding the largest Router prefix once."""
        lr = self._prepare_lr(lr_image)
        lr_context, lr_summary = self._lr_context(lr)
        probabilities = self.route(self._prepare_router_lr(lr_image))
        ranked = [
            int(index)
            for index in probabilities.argsort(descending=True).tolist()
        ]
        counts = sorted(
            {0, *(int(value) for value in self.pc_policy.values.tolist())}
        )
        selections = [ranked[:count] for count in counts]
        encoded = self.encode_selected_hr(hr_provider, ranked[: max(counts)])
        score_chunks: list[torch.Tensor] = []
        tile_score_chunks: list[torch.Tensor] = []
        step = max(1, int(count_batch_size))
        text = self.class_text_features()
        for start in range(0, len(counts), step):
            predicted = self.complete_encoded_batch(
                lr_context,
                encoded,
                selections[start : start + step],
            )
            class_scores, tile_scores = self.score(predicted, text)
            score_chunks.append(class_scores)
            tile_score_chunks.append(tile_scores)
        scores_by_pc = torch.cat(score_chunks)
        tile_scores_by_pc = torch.cat(tile_score_chunks)
        observed_scores = self.pc_policy.combine_class_scores(
            scores_by_pc,
            counts,
        )
        zero_hr_scores = scores_by_pc[counts.index(0)]
        predictor_scores = self.apply_hr_residual_gain(
            observed_scores,
            zero_hr_scores,
        )
        count_lookup = {
            int(value): index for index, value in enumerate(counts)
        }
        observed_tile_scores = torch.stack(
            [
                tile_scores_by_pc[
                    count_lookup[int(pc)],
                    :,
                    class_index,
                ]
                for class_index, pc in enumerate(
                    self.pc_policy.values.tolist()
                )
            ],
            dim=1,
        )
        zero_hr_tile_scores = tile_scores_by_pc[counts.index(0)]
        predictor_tile_scores = self.apply_hr_residual_gain(
            observed_tile_scores,
            zero_hr_tile_scores,
        )
        lr_scores, lr_tile_scores = self.score_lr_summary(lr_summary, text)
        return {
            "pred_hr_scores": predictor_scores,
            "pred_hr_tile_scores": predictor_tile_scores,
            "zero_hr_scores": zero_hr_scores,
            "lr_mean_scores": lr_scores["mean"][0],
            "lr_tile_scores": lr_tile_scores[0],
            "router_scores": probabilities,
            "maximum_union_pc": int(max(counts)),
        }

    @torch.inference_mode()
    def full_target_features(
        self,
        paths: Sequence[str | Path],
    ) -> torch.Tensor:
        provider = FileHRTileProvider(paths, self.config.image_size)
        raw = provider.load(range(100))
        chunks = []
        for start in range(0, 100, 16):
            images = torch.stack(
                [
                    raw[index]
                    for index in range(start, min(100, start + 16))
                ]
            )
            chunks.append(self._encode_hr(images.to(self.device)).cpu())
        return torch.cat(chunks)


def _row_paths(
    row: pd.Series,
    root: Path,
) -> tuple[Path, list[Path]]:
    lr = resolve_path(root, str(row["lr_path"]))
    hr = [
        resolve_path(root, str(row[f"hr_{index}"]))
        for index in range(100)
    ]
    return lr, hr


@dataclass
class GL10MConfig:
    path: Path
    raw: dict[str, Any]


def load_gl10m_config(path: str | Path) -> GL10MConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"GL-10M config must be a mapping: {config_path}")
    return GL10MConfig(path=config_path, raw=raw)


def _resolve_gl10m_path(value: str | Path, image_root: Path) -> Path:
    """Map a manifest path to the downloaded region/LR/file hierarchy."""
    source = Path(value).expanduser()
    if len(source.parts) < 3:
        raise FileNotFoundError(f"Cannot map GL-10M path: {source}")
    candidate = image_root.joinpath(*source.parts[-3:])
    if candidate.is_file():
        return candidate
    if source.is_file():
        return source
    raise FileNotFoundError(f"Missing GL-10M image: {candidate} (from {source})")


def _gl10m_labels(row: pd.Series, class_count: int) -> torch.Tensor:
    labels = torch.zeros(class_count, dtype=torch.uint8)
    columns = sorted(
        (name for name in row.index if str(name).startswith("label_")),
        key=lambda value: int(str(value).split("_")[-1]),
    )
    for column in columns:
        value = row[column]
        if pd.isna(value):
            continue
        index = int(value)
        if not 0 <= index < class_count:
            raise ValueError(f"GL-10M label {index} is outside [0, {class_count})")
        labels[index] = 1
    return labels


class _GL10MDataset(Dataset):
    """Loads only LR imagery; selected HR tiles are opened after routing."""

    def __init__(
        self,
        frame: pd.DataFrame,
        indices: list[int],
        *,
        image_root: Path,
        class_count: int,
        lr_transform: Any,
        router_transform: Any,
    ) -> None:
        self.frame = frame
        self.indices = indices
        self.image_root = image_root
        self.class_count = class_count
        self.lr_transform = lr_transform
        self.router_transform = router_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int) -> tuple[Any, ...]:
        frame_index = self.indices[offset]
        row = self.frame.iloc[frame_index]
        path = _resolve_gl10m_path(str(row["filepath"]), self.image_root)
        with Image.open(path) as source:
            image = source.convert("RGB")
            lr = self.lr_transform(image)
            router_lr = self.router_transform(image)
        return (
            lr,
            router_lr,
            _gl10m_labels(row, self.class_count),
            frame_index,
            str(path),
        )


def _gl10m_lr_context_and_summary(
    model: ZeroShotCrossSO, lr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model.lr_encoder(pixel_values=lr, return_dict=True).last_hidden_state
    register_count = int(getattr(model.lr_encoder.config, "num_register_tokens", 0))
    patches = output[:, 1 + register_count :].float()
    side = math.isqrt(patches.size(1))
    if side * side != patches.size(1):
        raise ValueError(f"Non-square GL-10M LR token count: {patches.size(1)}")
    grid = patches.transpose(1, 2).reshape(lr.size(0), patches.size(-1), side, side)
    k = model.config.lr_tile_k
    pooled = F.adaptive_avg_pool2d(grid, (10 * k, 10 * k))
    pooled = pooled.reshape(lr.size(0), model.config.hidden_dim, 10, k, 10, k)
    pooled = pooled.permute(0, 2, 4, 3, 5, 1).reshape(
        lr.size(0), 100, k * k, model.config.hidden_dim
    )
    projected = model.lr_proj(pooled)
    summary = projected.mean(dim=2)
    tile_position = model.tile_pos_embed(device=projected.device, dtype=projected.dtype)
    sub_position = model.lr_sub_pos_embed(device=projected.device, dtype=projected.dtype)
    projected = projected + tile_position[:, :, None] + sub_position[:, None]
    context = projected.reshape(lr.size(0), 100 * k * k, model.config.hidden_dim)
    context = torch.cat((context.mean(dim=1, keepdim=True), context), dim=1)
    return context, summary


def _gl10m_hr_directory(lr_path: str | Path) -> Path:
    path = Path(lr_path)
    return path.parent.parent / "HR" / path.stem


def _load_gl10m_hr(path: Path, transform: Any) -> torch.Tensor:
    with Image.open(path) as source:
        return transform(source.convert("RGB"))


def _complete_gl10m_batch(
    model: ZeroShotCrossSO,
    lr_context: torch.Tensor,
    lr_paths: list[str],
    selections: list[list[int]],
    *,
    hr_workers: int,
    hr_encode_batch: int,
) -> torch.Tensor:
    """Open and encode exactly the HR tiles selected by the LR-only Router."""
    requested = [
        (
            sample_index,
            tile_index,
            _gl10m_hr_directory(lr_paths[sample_index]) / f"{tile_index}.jpg",
        )
        for sample_index, selected in enumerate(selections)
        for tile_index in selected
    ]
    with ThreadPoolExecutor(max_workers=max(1, hr_workers)) as executor:
        images = list(
            executor.map(
                lambda item: _load_gl10m_hr(item[2], model.hr_transform), requested
            )
        )

    batch = len(selections)
    shape = (batch, 100, model.config.tile_pool_k**2, model.config.hidden_dim)
    value = model.mask_token.expand(shape).clone()
    if images:
        encoded = []
        stacked = torch.stack(images)
        for start in range(0, stacked.size(0), hr_encode_batch):
            encoded.append(
                model._encode_hr(
                    stacked[start : start + hr_encode_batch].to(model.device)
                )
            )
        encoded_tensor = torch.cat(encoded)
        sample_indices = torch.tensor([item[0] for item in requested], device=model.device)
        tile_indices = torch.tensor([item[1] for item in requested], device=model.device)
        value[sample_indices, tile_indices] = encoded_tensor.to(value.dtype)

    tile_position = model.tile_pos_embed(device=value.device, dtype=value.dtype)
    sub_position = model.sub_pos_embed(device=value.device, dtype=value.dtype)
    value = value + tile_position[:, :, None] + sub_position[:, None]
    prediction = model.predictor(
        value.reshape(batch, -1, model.config.hidden_dim), lr_context
    )
    return prediction.reshape(shape)


__all__ = ["GL10MConfig", "load_gl10m_config"]


FUSION_CACHE_FORMAT = "crossso-cross-scale-score-cache-v1"
FUSION_REPORT_FORMAT = "crossso-lr-fusion-evaluation-v1"
PREDICTOR_REPORT_FORMAT = "crossso-predictor-evaluation-v1"
POOLINGS = ("mean",)


@dataclass
class LRFusionConfig:
    path: Path
    root: Path
    cache_root: Path
    zero_shot_model_config: Path
    gl10m_config: Path
    artifact: Path

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


@dataclass
class CacheBundle:
    task: str
    split: str
    sample_indices: torch.Tensor
    targets: torch.Tensor
    pred_hr_scores: torch.Tensor
    lr_scores: dict[str, torch.Tensor]
    selected_counts: torch.Tensor | None
    used_hr_ratio: float
    class_names: tuple[str, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_lr_fusion_config(path: str | Path) -> LRFusionConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"LR fusion config must be a mapping: {config_path}")
    root = config_path.parent.parent
    paths = raw.get("paths", {})
    required = ("cache_root", "gl10m_config", "checkpoint")
    missing = [name for name in required if not paths.get(name)]
    if missing:
        raise ValueError(f"LR fusion config is missing paths: {', '.join(missing)}")
    return LRFusionConfig(
        path=config_path,
        root=root,
        cache_root=_resolve(root, paths["cache_root"]),
        zero_shot_model_config=config_path,
        gl10m_config=_resolve(root, paths["gl10m_config"]),
        artifact=_resolve(root, paths["checkpoint"]),
    )


def robust_mad_tail_fuse_scores(
    pred_hr_scores: torch.Tensor,
    lr_scores: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    *,
    positive_weight: float,
    negative_weight: float,
    tail_threshold: float,
) -> torch.Tensor:
    """Add confident LR residuals while retaining the HR score scale."""
    if pred_hr_scores.shape != lr_scores.shape or pred_hr_scores.ndim != 2:
        raise ValueError("Robust tail fusion requires matching [samples,classes] tensors")
    positive = float(positive_weight)
    negative = float(negative_weight)
    threshold = float(tail_threshold)
    if positive < 0 or negative < 0:
        raise ValueError("Robust tail weights must be non-negative")
    if threshold < 0:
        raise ValueError("Robust tail threshold must be non-negative")
    required = ("hr_mad", "lr_median", "lr_mad")
    missing = [name for name in required if name not in statistics]
    if missing:
        raise ValueError(f"Robust score statistics are missing: {missing}")
    for name in required:
        value = statistics[name]
        if value.shape != (pred_hr_scores.size(1),):
            raise ValueError(
                f"Statistic {name} has shape {tuple(value.shape)}; "
                f"expected {(pred_hr_scores.size(1),)}"
            )
    normalized_lr = (
        lr_scores - statistics["lr_median"].to(lr_scores)
    ) / statistics["lr_mad"].to(lr_scores).clamp_min(1e-5)
    residual = positive * F.relu(normalized_lr - threshold)
    residual = residual - negative * F.relu(-normalized_lr - threshold)
    return pred_hr_scores + statistics["hr_mad"].to(pred_hr_scores) * residual


def _deserialize_score_statistics(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    required = ("hr_mad", "lr_median", "lr_mad")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Robust fusion artifact is missing statistics: {missing}")
    statistics = {
        name: torch.as_tensor(payload[name], dtype=torch.float32) for name in required
    }
    lengths = {int(value.numel()) for value in statistics.values()}
    if len(lengths) != 1:
        raise ValueError("Robust fusion statistics use inconsistent class counts")
    if not all(torch.isfinite(value).all() for value in statistics.values()):
        raise ValueError("Robust fusion statistics must be finite")
    return statistics


def _load_fusion_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            serialized = (handle.metadata() or {}).get("fusion")
        if serialized is None:
            raise ValueError(f"Predictor is missing fusion metadata: {source}")
        payload = json.loads(serialized)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected fusion artifact: {source}")
    return dict(payload)


@dataclass
class LRFusion:
    """Label-free score fusion for the configured vocabulary."""

    method: str
    pooling: str
    positive_weight: float
    negative_weight: float
    tail_threshold: float
    class_names: tuple[str, ...]
    statistics: dict[str, torch.Tensor]

    @classmethod
    def from_config(
        cls,
        config: str | Path | LRFusionConfig,
    ) -> "LRFusion":
        cfg = (
            load_lr_fusion_config(config)
            if not isinstance(config, LRFusionConfig)
            else config
        )
        payload = _load_fusion_artifact(cfg.artifact)
        if payload.get("method") != "robust_mad_tail":
            raise ValueError("Expected robust_mad_tail fusion")
        pooling = str(payload.get("pooling"))
        if pooling not in POOLINGS:
            raise ValueError(f"Unsupported LR pooling: {pooling!r}")
        serialized = payload.get("statistics")
        if not isinstance(serialized, dict):
            raise ValueError("LR fusion artifact is missing robust statistics")
        class_names = tuple(str(value) for value in serialized.get("class_names", ()))
        if not class_names:
            raise ValueError("LR fusion artifact is missing its class vocabulary")
        return cls(
            method="robust_mad_tail",
            pooling=pooling,
            positive_weight=float(payload["positive_weight"]),
            negative_weight=float(payload["negative_weight"]),
            tail_threshold=float(payload["tail_threshold"]),
            class_names=class_names,
            statistics=_deserialize_score_statistics(serialized),
        )

    def apply(
        self,
        pred_hr_scores: torch.Tensor,
        lr_scores: dict[str, torch.Tensor],
        *,
        class_names: Sequence[str],
    ) -> torch.Tensor:
        if tuple(str(value) for value in class_names) != self.class_names:
            raise ValueError(
                "LR fusion statistics require the configured class order"
            )
        if self.pooling not in lr_scores:
            raise ValueError(f"LR scores do not contain pooling {self.pooling!r}")
        predicted = pred_hr_scores.float()
        low_resolution = lr_scores[self.pooling].float()
        squeeze = predicted.ndim == 1
        if squeeze:
            predicted = predicted.unsqueeze(0)
            low_resolution = low_resolution.unsqueeze(0)
        fused = robust_mad_tail_fuse_scores(
            predicted,
            low_resolution,
            self.statistics,
            positive_weight=self.positive_weight,
            negative_weight=self.negative_weight,
            tail_threshold=self.tail_threshold,
        )
        return fused[0] if squeeze else fused

def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _load_cache_shard(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FUSION_CACHE_FORMAT:
        raise ValueError(f"Unexpected fusion cache: {source}")
    for key in ("sample_indices", "targets", "pred_hr_scores"):
        if not isinstance(payload.get(key), torch.Tensor):
            raise ValueError(f"Fusion cache {source} is missing tensor {key!r}")
    lr_scores = payload.get("lr_scores", {})
    if not isinstance(lr_scores, dict):
        raise ValueError(f"Fusion cache {source} has invalid LR score branches")
    if any(not isinstance(value, torch.Tensor) for value in lr_scores.values()):
        raise ValueError(f"Fusion cache {source} has a non-tensor LR score branch")
    return source, payload


def merge_caches(paths: Sequence[str | Path]) -> CacheBundle:
    if not paths:
        raise ValueError("At least one cross-scale score cache is required")
    loaded = [_load_cache_shard(path) for path in paths]
    shards = [value[1] for value in loaded]
    first = shards[0]
    task = str(first.get("task"))
    split = str(first.get("split"))
    class_names = tuple(str(value) for value in first.get("class_names", ()))
    lr_names = tuple(sorted(first["lr_scores"]))
    for shard in shards:
        if str(shard.get("task")) != task or str(shard.get("split")) != split:
            raise ValueError("Fusion cache shards use different tasks or splits")
        if tuple(str(value) for value in shard.get("class_names", ())) != class_names:
            raise ValueError("Fusion cache shards use different class names")
        if tuple(sorted(shard["lr_scores"])) != lr_names:
            raise ValueError("Fusion cache shards expose different LR pooling branches")
    sample_indices = torch.cat([value["sample_indices"].long() for value in shards])
    if sample_indices.numel() != torch.unique(sample_indices).numel():
        raise ValueError("Fusion cache shards contain duplicate sample indices")
    order = sample_indices.argsort()

    def merged(name: str) -> torch.Tensor:
        return torch.cat([value[name] for value in shards])[order].float()

    targets = torch.cat([value["targets"] for value in shards])[order]
    pred_hr_scores = merged("pred_hr_scores")
    lr_scores = {
        name: torch.cat([value["lr_scores"][name] for value in shards])[order].float()
        for name in lr_names
    }
    expected = targets.shape
    values = {
        "pred_hr_scores": pred_hr_scores,
        **{f"lr_scores.{key}": value for key, value in lr_scores.items()},
    }
    for name, value in values.items():
        if value.shape != expected:
            raise ValueError(
                f"Fusion cache shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(expected)}"
            )
    selected_values = [value.get("selected_counts") for value in shards]
    selected_counts: torch.Tensor | None
    if all(isinstance(value, torch.Tensor) for value in selected_values):
        selected_counts = torch.cat(selected_values)[order].long()
        used_hr_ratio = float(selected_counts.float().mean() / 100)
    elif any(value is not None for value in selected_values):
        raise ValueError("Only some fusion cache shards contain selected_counts")
    else:
        selected_counts = None
        ratios = [float(value.get("used_hr_ratio", float("nan"))) for value in shards]
        if any(not math.isfinite(value) for value in ratios):
            raise ValueError("Fusion cache lacks both selected_counts and used_hr_ratio")
        if max(ratios) - min(ratios) > 1e-9:
            raise ValueError("Fusion cache shards disagree on used_hr_ratio")
        used_hr_ratio = ratios[0]
    return CacheBundle(
        task=task,
        split=split,
        sample_indices=sample_indices[order],
        targets=targets,
        pred_hr_scores=pred_hr_scores,
        lr_scores=lr_scores,
        selected_counts=selected_counts,
        used_hr_ratio=used_hr_ratio,
        class_names=class_names,
    )


def _require_full_coverage(cache: CacheBundle, expected_count: int) -> None:
    expected = torch.arange(int(expected_count), dtype=cache.sample_indices.dtype)
    if cache.sample_indices.shape != expected.shape or not torch.equal(
        cache.sample_indices.cpu(),
        expected,
    ):
        raise ValueError(
            f"{cache.task} {cache.split} cache must cover every sample exactly once"
        )


def _retrieval_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, float]:
    values = retrieval_map_at_ks(scores[:, indices], targets[:, indices], (20, 100))
    return {"mAP@20": float(values["mAP@20"]), "mAP@100": float(values["mAP@100"])}


def _gl10m_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    values = multilabel_average_precision(scores, targets)
    return {
        "support_weighted_mAP": float(values["support_weighted"]),
    }


def _apply_artifact(
    cache: CacheBundle,
    artifact: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    method = str(artifact.get("method"))
    pooling = str(artifact.get("pooling"))
    if pooling not in cache.lr_scores:
        raise ValueError(f"Fusion artifact requests unavailable pooling {pooling!r}")
    if method != "robust_mad_tail":
        raise ValueError(f"Unsupported fusion method: {method}")
    artifact_statistics = dict(artifact.get("statistics", {}))
    artifact_class_names = tuple(
        str(value) for value in artifact_statistics.get("class_names", ())
    )
    if artifact_class_names != cache.class_names:
        raise ValueError("Fusion statistics use a different class vocabulary")
    statistics = _deserialize_score_statistics(artifact_statistics)
    positive = float(artifact["positive_weight"])
    negative = float(artifact["negative_weight"])
    threshold = float(artifact["tail_threshold"])
    scores = robust_mad_tail_fuse_scores(
        cache.pred_hr_scores,
        cache.lr_scores[pooling],
        statistics,
        positive_weight=positive,
        negative_weight=negative,
        tail_threshold=threshold,
    )
    return scores, {
        "method": method,
        "pooling": pooling,
        "positive_weight": positive,
        "negative_weight": negative,
        "tail_threshold": threshold,
    }


def _full_retrieval_report(cache: CacheBundle, scores: torch.Tensor) -> dict[str, Any]:
    all_metrics = _retrieval_metrics(
        scores, cache.targets, torch.arange(cache.targets.size(1))
    )
    report: dict[str, Any] = {"all": all_metrics}
    if cache.targets.size(1) == 40:
        report["base"] = _retrieval_metrics(scores, cache.targets, torch.arange(30))
        report["novel"] = _retrieval_metrics(
            scores, cache.targets, torch.arange(30, 40)
        )
    return report


def evaluate_lr_fusion(
    config: str | Path | LRFusionConfig,
    *,
    inputs: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    cfg = (
        load_lr_fusion_config(config)
        if not isinstance(config, LRFusionConfig)
        else config
    )
    cache = merge_caches(inputs)
    if cache.task != "zero-shot":
        raise ValueError(f"Fusion evaluation requires zero-shot cache, got {cache.task}")
    model_config = load_zero_shot_config(cfg.zero_shot_model_config)
    frame = pd.read_csv(model_config.data_root / f"{cache.split}_loc_group.csv")
    _require_full_coverage(cache, len(frame))
    selection = _load_fusion_artifact(cfg.artifact)
    scores, fusion_summary = _apply_artifact(
        cache,
        selection,
    )
    metrics = _full_retrieval_report(cache, scores)
    report = {
        "format": FUSION_REPORT_FORMAT,
        "task": cache.task,
        "split": cache.split,
        "samples": int(len(cache.sample_indices)),
        "classes": int(cache.targets.size(1)),
        "used_hr_ratio": cache.used_hr_ratio,
        "fusion": fusion_summary,
        "metrics": metrics,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_predictor_scores(
    *,
    config: str | Path | LRFusionConfig,
    inputs: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    cfg = (
        load_lr_fusion_config(config)
        if not isinstance(config, LRFusionConfig)
        else config
    )
    cache = merge_caches(inputs)
    if cache.task != "gl10m":
        raise ValueError(f"Predictor-only evaluation requires a GL-10M cache, got {cache.task}")
    benchmark_config = load_gl10m_config(cfg.gl10m_config)
    root = benchmark_config.path.parent.parent
    dataset_spec = benchmark_config.raw["dataset"]
    configured_csv = Path(str(dataset_spec["split_csv"]))
    csv_path = _resolve(root, configured_csv.with_name(f"{cache.split}_loc.csv"))
    _require_full_coverage(cache, len(pd.read_csv(csv_path)))
    report = {
        "format": PREDICTOR_REPORT_FORMAT,
        "task": cache.task,
        "split": cache.split,
        "samples": int(len(cache.sample_indices)),
        "classes": int(cache.targets.size(1)),
        "used_hr_ratio": cache.used_hr_ratio,
        "score": "predictor",
        "metrics": _gl10m_metrics(cache.pred_hr_scores, cache.targets),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _cache_destination(
    config: LRFusionConfig,
    task: str,
    split: str,
    shard_index: int,
    num_shards: int,
    output: str | Path | None,
) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return (
        config.cache_root
        / task
        / split
        / f"part-{shard_index:05d}-of-{num_shards:05d}.pt"
    )


def _cache_payload(
    *,
    task: str,
    split: str,
    sample_indices: Sequence[torch.Tensor] | torch.Tensor,
    targets: Sequence[torch.Tensor] | torch.Tensor,
    pred_hr_scores: Sequence[torch.Tensor] | torch.Tensor,
    class_names: Sequence[str],
    lr_scores: dict[str, Sequence[torch.Tensor] | torch.Tensor] | None = None,
    selected_counts: Sequence[torch.Tensor] | torch.Tensor | None = None,
    used_hr_ratio: float | None = None,
) -> dict[str, Any]:
    def joined(value: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if not value:
            return torch.empty(0)
        first = value[0]
        return torch.stack(list(value)) if first.ndim <= 1 else torch.cat(list(value))

    indices = joined(sample_indices).long()
    target_values = joined(targets).to(torch.uint8)
    pred_values = joined(pred_hr_scores).float()
    lr_values = {
        name: joined(value).float()
        for name, value in (lr_scores or {}).items()
    }
    if pred_values.shape != target_values.shape:
        raise ValueError("Cross-scale cache score/target shapes differ")
    if any(value.shape != target_values.shape for value in lr_values.values()):
        raise ValueError("A cross-scale LR branch does not match the target shape")
    if len(indices) != len(target_values):
        raise ValueError("Cross-scale cache sample index count differs from score count")
    payload: dict[str, Any] = {
        "format": FUSION_CACHE_FORMAT,
        "task": task,
        "split": split,
        "sample_indices": indices,
        "targets": target_values,
        "pred_hr_scores": pred_values,
        "lr_scores": lr_values,
        "class_names": [str(value) for value in class_names],
    }
    if selected_counts is not None:
        selected = joined(selected_counts).to(torch.int16)
        if len(selected) != len(indices):
            raise ValueError("selected_counts length differs from cache sample count")
        payload["selected_counts"] = selected
        payload["used_hr_ratio"] = float(selected.float().mean() / 100)
    elif used_hr_ratio is not None:
        payload["used_hr_ratio"] = float(used_hr_ratio)
    else:
        raise ValueError("Cross-scale cache requires selected_counts or used_hr_ratio")
    return payload


@torch.inference_mode()
def _cache_zero_shot_scores(
    config: LRFusionConfig,
    *,
    split: str,
    output: Path,
    device: str,
    shard_index: int,
    num_shards: int,
    count_batch_size: int,
) -> dict[str, Any]:
    model_config = load_zero_shot_config(config.zero_shot_model_config)
    frame = pd.read_csv(model_config.data_root / f"{split}_loc_group.csv")
    indices = list(range(shard_index, len(frame), num_shards))
    model = ZeroShotCrossSO(model_config, device=device)
    pred_scores: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    lr_values: dict[str, list[torch.Tensor]] = {"mean": []}
    for index in tqdm(indices, desc=f"lr-fusion:zero-shot:{split}:{shard_index}"):
        row = frame.iloc[index]
        lr_path, hr_paths = _row_paths(row, model_config.data_root)
        provider = FileHRTileProvider(hr_paths, model_config.image_size)
        components = model.predict_classwise_pc_score_components(
            lr_path,
            provider,
            count_batch_size=count_batch_size,
        )
        if len(provider.loaded_indices) != int(components["maximum_union_pc"]):
            raise RuntimeError("Fusion cache opened HR tiles outside the shared Pc prefix")
        pred = components["pred_hr_scores"]
        assert isinstance(pred, torch.Tensor)
        pred_scores.append(pred.float().cpu())
        lr_mean = components["lr_mean_scores"]
        assert isinstance(lr_mean, torch.Tensor)
        lr_values["mean"].append(lr_mean.float().cpu())
        target_values.append(labels_from_row(row).to(torch.uint8))
    payload = _cache_payload(
        task="zero-shot",
        split=split,
        sample_indices=torch.tensor(indices, dtype=torch.int64),
        targets=target_values,
        pred_hr_scores=pred_scores,
        class_names=CLASS_NAMES,
        lr_scores=lr_values,
        used_hr_ratio=model.pc_policy.obr,
    )
    _atomic_torch_save(payload, output)
    return {
        "format": FUSION_CACHE_FORMAT,
        "task": "zero-shot",
        "split": split,
        "samples": len(indices),
        "used_hr_ratio": model.pc_policy.obr,
        "output": str(output),
    }


@torch.inference_mode()
def _cache_gl10m_scores(
    config: LRFusionConfig,
    *,
    split: str,
    output: Path,
    device: str,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    num_workers: int,
    hr_workers: int,
    hr_encode_batch: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() and str(device) != "cpu":
        raise RuntimeError(
            "GL-10M cross-scale cache requires CUDA unless --device cpu is explicit"
        )
    benchmark_config = load_gl10m_config(config.gl10m_config)
    root = benchmark_config.path.parent.parent
    dataset_spec = benchmark_config.raw["dataset"]
    configured_csv = Path(str(dataset_spec["split_csv"]))
    csv_path = _resolve(root, configured_csv.with_name(f"{split}_loc.csv"))
    image_root = _resolve(root, dataset_spec["image_root"])
    class_names = tuple(str(value) for value in dataset_spec["class_names"])
    class_count = len(class_names)
    frame = pd.read_csv(csv_path)
    indices = list(range(shard_index, len(frame), num_shards))

    model_config = load_zero_shot_config(config.zero_shot_model_config)
    model = ZeroShotCrossSO(model_config, device=device)
    router = Router(model_config.dinov2_source)
    router.load_checkpoint(model_config.router_checkpoint)
    router.to(model.device).eval()
    for parameter in router.parameters():
        parameter.requires_grad = False
    template = str(benchmark_config.raw["text"]["template"])
    text = model.encode_texts([template.format(labels=name) for name in class_names])
    topk = int(benchmark_config.raw["observation"]["topk"])
    dataset = _GL10MDataset(
        frame,
        indices,
        image_root=image_root,
        class_count=class_count,
        lr_transform=model.lr_transform,
        router_transform=model.router_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        num_workers=max(0, int(num_workers)),
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    selected_chunks: list[torch.Tensor] = []
    for lr, router_lr, targets, frame_indices, lr_paths in tqdm(
        loader, desc=f"lr-fusion:gl10m:{split}:{shard_index}"
    ):
        lr = lr.to(model.device, non_blocking=True)
        router_lr = router_lr.to(model.device, non_blocking=True)
        context, _ = _gl10m_lr_context_and_summary(model, lr)
        probabilities = router(router_lr).sigmoid().flatten(1)
        selections = [
            row.topk(topk).indices.tolist() if topk else []
            for row in probabilities
        ]
        predicted = _complete_gl10m_batch(
            model,
            context,
            list(lr_paths),
            selections,
            hr_workers=hr_workers,
            hr_encode_batch=hr_encode_batch,
        )
        pred_scores, _ = model.score(predicted, text)
        pred_chunks.append(pred_scores.float().cpu())
        target_chunks.append(targets.to(torch.uint8).cpu())
        index_chunks.append(frame_indices.cpu())
        selected_chunks.append(
            torch.tensor([len(value) for value in selections], dtype=torch.int16)
        )
    payload = _cache_payload(
        task="gl10m",
        split=split,
        sample_indices=torch.cat(index_chunks),
        targets=target_chunks,
        pred_hr_scores=pred_chunks,
        class_names=class_names,
        selected_counts=torch.cat(selected_chunks),
    )
    _atomic_torch_save(payload, output)
    return {
        "format": FUSION_CACHE_FORMAT,
        "task": "gl10m",
        "split": split,
        "samples": len(indices),
        "used_hr_ratio": float(payload["used_hr_ratio"]),
        "output": str(output),
    }


def cache_cross_scale_scores(
    config: str | Path | LRFusionConfig,
    *,
    task: str,
    split: str,
    output: str | Path | None = None,
    device: str = "auto",
    shard_index: int = 0,
    num_shards: int = 1,
    batch_size: int = 4,
    num_workers: int = 2,
    hr_workers: int = 8,
    hr_encode_batch: int = 64,
    count_batch_size: int = 8,
) -> dict[str, Any]:
    cfg = (
        load_lr_fusion_config(config)
        if not isinstance(config, LRFusionConfig)
        else config
    )
    if task not in {"zero-shot", "gl10m"}:
        raise ValueError(f"Unknown cross-scale cache task: {task}")
    if split not in {"valid", "test"}:
        raise ValueError(f"Unknown cross-scale cache split: {split}")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    destination = _cache_destination(
        cfg, task, split, shard_index, num_shards, output
    )
    if task == "zero-shot":
        return _cache_zero_shot_scores(
            cfg,
            split=split,
            output=destination,
            device=device,
            shard_index=shard_index,
            num_shards=num_shards,
            count_batch_size=count_batch_size,
        )
    return _cache_gl10m_scores(
        cfg,
        split=split,
        output=destination,
        device=device,
        shard_index=shard_index,
        num_shards=num_shards,
        batch_size=batch_size,
        num_workers=num_workers,
        hr_workers=hr_workers,
        hr_encode_batch=hr_encode_batch,
    )
