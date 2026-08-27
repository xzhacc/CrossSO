from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from crossso.metrics import multilabel_average_precision
from crossso.models import (
    GraftAnchorFusionConfig,
    GraftZeroHRCompletion,
    checkpoint_state,
    fuse_graft_anchor_scores,
    load_graft_anchor_fusion_config,
)


TRANSFER_PREDICTIONS_FORMAT = "crossso-transfer-predictions-v1"


def _load_prediction(path: str | Path) -> dict[str, Any]:
    from torch.torch_version import TorchVersion

    source = Path(path).expanduser().resolve()
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Transfer artifact must be a mapping: {source}")
    return payload


@dataclass
class TransferConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    prediction_path: Path
    report_path: Path

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def load_transfer_config(
    path: str | Path,
    *,
    profile: str | None = None,
) -> TransferConfig:
    config_path = Path(path).expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Transfer config must be a mapping: {config_path}")
    profiles = document.get("profiles")
    if profiles is not None:
        if not isinstance(profiles, dict) or profile not in profiles:
            available = ", ".join(sorted(str(value) for value in profiles or ()))
            raise ValueError(f"Choose one transfer profile: {available}")
        raw = profiles[profile]
    else:
        raw = document
    if not isinstance(raw, dict):
        raise ValueError(f"Transfer profile must be a mapping: {profile!r}")
    dataset = raw.get("dataset")
    model = raw.get("model")
    text = raw.get("text")
    artifacts = raw.get("artifacts")
    if not all(isinstance(value, dict) for value in (dataset, model, text, artifacts)):
        raise ValueError("Transfer config requires dataset/model/text/artifacts")
    task = str(dataset.get("task", ""))
    if task not in {"single-label-classification", "multi-label-retrieval"}:
        raise ValueError(f"Unsupported cross-dataset transfer task: {task!r}")
    prompt_groups = text.get("prompt_groups")
    primary_group = str(text.get("primary_group", ""))
    if not isinstance(prompt_groups, dict) or primary_group not in prompt_groups:
        fusion = load_graft_anchor_fusion_config(
            raw.get("cross_scale_fusion"), root=config_path.parent.parent
        )
        if fusion is None or primary_group != fusion.output_group:
            raise ValueError(
                "text.primary_group must name one text.prompt_groups entry or the "
                "cross_scale_fusion.output_group"
            )
    fusion = load_graft_anchor_fusion_config(
        raw.get("cross_scale_fusion"), root=config_path.parent.parent
    )
    if fusion is not None:
        if fusion.input_group not in prompt_groups:
            raise ValueError(
                "cross_scale_fusion.input_group must name one text.prompt_groups entry"
            )
        if fusion.output_group in prompt_groups:
            raise ValueError(
                "cross_scale_fusion.output_group must not overwrite a prompt group"
            )
    root = config_path.parent.parent
    return TransferConfig(
        path=config_path,
        root=root,
        raw=raw,
        prediction_path=(root / str(artifacts["predictions"])).resolve(),
        report_path=(root / str(artifacts["report"])).resolve(),
    )


def _load_graft_vision(
    model_source: Path, checkpoint: Path, device: torch.device
) -> CLIPVisionModelWithProjection:
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_source, local_files_only=True
    )
    state = checkpoint_state(checkpoint)
    prefix = "satellite_image_backbone."
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        selected = state
    result = model.load_state_dict(selected, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith("position_ids")]
    unexpected = [
        key for key in result.unexpected_keys if not key.endswith("position_ids")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"GRAFT vision state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval()


@torch.inference_mode()
def _encode_prompt_groups(
    config: TransferConfig,
    model_source: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text_spec = config.raw["text"]
    class_names = [str(value) for value in text_spec["class_names"]]
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(
        model_source, local_files_only=True
    ).to(device)
    model.eval()
    output: dict[str, torch.Tensor] = {}
    for group_name, raw_templates in text_spec["prompt_groups"].items():
        templates_ = [str(value) for value in raw_templates]
        if not templates_:
            raise ValueError(f"Empty prompt group: {group_name}")
        encoded = []
        for template in templates_:
            tokens = tokenizer(
                [template.format(label=name) for name in class_names],
                padding=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            encoded.append(F.normalize(model(**tokens).text_embeds.float(), dim=-1))
        output[str(group_name)] = F.normalize(torch.stack(encoded).mean(0), dim=-1)
    return output


def _make_transform(spec: dict[str, Any]) -> Any:
    method = str(spec.get("method", "black-pad"))
    if method != "black-pad":
        raise ValueError(f"Cross-dataset transfer supports only black-pad, got {method!r}")
    padding = int(spec["padding"])
    mean = tuple(float(value) for value in spec["mean"])
    std = tuple(float(value) for value in spec["std"])
    return transforms.Compose(
        [
            transforms.Pad((padding, padding, padding, padding), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class _EuroSATDataset(Dataset):
    def __init__(
        self,
        config: TransferConfig,
        *,
        completion_transform: Any | None = None,
    ) -> None:
        dataset = config.raw["dataset"]
        self.names = config.resolve(dataset["split_file"]).read_text(encoding="utf-8").splitlines()
        self.image_root = config.resolve(dataset["image_root"])
        self.class_directories = tuple(str(value) for value in dataset["class_directories"])
        self.class_to_index = {
            name: index for index, name in enumerate(self.class_directories)
        }
        self.transform = _make_transform(config.raw["preprocess"])
        self.completion_transform = completion_transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        filename = self.names[index]
        directory = filename.split("_", 1)[0]
        if directory not in self.class_to_index:
            raise ValueError(f"Unknown EuroSAT class directory: {directory}")
        with Image.open(self.image_root / directory / filename) as image:
            rgb = image.convert("RGB")
            value = self.transform(rgb)
            completion = (
                self.completion_transform(rgb)
                if self.completion_transform is not None
                else None
            )
        if completion is None:
            return value, self.class_to_index[directory]
        return value, completion, self.class_to_index[directory]


def _prediction_payload(
    config: TransferConfig,
    *,
    score_parts: dict[str, list[torch.Tensor]],
    targets: list[torch.Tensor],
    sample_indices: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    class_names = tuple(str(value) for value in config.raw["text"]["class_names"])
    task = str(config.raw["dataset"]["task"])
    empty_target_shape = (0,) if task == "single-label-classification" else (0, len(class_names))
    empty_target_dtype = torch.int64 if task == "single-label-classification" else torch.uint8
    target_tensor = (
        torch.cat(targets)
        if targets
        else torch.empty(empty_target_shape, dtype=empty_target_dtype)
    )
    if sample_indices is None:
        index_tensor = torch.arange(target_tensor.shape[0], dtype=torch.int64)
    else:
        index_tensor = (
            torch.cat(sample_indices).long()
            if sample_indices
            else torch.empty(0, dtype=torch.int64)
        )
        if index_tensor.numel() != target_tensor.shape[0]:
            raise ValueError("Transfer sample_indices and targets have different lengths")
    return {
        "format": TRANSFER_PREDICTIONS_FORMAT,
        "dataset": str(config.raw["dataset"]["kind"]),
        "task": task,
        "class_names": class_names,
        "scores": {
            name: torch.cat(parts).float()
            if parts
            else torch.empty(0, len(class_names), dtype=torch.float32)
            for name, parts in score_parts.items()
        },
        "targets": target_tensor,
        "sample_indices": index_tensor,
    }


@torch.inference_mode()
def _collect_eurosat(
    config: TransferConfig,
    *,
    model: CLIPVisionModelWithProjection,
    text: dict[str, torch.Tensor],
    device: torch.device,
    completion: GraftZeroHRCompletion | None = None,
    fusion: GraftAnchorFusionConfig | None = None,
) -> dict[str, Any]:
    runtime = config.raw.get("runtime", {})
    if (completion is None) != (fusion is None):
        raise ValueError("EuroSAT completion model and fusion config must be provided together")
    dataset = _EuroSATDataset(
        config,
        completion_transform=completion.transform if completion is not None else None,
    )
    batch_size = int(
        runtime.get("fusion_batch_size", 4)
        if fusion is not None
        else runtime.get("batch_size", 128)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(runtime.get("workers", 8)),
        shuffle=False,
        pin_memory=True,
        persistent_workers=int(runtime.get("workers", 8)) > 0,
    )
    score_parts = {name: [] for name in text}
    if fusion is not None:
        score_parts[fusion.output_group] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        if fusion is None:
            images, labels = batch
            completion_images = None
        else:
            images, completion_images, labels = batch
        embeddings = F.normalize(
            model(pixel_values=images.to(device, non_blocking=True)).image_embeds.float(),
            dim=-1,
        )
        batch_scores: dict[str, torch.Tensor] = {}
        for name, text_features in text.items():
            values = embeddings @ text_features.T
            batch_scores[name] = values
            score_parts[name].append(values.cpu())
        if fusion is not None and completion is not None:
            assert completion_images is not None
            predicted = completion.completed_features(completion_images)
            predictor_scores = completion.class_scores(
                predicted,
                text[fusion.input_group],
                pool=fusion.predictor_pool,
            )
            fused = fuse_graft_anchor_scores(
                batch_scores[fusion.input_group],
                predictor_scores,
                weight=fusion.weight,
                normalization=fusion.normalization,
            )
            score_parts[fusion.output_group].append(fused.cpu())
        targets.append(labels.long().cpu())
    return _prediction_payload(
        config,
        score_parts=score_parts,
        targets=targets,
    )


def _bigearthnet_label_mapping(config: TransferConfig) -> dict[str, int]:
    groups = config.raw["dataset"]["label_groups"]
    mapping: dict[str, int] = {}
    for destination, source_names in enumerate(groups):
        for source_name in source_names:
            name = str(source_name)
            if name in mapping:
                raise ValueError(f"Duplicate BigEarthNet source label: {name}")
            mapping[name] = destination
    return mapping


@torch.inference_mode()
def _collect_bigearthnet(
    config: TransferConfig,
    *,
    model: CLIPVisionModelWithProjection,
    text: dict[str, torch.Tensor],
    device: torch.device,
    completion: GraftZeroHRCompletion | None = None,
    fusion: GraftAnchorFusionConfig | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    if (completion is None) != (fusion is None):
        raise ValueError(
            "BigEarthNet completion model and fusion config must be provided together"
        )
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "BigEarthNet evaluation requires: pip install -e '.[eval]'"
        ) from error

    dataset = config.raw["dataset"]
    runtime = config.raw.get("runtime", {})
    root = config.resolve(dataset["parquet_root"])
    paths = sorted(root.glob(str(dataset["parquet_glob"])))
    if not paths:
        raise FileNotFoundError(f"No BigEarthNet parquet files under {root}")
    mapping = _bigearthnet_label_mapping(config)
    class_count = len(config.raw["text"]["class_names"])
    transform = _make_transform(config.raw["preprocess"])
    score_parts = {name: [] for name in text}
    if fusion is not None:
        score_parts[fusion.output_group] = []
    targets: list[torch.Tensor] = []
    sample_indices: list[torch.Tensor] = []
    batch_size = int(runtime.get("batch_size", 256))
    available = max(0, (int(dataset["samples"]) - shard_index + num_shards - 1) // num_shards)
    global_offset = 0
    with tqdm(total=available, desc="transfer:bigearthnet", unit="image") as progress:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for record_batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=(str(dataset["image_column"]), str(dataset["label_column"])),
            ):
                all_rows = record_batch.to_pylist()
                selected_rows = [
                    (global_offset + offset, row)
                    for offset, row in enumerate(all_rows)
                    if (global_offset + offset) % num_shards == shard_index
                ]
                global_offset += len(all_rows)
                if not selected_rows:
                    continue
                images = []
                completion_images = []
                batch_targets = []
                batch_sample_indices = []
                for sample_index, row in selected_rows:
                    image_value = row[str(dataset["image_column"])]
                    encoded = (
                        image_value["bytes"]
                        if isinstance(image_value, dict)
                        else image_value
                    )
                    with Image.open(io.BytesIO(encoded)) as image:
                        rgb = image.convert("RGB")
                        images.append(transform(rgb))
                        if completion is not None:
                            completion_images.append(completion.transform(rgb))
                    target = np.zeros(class_count, dtype=np.uint8)
                    for label in row[str(dataset["label_column"])]:
                        destination = mapping.get(str(label))
                        if destination is not None:
                            target[destination] = 1
                    batch_targets.append(target)
                    batch_sample_indices.append(sample_index)
                embeddings = F.normalize(
                    model(
                        pixel_values=torch.stack(images).to(device, non_blocking=True)
                    ).image_embeds.float(),
                    dim=-1,
                )
                batch_scores: dict[str, torch.Tensor] = {}
                for name, text_features in text.items():
                    values = embeddings @ text_features.T
                    batch_scores[name] = values
                    score_parts[name].append(values.cpu())
                if fusion is not None and completion is not None:
                    chunk_size = int(runtime.get("fusion_batch_size", 4))
                    predictor_parts = []
                    stacked = torch.stack(completion_images)
                    for start in range(0, stacked.size(0), chunk_size):
                        predicted = completion.completed_features(
                            stacked[start : start + chunk_size]
                        )
                        predictor_parts.append(
                            completion.class_scores(
                                predicted,
                                text[fusion.input_group],
                                pool=fusion.predictor_pool,
                            )
                        )
                    predictor_scores = torch.cat(predictor_parts)
                    fused = fuse_graft_anchor_scores(
                        batch_scores[fusion.input_group],
                        predictor_scores,
                        weight=fusion.weight,
                        normalization=fusion.normalization,
                    )
                    score_parts[fusion.output_group].append(fused.cpu())
                targets.append(torch.from_numpy(np.stack(batch_targets)))
                sample_indices.append(
                    torch.tensor(batch_sample_indices, dtype=torch.int64)
                )
                progress.update(len(selected_rows))
    return _prediction_payload(
        config,
        score_parts=score_parts,
        targets=targets,
        sample_indices=sample_indices,
    )


@torch.inference_mode()
def collect_transfer_predictions(
    config: str | Path | TransferConfig,
    *,
    output: str | Path | None = None,
    device: str = "auto",
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Run one EuroSAT or BigEarthNet transfer profile."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    cfg = (
        load_transfer_config(config)
        if not isinstance(config, TransferConfig)
        else config
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    require_cuda = bool(cfg.raw.get("runtime", {}).get("require_cuda", True))
    if require_cuda and torch_device.type != "cuda":
        raise RuntimeError("Formal cross-dataset transfer inference requires CUDA")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_source = cfg.resolve(cfg.raw["model"]["clip_model"])
    checkpoint = cfg.resolve(cfg.raw["model"]["graft_checkpoint"])
    model = _load_graft_vision(model_source, checkpoint, torch_device)
    text = _encode_prompt_groups(cfg, model_source, torch_device)
    fusion = load_graft_anchor_fusion_config(
        cfg.raw.get("cross_scale_fusion"), root=cfg.root
    )
    completion = None
    if fusion is not None:
        completion = GraftZeroHRCompletion(
            clip_source=model_source,
            graft_naip_checkpoint=fusion.graft_naip_checkpoint,
            predictor_checkpoint=fusion.predictor_checkpoint,
            device=torch_device,
        )
    kind = str(cfg.raw["dataset"]["kind"])
    if kind == "eurosat":
        if num_shards != 1 or shard_index != 0:
            raise ValueError("EuroSAT transfer collection does not require sharding")
        payload = _collect_eurosat(
            cfg,
            model=model,
            text=text,
            device=torch_device,
            completion=completion,
            fusion=fusion,
        )
    elif kind == "bigearthnet":
        payload = _collect_bigearthnet(
            cfg,
            model=model,
            text=text,
            device=torch_device,
            completion=completion,
            fusion=fusion,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    else:
        raise ValueError(f"Unsupported cross-dataset transfer dataset kind: {kind!r}")
    destination = cfg.prediction_path if output is None else Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "output": str(destination),
        "dataset": payload["dataset"],
        "samples": int(payload["targets"].shape[0]),
        "classes": len(payload["class_names"]),
        "prompt_groups": list(payload["scores"]),
    }


def merge_transfer_prediction_shards(
    config: str | Path | TransferConfig,
    *,
    shards: list[str | Path],
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Merge independently collected transfer shards in canonical sample order."""
    if not shards:
        raise ValueError("At least one cross-dataset transfer shard is required")
    cfg = (
        load_transfer_config(config)
        if not isinstance(config, TransferConfig)
        else config
    )
    payloads = [_load_prediction(path) for path in shards]
    reference = payloads[0]
    required = ("format", "dataset", "task", "class_names", "scores", "targets")
    for payload in payloads:
        if any(payload.get(key) != reference.get(key) for key in required[:4]):
            raise ValueError("Incompatible cross-dataset transfer shard")
        if payload.get("format") != TRANSFER_PREDICTIONS_FORMAT:
            raise ValueError("Unexpected cross-dataset transfer shard format")
        if tuple(payload.get("class_names", ())) != tuple(reference["class_names"]):
            raise ValueError("Incompatible cross-dataset transfer shard class names")
        if set(payload.get("scores", {})) != set(reference["scores"]):
            raise ValueError("Incompatible cross-dataset transfer shard score groups")

    index_parts = []
    target_parts = []
    score_parts = {name: [] for name in reference["scores"]}
    for payload in payloads:
        targets = payload["targets"]
        indices = payload.get("sample_indices")
        if not isinstance(targets, torch.Tensor) or not isinstance(indices, torch.Tensor):
            raise ValueError("Transfer shards require tensor targets and sample_indices")
        if indices.ndim != 1 or indices.numel() != targets.shape[0]:
            raise ValueError("Transfer shard indices and targets have different lengths")
        index_parts.append(indices.long())
        target_parts.append(targets)
        for name, values in payload["scores"].items():
            if not isinstance(values, torch.Tensor) or values.shape[0] != targets.shape[0]:
                raise ValueError(f"Invalid score tensor in shard group {name!r}")
            score_parts[name].append(values)

    indices = torch.cat(index_parts)
    if indices.unique().numel() != indices.numel():
        raise ValueError("Duplicate sample indices across cross-dataset transfer shards")
    order = indices.argsort()
    indices = indices[order]
    expected_samples = int(cfg.raw["dataset"]["samples"])
    if not torch.equal(
        indices, torch.arange(expected_samples, dtype=torch.int64)
    ):
        raise ValueError(
            f"Transfer shards do not cover every sample exactly once: "
            f"{indices.numel()} != {expected_samples}"
        )
    targets = torch.cat(target_parts)[order]
    scores = {
        name: torch.cat(parts)[order].float() for name, parts in score_parts.items()
    }
    merged = {
        "format": TRANSFER_PREDICTIONS_FORMAT,
        "dataset": reference["dataset"],
        "task": reference["task"],
        "class_names": tuple(reference["class_names"]),
        "scores": scores,
        "targets": targets,
        "sample_indices": indices,
    }
    destination = (
        cfg.prediction_path if output is None else Path(output).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, destination)
    return {
        "output": str(destination),
        "dataset": merged["dataset"],
        "samples": int(targets.shape[0]),
        "classes": len(merged["class_names"]),
        "prompt_groups": list(scores),
        "shards": len(shards),
    }


def evaluate_transfer_predictions(
    config: str | Path | TransferConfig,
    *,
    predictions: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate stored cross-dataset transfer scores without fitting."""
    cfg = (
        load_transfer_config(config)
        if not isinstance(config, TransferConfig)
        else config
    )
    prediction_path = (
        cfg.prediction_path
        if predictions is None
        else Path(predictions).expanduser().resolve()
    )
    payload = _load_prediction(prediction_path)
    if payload.get("format") != TRANSFER_PREDICTIONS_FORMAT:
        raise ValueError(f"Unexpected cross-dataset transfer prediction artifact: {prediction_path}")
    scores = payload.get("scores")
    targets = payload.get("targets")
    sample_indices = payload.get("sample_indices")
    if not isinstance(scores, dict) or not isinstance(targets, torch.Tensor):
        raise ValueError("cross-dataset transfer artifact requires scores and targets")
    class_names = tuple(str(value) for value in cfg.raw["text"]["class_names"])
    dataset_kind = str(cfg.raw["dataset"]["kind"])
    task = str(cfg.raw["dataset"]["task"])
    if str(payload.get("dataset")) != dataset_kind or str(payload.get("task")) != task:
        raise ValueError("cross-dataset transfer artifact does not match the config")
    if tuple(str(value) for value in payload.get("class_names", ())) != class_names:
        raise ValueError("cross-dataset transfer class order differs from the config")
    expected_samples = int(cfg.raw["dataset"]["samples"])
    sample_count = int(targets.shape[0])
    if sample_count != expected_samples:
        raise ValueError(f"Sample count mismatch: {sample_count} != {expected_samples}")
    expected_indices = torch.arange(expected_samples, dtype=torch.int64)
    if not isinstance(sample_indices, torch.Tensor) or not torch.equal(
        sample_indices.long().cpu(),
        expected_indices,
    ):
        raise ValueError("cross-dataset transfer artifact must cover every sample exactly once")
    primary_group = str(cfg.raw["text"]["primary_group"])
    if primary_group not in scores:
        raise ValueError(f"Missing primary prompt group {primary_group!r}")

    primary_scores = scores[primary_group].float()
    if task == "single-label-classification":
        if targets.ndim != 1:
            raise ValueError("Single-label targets must have shape [samples]")
        if primary_scores.shape != (sample_count, len(class_names)):
            raise ValueError(f"Score shape mismatch: {tuple(primary_scores.shape)}")
        guesses = primary_scores.argmax(1)
        primary_metric = "accuracy"
        primary_value = float((guesses == targets).float().mean())
    else:
        if targets.shape != (sample_count, len(class_names)):
            raise ValueError(f"Target shape mismatch: {tuple(targets.shape)}")
        if primary_scores.shape != targets.shape:
            raise ValueError(f"Score shape mismatch: {tuple(primary_scores.shape)}")
        metrics = multilabel_average_precision(primary_scores, targets)
        primary_metric = "macro_mAP"
        primary_value = float(metrics["macro_supported"])

    report = {
        "format": "crossso-transfer-evaluation-v1",
        "dataset": dataset_kind,
        "task": task,
        "samples": sample_count,
        "classes": len(class_names),
        "primary_prompt_group": primary_group,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
    }
    destination = cfg.report_path if output is None else Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "TRANSFER_PREDICTIONS_FORMAT",
    "TransferConfig",
    "collect_transfer_predictions",
    "evaluate_transfer_predictions",
    "load_transfer_config",
    "merge_transfer_prediction_shards",
]
