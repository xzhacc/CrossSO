from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _zero(args: argparse.Namespace) -> dict[str, Any]:
    from crossso.fusion import cache_cross_scale_scores, evaluate_lr_fusion

    inputs = args.inputs
    if not inputs:
        cached = cache_cross_scale_scores(
            args.config,
            task="zero-shot",
            split=args.split,
            output=args.cache_output,
            device=args.device,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            count_batch_size=args.count_batch_size,
        )
        if args.num_shards > 1:
            return cached
        inputs = [cached["output"]]
    output = args.output or f"outputs/zero-{args.split}.json"
    return evaluate_lr_fusion(args.config, inputs=inputs, output=output)


def _gl10m(args: argparse.Namespace) -> dict[str, Any]:
    from crossso.fusion import cache_cross_scale_scores, evaluate_predictor_scores
    from crossso.fusion import load_gl10m_config

    config = load_gl10m_config(args.gl10m_config)
    root = config.path.parent.parent
    profile = config.raw["evaluation"]
    if args.inputs:
        inputs = args.inputs
    else:
        cached = cache_cross_scale_scores(
            _resolve(root, profile["config"]),
            task="gl10m",
            split=args.split,
            output=args.cache_output,
            device=args.device,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_workers=args.workers,
            hr_workers=args.hr_workers,
            hr_encode_batch=args.hr_encode_batch,
        )
        if args.num_shards > 1:
            return cached
        inputs = [cached["output"]]
    output = args.output or _resolve(root, profile["report"])
    return evaluate_predictor_scores(
        config=_resolve(root, profile["config"]),
        inputs=inputs,
        output=output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossso")
    parser.add_argument("--version", action="version", version="%(prog)s 4.0.0")
    commands = parser.add_subparsers(dest="command", required=True)

    zero = commands.add_parser("zero")
    zero.add_argument("--config", default="configs/zero.yaml")
    zero.add_argument("--split", choices=("valid", "test"), default="test")
    zero.add_argument("--device", default="auto")
    zero.add_argument("--inputs", nargs="*")
    zero.add_argument("--output")
    zero.add_argument("--cache-output")
    zero.add_argument("--shard-index", type=int, default=0)
    zero.add_argument("--num-shards", type=int, default=1)
    zero.add_argument("--count-batch-size", type=int, default=8)

    transfer_infer = commands.add_parser("transfer-infer")
    transfer_infer.add_argument("--config", default="configs/transfer.yaml")
    transfer_infer.add_argument("--dataset", choices=("eurosat", "bigearthnet"), required=True)
    transfer_infer.add_argument("--device", default="auto")
    transfer_infer.add_argument("--output")
    transfer_infer.add_argument("--shard-index", type=int, default=0)
    transfer_infer.add_argument("--num-shards", type=int, default=1)

    transfer_merge = commands.add_parser("transfer-merge")
    transfer_merge.add_argument("--config", default="configs/transfer.yaml")
    transfer_merge.add_argument("--dataset", choices=("eurosat", "bigearthnet"), required=True)
    transfer_merge.add_argument("--inputs", nargs="+", required=True)
    transfer_merge.add_argument("--output")

    transfer_eval = commands.add_parser("transfer-eval")
    transfer_eval.add_argument("--config", default="configs/transfer.yaml")
    transfer_eval.add_argument("--dataset", choices=("eurosat", "bigearthnet"), required=True)
    transfer_eval.add_argument("--predictions")
    transfer_eval.add_argument("--output")

    gl10m = commands.add_parser("gl10m")
    gl10m.add_argument("--gl10m-config", default="configs/gl10m.yaml")
    gl10m.add_argument("--split", choices=("test",), default="test")
    gl10m.add_argument("--device", default="auto")
    gl10m.add_argument("--inputs", nargs="*")
    gl10m.add_argument("--output")
    gl10m.add_argument("--cache-output")
    gl10m.add_argument("--shard-index", type=int, default=0)
    gl10m.add_argument("--num-shards", type=int, default=1)
    gl10m.add_argument("--batch-size", type=int, default=4)
    gl10m.add_argument("--workers", type=int, default=2)
    gl10m.add_argument("--hr-workers", type=int, default=8)
    gl10m.add_argument("--hr-encode-batch", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "zero":
        _json(_zero(args))
    elif args.command == "transfer-infer":
        from crossso.evaluation import collect_transfer_predictions, load_transfer_config

        _json(
            collect_transfer_predictions(
                load_transfer_config(args.config, profile=args.dataset),
                output=args.output,
                device=args.device,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        )
    elif args.command == "transfer-merge":
        from crossso.evaluation import load_transfer_config, merge_transfer_prediction_shards

        _json(
            merge_transfer_prediction_shards(
                load_transfer_config(args.config, profile=args.dataset),
                shards=args.inputs,
                output=args.output,
            )
        )
    elif args.command == "transfer-eval":
        from crossso.evaluation import evaluate_transfer_predictions, load_transfer_config

        _json(
            evaluate_transfer_predictions(
                load_transfer_config(args.config, profile=args.dataset),
                predictions=args.predictions,
                output=args.output,
            )
        )
    elif args.command == "gl10m":
        _json(_gl10m(args))


if __name__ == "__main__":
    main()
