from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossso.prepare import PREPARE_STAGES, run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("stage", choices=PREPARE_STAGES)
    root.add_argument("--work-dir", type=Path)
    root.add_argument("--regions", nargs="+")
    root.add_argument("--raw-root", type=Path)
    root.add_argument("--output-dir", type=Path)
    root.add_argument("--loc-manifest-dir", type=Path)
    root.add_argument("--no-verify", action="store_true")
    root.add_argument("--ee-project")
    root.add_argument("--workers", type=int, default=1)
    root.add_argument("--overwrite", action="store_true")
    root.add_argument("--repair", action="store_true")
    root.add_argument("--dry-run", action="store_true")
    root.add_argument("--limit", type=int)
    root.add_argument("--url-template")
    root.add_argument("--max-samples", type=int)
    root.add_argument("--seed", type=int)
    root.add_argument("--csv-suffix", default="_sampled")
    root.add_argument("--output-region-suffix", default="")
    root.add_argument("--absolute-paths", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    result = run(
        args.stage,
        work_dir=args.work_dir,
        regions=args.regions,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        loc_manifest_dir=args.loc_manifest_dir,
        verify_paths=not args.no_verify,
        ee_project=args.ee_project,
        workers=args.workers,
        overwrite=args.overwrite,
        repair=args.repair,
        dry_run=args.dry_run,
        limit=args.limit,
        url_template=args.url_template,
        max_samples=args.max_samples,
        seed=args.seed,
        csv_suffix=args.csv_suffix,
        output_region_suffix=args.output_region_suffix,
        absolute_paths=args.absolute_paths,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
