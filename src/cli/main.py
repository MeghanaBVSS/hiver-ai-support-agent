"""CLI entrypoints for Phase 0 workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config
from src.data.acquire import (
    download_full_dataset,
    download_sample_csv,
    print_acquisition_instructions,
)
from src.data.audit import run_audit
from src.data.brand_analysis import run_brand_analysis
from src.data.loader import load_tweets
from src.evaluation.golden_design import GoldenSamplingPlan, write_golden_design_artifact
from src.intent.discovery import propose_from_brand_frame
from src.preprocessing.conversations import (
    build_customer_support_pairs,
    conversations_to_frame,
    pairs_to_frame,
    reconstruct_conversations,
)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def cmd_acquire(args: argparse.Namespace) -> int:
    if args.instructions or (
        not args.sample
        and not args.full
        and not getattr(args, "sample_mirror", False)
        and not getattr(args, "hf_twcs", False)
    ):
        print_acquisition_instructions()
        return 0
    cfg = load_config()
    if args.sample:
        path = download_sample_csv(cfg=cfg)
        print(f"Downloaded sample to {path}")
        return 0
    if getattr(args, "sample_mirror", False):
        from src.data.acquire import download_sample_mirror

        path = download_sample_mirror(cfg=cfg)
        print(f"Downloaded sample mirror to {path}")
        return 0
    if getattr(args, "hf_twcs", False):
        from src.data.acquire import download_twcs_from_hf

        path = download_twcs_from_hf(cfg=cfg)
        print(f"Downloaded twcs.csv to {path}")
        return 0
    if args.full:
        path = download_full_dataset(cfg=cfg)
        print(f"Downloaded full dataset to {path}")
        return 0
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    data_path = Path(args.data_path) if args.data_path else cfg.resolve_data_path()
    out = Path(args.output) if args.output else cfg.reports_dir / "phase0" / "data_audit.json"
    report = run_audit(
        data_path,
        output_path=out,
        sample_size=args.sample_size if args.sample_size is not None else cfg.sample_size,
        random_seed=args.seed if args.seed is not None else cfg.random_seed,
    )
    print(json.dumps({"output": str(out), "row_count": report["row_count"], "quality_issues": report["quality_issues"]}, indent=2))
    return 0


def cmd_brands(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    data_path = Path(args.data_path) if args.data_path else cfg.resolve_data_path()
    out = Path(args.output) if args.output else cfg.reports_dir / "phase0" / "brand_analysis.json"
    result = run_brand_analysis(
        data_path,
        output_path=out,
        sample_size=args.sample_size if args.sample_size is not None else cfg.sample_size,
        random_seed=args.seed if args.seed is not None else cfg.random_seed,
        top_n=args.top_n,
    )
    rec = result["recommendation"]
    print(
        json.dumps(
            {
                "output": str(out),
                "recommended_brand": rec.get("brand"),
                "status": rec.get("status"),
                "top5": [
                    {"rank": r["rank"], "brand": r["brand"], "score": r["composite_score"]}
                    for r in result["ranked_candidates"][:5]
                ],
            },
            indent=2,
        )
    )
    return 0


def cmd_conversations(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    data_path = Path(args.data_path) if args.data_path else cfg.resolve_data_path()
    df = load_tweets(
        data_path,
        sample_size=args.sample_size if args.sample_size is not None else cfg.sample_size,
        random_seed=args.seed if args.seed is not None else cfg.random_seed,
    )
    brand = args.brand or cfg.selected_brand
    conversations = reconstruct_conversations(df)
    pairs = build_customer_support_pairs(df, brand=brand)
    conv_out = Path(args.conversations_out) if args.conversations_out else cfg.interim_dir / "conversations.csv"
    pairs_out = Path(args.pairs_out) if args.pairs_out else cfg.interim_dir / "customer_support_pairs.csv"
    conversations_to_frame(conversations).to_csv(conv_out, index=False)
    pairs_to_frame(pairs).to_csv(pairs_out, index=False)
    print(
        json.dumps(
            {
                "n_conversations": len(conversations),
                "n_pairs": len(pairs),
                "brand_filter": brand,
                "conversations_out": str(conv_out),
                "pairs_out": str(pairs_out),
            },
            indent=2,
        )
    )
    return 0


def cmd_intents(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    data_path = Path(args.data_path) if args.data_path else cfg.resolve_data_path()
    brand = args.brand or cfg.selected_brand
    if not brand:
        # Try reading recommendation artifact.
        rec_path = cfg.reports_dir / "phase0" / "brand_analysis.json"
        if rec_path.exists():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            brand = rec.get("recommendation", {}).get("brand")
    if not brand:
        print(
            "SELECTED_BRAND is not set and no brand recommendation artifact was found. "
            "Run `python -m src.cli.main brands` first or pass --brand.",
            file=sys.stderr,
        )
        return 2
    df = load_tweets(
        data_path,
        sample_size=args.sample_size if args.sample_size is not None else cfg.sample_size,
        random_seed=args.seed if args.seed is not None else cfg.random_seed,
    )
    out = Path(args.output) if args.output else cfg.reports_dir / "phase0" / "intent_proposals.json"
    result = propose_from_brand_frame(
        df,
        brand=brand,
        n_clusters=args.n_clusters or cfg.discovery_n_clusters,
        random_seed=args.seed if args.seed is not None else cfg.random_seed,
        sample_size=cfg.discovery_sample_size,
        output_path=out,
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "brand": brand,
                "n_intents_proposed": result["n_intents_proposed"],
                "status": result["status"],
            },
            indent=2,
        )
    )
    return 0


def cmd_golden_design(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    out = Path(args.output) if args.output else cfg.reports_dir / "phase0" / "golden_sampling_plan.json"
    plan = GoldenSamplingPlan(
        target_size_min=cfg.golden_size_min,
        target_size_max=cfg.golden_size_max,
    )
    write_golden_design_artifact(out, plan)
    print(json.dumps({"output": str(out), "status": "DESIGN_ONLY_NO_LABELS"}, indent=2))
    return 0


def cmd_phase0(args: argparse.Namespace) -> int:
    """Run the Phase 0 pipeline when data is available."""
    cfg = load_config()
    cfg.ensure_dirs()
    try:
        data_path = Path(args.data_path) if args.data_path else cfg.resolve_data_path()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print_acquisition_instructions()
        # Still write golden design artifact (does not need data).
        cmd_golden_design(argparse.Namespace(output=None))
        return 2

    ns = argparse.Namespace(
        data_path=str(data_path),
        sample_size=args.sample_size,
        seed=args.seed,
        output=None,
        top_n=15,
        brand=args.brand,
        conversations_out=None,
        pairs_out=None,
        n_clusters=None,
    )
    cmd_audit(ns)
    cmd_brands(ns)
    # Use recommended brand if not provided.
    rec_path = cfg.reports_dir / "phase0" / "brand_analysis.json"
    if rec_path.exists() and not ns.brand:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        ns.brand = rec.get("recommendation", {}).get("brand")
    cmd_conversations(ns)
    cmd_intents(ns)
    cmd_golden_design(argparse.Namespace(output=None))
    print("Phase 0 pipeline complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hiver AI support agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_acq = sub.add_parser("acquire", help="Dataset acquisition helpers")
    p_acq.add_argument("--instructions", action="store_true")
    p_acq.add_argument("--sample", action="store_true", help="Download sample.csv via Kaggle CLI")
    p_acq.add_argument(
        "--sample-mirror",
        action="store_true",
        help="Download public sample.csv mirror (no Kaggle credentials)",
    )
    p_acq.add_argument(
        "--hf-twcs",
        action="store_true",
        help="Explicitly download full twcs.csv from HuggingFace mirror",
    )
    p_acq.add_argument("--full", action="store_true", help="Download full dataset via Kaggle (explicit)")
    p_acq.set_defaults(func=cmd_acquire)

    p_audit = sub.add_parser("audit", help="Run dataset audit")
    _add_data_args(p_audit)
    p_audit.add_argument("--output", type=str, default=None)
    p_audit.set_defaults(func=cmd_audit)

    p_brands = sub.add_parser("brands", help="Rank brand candidates")
    _add_data_args(p_brands)
    p_brands.add_argument("--output", type=str, default=None)
    p_brands.add_argument("--top-n", type=int, default=15)
    p_brands.set_defaults(func=cmd_brands)

    p_conv = sub.add_parser("conversations", help="Reconstruct conversations/pairs")
    _add_data_args(p_conv)
    p_conv.add_argument("--brand", type=str, default=None)
    p_conv.add_argument("--conversations-out", type=str, default=None)
    p_conv.add_argument("--pairs-out", type=str, default=None)
    p_conv.set_defaults(func=cmd_conversations)

    p_int = sub.add_parser("intents", help="Propose initial intent taxonomy")
    _add_data_args(p_int)
    p_int.add_argument("--brand", type=str, default=None)
    p_int.add_argument("--n-clusters", type=int, default=None)
    p_int.add_argument("--output", type=str, default=None)
    p_int.set_defaults(func=cmd_intents)

    p_gold = sub.add_parser("golden-design", help="Write golden-set sampling plan")
    p_gold.add_argument("--output", type=str, default=None)
    p_gold.set_defaults(func=cmd_golden_design)

    p0 = sub.add_parser("phase0", help="Run Phase 0 pipeline")
    _add_data_args(p0)
    p0.add_argument("--brand", type=str, default=None)
    p0.set_defaults(func=cmd_phase0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
