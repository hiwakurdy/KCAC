from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .doctor import format_doctor, run_doctor

_CONFIG_SEARCH_ORDER = ["config.yaml", "config.yaml.example"]


def _default_config() -> str:
    for name in _CONFIG_SEARCH_ORDER:
        if Path(name).exists():
            return name
    return _CONFIG_SEARCH_ORDER[-1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description=(
            "KCAC OCR Pipeline v0.1\n\n"
            "Typical first run:\n"
            "  cp config.yaml.example config.yaml   # then edit paths\n"
            "  python -m pipeline doctor\n"
            "  python -m pipeline bootstrap --limit 1\n\n"
            "Stage order: bootstrap → consensus → pagexml → escriptorium → queue → reports\n"
            "Run all stages: python -m pipeline run-all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to pipeline YAML config "
            f"(auto-detects {' → '.join(_CONFIG_SEARCH_ORDER)} if omitted)"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")

    def add_stage(name: str) -> None:
        stage = sub.add_parser(name)
        stage.add_argument("--limit", type=int, default=None, help="Limit pages for smoke runs")
        stage.add_argument("--force", action="store_true", help="Regenerate outputs even if files exist")

    for command in [
        "bootstrap",
        "consensus",
        "pagexml",
        "escriptorium",
        "queue",
        "reports",
        "benchmark",
        "hf-export",
        "run-all",
    ]:
        add_stage(command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config if args.config is not None else _default_config()
    config = load_config(Path(config_path))
    if args.command == "doctor":
        results = run_doctor(config)
        print(format_doctor(results))
        return 0 if all(result.ok for result in results) else 1
    if args.command == "bootstrap":
        from .bootstrap import bootstrap_pages

        paths = bootstrap_pages(config, limit=args.limit, force=args.force)
    elif args.command == "consensus":
        from .consensus import consensus_for_pages

        paths = consensus_for_pages(config, limit=args.limit, force=args.force)
    elif args.command == "pagexml":
        from .pagexml_export import export_pagexml

        paths = export_pagexml(config, limit=args.limit, force=args.force)
    elif args.command == "escriptorium":
        from .escriptorium_import import build_escriptorium_import

        paths = [build_escriptorium_import(config, limit=args.limit)]
    elif args.command == "queue":
        from .queue import build_annotation_queue

        paths = [build_annotation_queue(config, limit=args.limit)]
    elif args.command == "reports":
        from .reports import build_reports

        paths = build_reports(config, limit=args.limit)
    elif args.command == "benchmark":
        from .benchmark import benchmark

        paths = [benchmark(config, limit=args.limit)]
    elif args.command == "hf-export":
        from .hf_export import export_hf_jsonl

        paths = [export_hf_jsonl(config, limit=args.limit)]
    elif args.command == "run-all":
        from .bootstrap import bootstrap_pages
        from .consensus import consensus_for_pages
        from .escriptorium_import import build_escriptorium_import
        from .hf_export import export_hf_jsonl
        from .pagexml_export import export_pagexml
        from .queue import build_annotation_queue
        from .reports import build_reports

        paths = []
        paths.extend(bootstrap_pages(config, limit=args.limit, force=args.force))
        paths.extend(consensus_for_pages(config, limit=args.limit, force=args.force))
        paths.extend(export_pagexml(config, limit=args.limit, force=args.force))
        paths.append(build_escriptorium_import(config, limit=args.limit))
        paths.append(build_annotation_queue(config, limit=args.limit))
        paths.extend(build_reports(config, limit=args.limit))
        paths.append(export_hf_jsonl(config, limit=args.limit))
    else:
        raise AssertionError(args.command)
    for path in paths:
        print(path)
    return 0
