"""Chunk 9/10 demo: A/B/C/D comparison + reproducible report.

    python scripts/demo_eval.py
    python scripts/demo_eval.py --out-dir results

Runs every default system over every default case, prints the markdown
report and writes both JSON (full audit trail) and Markdown (summary) to
the output directory.

The harness uses deterministic ids inside a run and the mock model stack
in mock mode, so repeated offline runs produce the same metric values.

Systems:
  A = LLM only (no workspace)
  B = flat context (all units, no relations)
  C = vector RAG (semantic top-k only)
  D = full DRLW (plan + typed subgraph retrieval + workers + synthesis)
  ablation = DRLW with graph expansion turned off
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latent_workspace.config import get_settings
from latent_workspace.evaluation import EvaluationHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="directory for evaluation_report.json/.md (default: results)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the report without writing files",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.mock_mode:
        print("NOTE: DLRW_MOCK_MODE=false -> this runs against a live LLM.")
        print("      Metrics/reports are still produced, but calls may cost money.\n")

    report = EvaluationHarness(settings=settings).run()
    markdown = report.to_markdown()
    print(markdown)

    print("ranking (composite):")
    for position, summary in enumerate(report.ranking(), start=1):
        print(
            f"  {position}. {summary.system:24s} {summary.composite:.3f} "
            f"(cases={summary.cases}, errors={summary.errors}, "
            f"context_tokens={summary.mean_context_tokens:.0f})"
        )

    if not args.no_write:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "evaluation_report.json"
        md_path = out_dir / "evaluation_report.md"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(markdown, encoding="utf-8")
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
