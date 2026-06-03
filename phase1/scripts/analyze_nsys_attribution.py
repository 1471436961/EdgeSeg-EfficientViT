#!/usr/bin/env python3
"""
Summarize Nsight Systems SQLite attribution by NVTX range.

This script intentionally does NOT use NVTX range duration as GPU time.
NVTX ranges are structural boundaries. GPU attribution is computed by:

1. finding CUDA runtime launches whose start timestamp falls inside a range;
2. joining those launches to CUDA kernels with the same correlationId;
3. summing kernel durations per NVTX range.

The output is a Markdown table suitable for Phase 1 bottleneck notes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attribute Nsight CUDA kernel time to NVTX ranges.",
    )
    p.add_argument("--sqlite", required=True, help="Nsight-exported .sqlite file.")
    p.add_argument("--metrics", required=True, help="baseline_inference.py JSON.")
    p.add_argument(
        "--ranges",
        nargs="*",
        default=None,
        help=(
            "NVTX range names to include. Defaults to metrics.nvtx.component_ranges "
            "when present, otherwise all NVTX range names in the sqlite."
        ),
    )
    p.add_argument(
        "--out-md",
        default=None,
        help="Optional Markdown output path. Prints to stdout when omitted.",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional machine-readable JSON summary output path.",
    )
    return p.parse_args()


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_range_name_expr() -> str:
    return "coalesce(text, (select value from StringIds where id=textId))"


def all_nvtx_range_names(con: sqlite3.Connection) -> List[str]:
    expr = resolve_range_name_expr()
    rows = con.execute(
        f"""
        select distinct {expr} as name
        from NVTX_EVENTS
        where name is not null and end is not null
        order by name
        """
    )
    return [str(r[0]) for r in rows]


def selected_ranges(metrics: Dict[str, Any],
                    con: sqlite3.Connection,
                    explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return list(explicit)
    component_ranges = metrics.get("nvtx", {}).get("component_ranges") or []
    if component_ranges:
        return list(component_ranges)
    return all_nvtx_range_names(con)


def kernel_durations_by_correlation(
    con: sqlite3.Connection,
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    rows = con.execute(
        """
        select correlationId, end - start
        from CUPTI_ACTIVITY_KIND_KERNEL
        where correlationId is not null
        """
    )
    for correlation_id, duration_ns in rows:
        out[int(correlation_id)] = out.get(int(correlation_id), 0) + int(duration_ns)
    return out


def nvtx_intervals(con: sqlite3.Connection, name: str) -> List[Tuple[int, int]]:
    expr = resolve_range_name_expr()
    rows = con.execute(
        f"""
        select start, end
        from NVTX_EVENTS
        where {expr} = ? and end is not null
        order by start
        """,
        (name,),
    )
    return [(int(s), int(e)) for s, e in rows]


def measured_intervals(intervals: List[Tuple[int, int]],
                       warmup: int,
                       measure: int) -> List[Tuple[int, int]]:
    if len(intervals) >= warmup + measure:
        return intervals[warmup:warmup + measure]
    return intervals


def runtime_launches_in_interval(
    con: sqlite3.Connection,
    start: int,
    end: int,
) -> Iterable[int]:
    rows = con.execute(
        """
        select correlationId
        from CUPTI_ACTIVITY_KIND_RUNTIME
        where correlationId is not null and start >= ? and start <= ?
        """,
        (start, end),
    )
    for (correlation_id,) in rows:
        yield int(correlation_id)


def summarize_ranges(con: sqlite3.Connection,
                     metrics: Dict[str, Any],
                     ranges: Sequence[str]) -> Dict[str, Any]:
    warmup = int(metrics.get("args", {}).get("warmup", 0))
    measure = int(metrics.get("args", {}).get("measure", 0))
    mean_ms = float(metrics.get("timing", {}).get("ms", {}).get("mean", 0.0))
    kernel_durations = kernel_durations_by_correlation(con)

    rows: List[Dict[str, Any]] = []
    for name in ranges:
        intervals = measured_intervals(nvtx_intervals(con, name), warmup, measure)
        launch_count = 0
        matched_launch_count = 0
        total_kernel_ns = 0
        for start, end in intervals:
            for correlation_id in runtime_launches_in_interval(con, start, end):
                launch_count += 1
                duration = kernel_durations.get(correlation_id)
                if duration is None:
                    continue
                matched_launch_count += 1
                total_kernel_ns += duration

        count = len(intervals)
        total_ms = total_kernel_ns / 1e6
        avg_ms = total_ms / count if count else 0.0
        rows.append({
            "name": name,
            "count": count,
            "launches": launch_count,
            "matched_launches": matched_launch_count,
            "total_kernel_ms": total_ms,
            "avg_kernel_ms": avg_ms,
        })

    attributed_total_ms = sum(r["total_kernel_ms"] for r in rows)
    attributed_avg_ms = attributed_total_ms / measure if measure else 0.0
    for row in rows:
        row["share_of_attributed_pct"] = (
            100.0 * row["total_kernel_ms"] / attributed_total_ms
            if attributed_total_ms else 0.0
        )
        row["share_of_forward_mean_pct"] = (
            100.0 * row["avg_kernel_ms"] / mean_ms if mean_ms else 0.0
        )

    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        group_name = row["name"].split("/", 1)[0]
        group = groups.setdefault(group_name, {
            "name": group_name,
            "total_kernel_ms": 0.0,
            "avg_kernel_ms": 0.0,
            "share_of_attributed_pct": 0.0,
            "share_of_forward_mean_pct": 0.0,
        })
        group["total_kernel_ms"] += row["total_kernel_ms"]

    for group in groups.values():
        group["avg_kernel_ms"] = group["total_kernel_ms"] / measure if measure else 0.0
        group["share_of_attributed_pct"] = (
            100.0 * group["total_kernel_ms"] / attributed_total_ms
            if attributed_total_ms else 0.0
        )
        group["share_of_forward_mean_pct"] = (
            100.0 * group["avg_kernel_ms"] / mean_ms if mean_ms else 0.0
        )

    return {
        "metrics_file": metrics.get("args", {}).get("out"),
        "script_version": metrics.get("script_version"),
        "nvtx_level": metrics.get("nvtx", {}).get("level"),
        "warmup": warmup,
        "measure": measure,
        "forward_mean_ms": mean_ms,
        "attributed_kernel_avg_ms": attributed_avg_ms,
        "attributed_share_of_forward_mean_pct": (
            100.0 * attributed_avg_ms / mean_ms if mean_ms else 0.0
        ),
        "ranges": sorted(rows, key=lambda r: r["total_kernel_ms"], reverse=True),
        "groups": sorted(groups.values(),
                         key=lambda r: r["total_kernel_ms"],
                         reverse=True),
    }


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def render_markdown(summary: Dict[str, Any],
                    sqlite_path: Path,
                    metrics_path: Path) -> str:
    lines = [
        "# Nsight Attribution Summary",
        "",
        f"- SQLite: `{sqlite_path.as_posix()}`",
        f"- Metrics: `{metrics_path.as_posix()}`",
        f"- Script version: `{summary.get('script_version')}`",
        f"- NVTX level: `{summary.get('nvtx_level')}`",
        f"- Warmup / measure: {summary['warmup']} / {summary['measure']}",
        f"- CUDA Event forward mean: {fmt_ms(summary['forward_mean_ms'])} ms",
        "- Attribution method: CUDA runtime launch `correlationId` -> CUDA kernel duration -> NVTX range.",
        "- Note: NVTX range duration itself is not GPU component time.",
        "",
        "## Group Summary",
        "",
        "| Group | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| `{row['name']}` | {fmt_ms(row['avg_kernel_ms'])} | "
            f"{fmt_pct(row['share_of_attributed_pct'])} | "
            f"{fmt_pct(row['share_of_forward_mean_pct'])} |"
        )

    lines.extend([
        "",
        "## Range Summary",
        "",
        "| Range | Count | Launches | Matched | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["ranges"]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {row['launches']} | "
            f"{row['matched_launches']} | {fmt_ms(row['avg_kernel_ms'])} | "
            f"{fmt_pct(row['share_of_attributed_pct'])} | "
            f"{fmt_pct(row['share_of_forward_mean_pct'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    sqlite_path = Path(args.sqlite)
    metrics_path = Path(args.metrics)
    metrics = load_metrics(metrics_path)
    con = sqlite3.connect(sqlite_path)
    ranges = selected_ranges(metrics, con, args.ranges)
    summary = summarize_ranges(con, metrics, ranges)
    markdown = render_markdown(summary, sqlite_path, metrics_path)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
