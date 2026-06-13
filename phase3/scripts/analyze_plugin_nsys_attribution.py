#!/usr/bin/env python3
"""Summarize Phase 3 Plugin engine Nsight Systems SQLite attribution.

The attribution method follows Phase 1/2: TensorRT/NVTX layer range -> CUDA
runtime launch inside range -> CUDA kernel with the same correlationId. NVTX
range duration itself is not used as GPU component time.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

import analyze_trt_nsys_attribution as base_attr  # noqa: E402


PLUGIN_LAYER_NAME = "EdgesegReluLinearAttention_TRT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 3 Plugin engine Nsight SQLite attribution.")
    parser.add_argument("--sqlite", required=True, type=Path, help="Nsight-exported SQLite file.")
    parser.add_argument("--metrics", required=True, type=Path, help="benchmark_plugin_engine.py JSON metadata.")
    parser.add_argument("--out-md", required=True, type=Path, help="Markdown summary output.")
    parser.add_argument("--out-json", default=None, type=Path, help="Optional JSON summary output.")
    parser.add_argument(
        "--baseline-summary-json",
        default=Path("phase2/results/metrics/trt_nsys_attribution_summary.json"),
        type=Path,
        help="Optional Phase 2 TensorRT baseline attribution JSON for before/after comparison.",
    )
    parser.add_argument("--top-k", default=25, type=int)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adapt_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "timing" in raw:
        return raw
    plugin = raw.get("benchmark", {}).get("plugin")
    if not plugin:
        raise ValueError("metrics JSON does not contain benchmark.plugin timing")
    return {
        "precision": raw.get("precision", "fp32"),
        "timing": plugin["timing"],
        "benchmark_target": raw.get("benchmark_target"),
        "plugin": raw.get("plugin"),
        "engine": raw.get("engines", {}).get("plugin"),
    }


def stage2_context_component(name: str) -> str:
    text = base_attr.normalize_layer_path(name)
    if "/backbone/stages.2/" not in text or "/context_module/" not in text:
        return ""
    if PLUGIN_LAYER_NAME in text:
        return "relu_linear_att_plugin"
    if "/qkv/" in text:
        return "qkv"
    if "/aggreg." in text:
        return "aggregation"
    if "/proj/" in text:
        return "proj_add"
    if "Reformatting" in text:
        return "reformat"
    if "Cast" in text:
        return "cast"
    if "Reshape" in text or "[Shuffle]" in text:
        return "reshape_shuffle"
    return "other_context"


def stage2_plugin_context_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for row in summary["layers"]:
        component = stage2_context_component(row["name"])
        if not component:
            continue
        enriched = dict(row)
        enriched["component"] = component
        block_match = re.search(r"/backbone/stages\.2/op_list\.(\d+)/context_module/", row["name"])
        enriched["block"] = f"op_list.{block_match.group(1)}" if block_match else "unknown"
        rows.append(enriched)

    component_map: Dict[str, Dict[str, Any]] = {}
    block_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        component = component_map.setdefault(
            row["component"],
            {
                "component": row["component"],
                "layer_count": 0,
                "avg_kernel_ms": 0.0,
                "launches_per_iter": 0.0,
                "share_of_execute_kernel_pct": 0.0,
                "share_of_latency_mean_pct": 0.0,
            },
        )
        component["layer_count"] += 1
        component["avg_kernel_ms"] += row["avg_kernel_ms"]
        component["launches_per_iter"] += row["launches_per_iter"]
        component["share_of_execute_kernel_pct"] += row["share_of_execute_kernel_pct"]
        component["share_of_latency_mean_pct"] += row["share_of_latency_mean_pct"]

        block = block_map.setdefault(
            row["block"],
            {
                "block": row["block"],
                "layer_count": 0,
                "avg_kernel_ms": 0.0,
                "launches_per_iter": 0.0,
                "share_of_execute_kernel_pct": 0.0,
                "share_of_latency_mean_pct": 0.0,
            },
        )
        block["layer_count"] += 1
        block["avg_kernel_ms"] += row["avg_kernel_ms"]
        block["launches_per_iter"] += row["launches_per_iter"]
        block["share_of_execute_kernel_pct"] += row["share_of_execute_kernel_pct"]
        block["share_of_latency_mean_pct"] += row["share_of_latency_mean_pct"]

    components = sorted(component_map.values(), key=lambda item: item["avg_kernel_ms"], reverse=True)
    component_by_name = {item["component"]: item for item in components}

    def boundary(name: str, includes: Sequence[str]) -> Dict[str, Any]:
        selected = [component_by_name[item] for item in includes if item in component_by_name]
        return {
            "candidate": name,
            "includes": list(includes),
            "avg_kernel_ms": sum(item["avg_kernel_ms"] for item in selected),
            "launches_per_iter": sum(item["launches_per_iter"] for item in selected),
            "share_of_execute_kernel_pct": sum(item["share_of_execute_kernel_pct"] for item in selected),
            "share_of_latency_mean_pct": sum(item["share_of_latency_mean_pct"] for item in selected),
        }

    candidate_boundaries = [
        boundary("relu_linear_att_plugin_only", ["relu_linear_att_plugin"]),
        boundary("aggregation_only", ["aggregation"]),
        boundary("aggregation_plus_plugin_proxy", ["aggregation", "relu_linear_att_plugin"]),
        boundary("qkv_proj_overhead", ["qkv", "proj_add"]),
        boundary("full_stage2_context_plugin_path", ["qkv", "aggregation", "relu_linear_att_plugin", "proj_add"]),
    ]

    return {
        "total_avg_kernel_ms": sum(row["avg_kernel_ms"] for row in rows),
        "total_launches_per_iter": sum(row["launches_per_iter"] for row in rows),
        "total_share_of_execute_kernel_pct": sum(row["share_of_execute_kernel_pct"] for row in rows),
        "total_share_of_latency_mean_pct": sum(row["share_of_latency_mean_pct"] for row in rows),
        "components": components,
        "candidate_boundaries": sorted(candidate_boundaries, key=lambda item: item["avg_kernel_ms"], reverse=True),
        "blocks": sorted(block_map.values(), key=lambda item: item["block"]),
        "layers": sorted(rows, key=lambda item: item["avg_kernel_ms"], reverse=True),
    }


def plugin_kernel_types(con: sqlite3.Connection, measure: int, plugin_layers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intervals: List[Tuple[int, int]] = []
    for row in plugin_layers:
        intervals.extend(base_attr.measured_intervals(base_attr.nvtx_intervals(con, row["name"]), 20, measure))
    correlations = base_attr.correlations_in_intervals(con, intervals)
    return base_attr.kernel_type_summary(con, measure, correlations)


def baseline_comparison(current: Dict[str, Any], baseline_path: Path) -> Dict[str, Any]:
    if not baseline_path.is_file():
        return {"status": "missing", "baseline_summary_json": str(baseline_path)}

    baseline = load_json(baseline_path)
    base_ctx = baseline.get("stage2_context", {})
    cur_ctx = current.get("stage2_plugin_context", {})

    def find_boundary(ctx: Dict[str, Any], name: str) -> Dict[str, Any] | None:
        for row in ctx.get("candidate_boundaries", []):
            if row.get("candidate") == name:
                return row
        return None

    def find_component(ctx: Dict[str, Any], name: str) -> Dict[str, Any] | None:
        for row in ctx.get("components", []):
            if row.get("component") == name:
                return row
        return None

    base_attention = find_boundary(base_ctx, "attention_core")
    base_mid = find_boundary(base_ctx, "aggregation_plus_attention_core")
    cur_plugin = find_boundary(cur_ctx, "relu_linear_att_plugin_only")
    cur_mid = find_boundary(cur_ctx, "aggregation_plus_plugin_proxy")
    base_aggregation = find_component(base_ctx, "aggregation")
    cur_aggregation = find_component(cur_ctx, "aggregation")

    def diff_row(label: str, before: Dict[str, Any] | None, after: Dict[str, Any] | None) -> Dict[str, Any]:
        if not before or not after:
            return {"label": label, "status": "missing"}
        before_ms = float(before.get("avg_kernel_ms", before.get("total_avg_kernel_ms", 0.0)))
        after_ms = float(after.get("avg_kernel_ms", after.get("total_avg_kernel_ms", 0.0)))
        return {
            "label": label,
            "before_ms": before_ms,
            "after_ms": after_ms,
            "delta_ms": after_ms - before_ms,
            "speedup_before_over_after": before_ms / after_ms if after_ms else None,
            "before_launches_per_iter": before.get("launches_per_iter", before.get("total_launches_per_iter", 0.0)),
            "after_launches_per_iter": after.get("launches_per_iter", after.get("total_launches_per_iter", 0.0)),
        }

    return {
        "status": "ok",
        "baseline_summary_json": str(baseline_path),
        "rows": [
            diff_row("relu_linear_att_proxy: baseline attention_core -> plugin layer", base_attention, cur_plugin),
            diff_row("p1b_proxy: baseline aggregation_plus_attention_core -> aggregation_plus_plugin", base_mid, cur_mid),
            diff_row("aggregation_preserved", base_aggregation, cur_aggregation),
            diff_row("stage2_context_total", base_ctx, cur_ctx),
        ],
    }


def render_markdown(summary: Dict[str, Any], sqlite_path: Path, metrics_path: Path, top_k: int) -> str:
    lines = [
        "# Phase 3 Plugin Engine Nsight Attribution Summary",
        "",
        f"- SQLite: `{sqlite_path.as_posix()}`",
        f"- Metrics: `{metrics_path.as_posix()}`",
        f"- Precision: `{summary['precision']}`",
        f"- Benchmark target: `{summary.get('benchmark_target')}`",
        f"- Warmup / measure: {summary['warmup']} / {summary['measure']}",
        f"- CUDA Events latency mean / p50: {summary['latency_mean_ms']:.3f} ms / {summary['latency_p50_ms']:.3f} ms",
        f"- `trt/execute` kernel avg: {summary['execute_kernel_avg_ms']:.3f} ms / iter",
        f"- `trt/execute` launches: {summary['execute_launches_per_iter']:.1f} / iter",
        f"- Layer-attributed kernel avg: {summary['layer_attributed_avg_ms']:.3f} ms / iter",
        f"- Layer attribution / execute kernel time: {summary['layer_attribution_vs_execute_pct']:.2f}%",
        "",
        "Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.",
        "NVTX range duration itself is not used as GPU component time.",
        "",
        "## Group Summary",
        "",
        "| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| `{row['group']}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | "
            f"{row['share_of_latency_mean_pct']:.2f}% | "
            f"{row['launches_per_iter']:.1f} | {row['layer_count']} |"
        )

    ctx = summary["stage2_plugin_context"]
    lines.extend(
        [
            "",
            "## Stage2 Context Plugin Detail",
            "",
            f"- Total stage2 context kernel avg: {ctx['total_avg_kernel_ms']:.3f} ms / iter",
            f"- Total stage2 context launches: {ctx['total_launches_per_iter']:.1f} / iter",
            f"- Share of execute kernel time: {ctx['total_share_of_execute_kernel_pct']:.2f}%",
            "",
            "| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ctx["components"]:
        lines.append(
            f"| `{row['component']}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} | {row['layer_count']} |"
        )

    lines.extend(
        [
            "",
            "### Candidate Boundaries",
            "",
            "| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in ctx["candidate_boundaries"]:
        includes = " + ".join(f"`{item}`" for item in row["includes"])
        lines.append(
            f"| `{row['candidate']}` | {includes} | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            "### Plugin Layer Rows",
            "",
            "| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in [item for item in ctx["layers"] if item["component"] == "relu_linear_att_plugin"]:
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| `{row['block']}` | `{name}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            "### Plugin Kernel Names",
            "",
            "| Rank | Kernel | Avg ms / iter | Share | Count |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["plugin_kernel_types"], 1):
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| {rank} | `{name}` | {row['avg_kernel_ms_per_iter']:.3f} | "
            f"{row['share_pct']:.2f}% | {row['count']} |"
        )

    lines.extend(
        [
            "",
            "## Baseline TensorRT Comparison",
            "",
            "| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    comparison = summary.get("baseline_comparison", {})
    for row in comparison.get("rows", []):
        if row.get("status") == "missing":
            lines.append(f"| `{row['label']}` | missing | missing | missing | missing | missing | missing |")
            continue
        lines.append(
            f"| `{row['label']}` | {row['before_ms']:.3f} | {row['after_ms']:.3f} | "
            f"{row['delta_ms']:.3f} | {row['speedup_before_over_after']:.3f}x | "
            f"{row['before_launches_per_iter']:.1f} | {row['after_launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            f"## Top {top_k} TensorRT Layer Ranges",
            "",
            "| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["layers"][:top_k], 1):
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| {rank} | `{row['group']}` | `{name}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.",
            "- `relu_linear_att_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.",
            "- `aggregation_plus_plugin_proxy` is the Phase 3 proxy for the previous `aggregation + cat + relu_linear_att` middle-boundary candidate; `cat` is no longer a separate TensorRT layer at this boundary.",
            "- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics = adapt_metrics(load_json(args.metrics))
    con = sqlite3.connect(args.sqlite)
    summary = base_attr.layer_attribution(con, metrics)
    execute_intervals = summary.pop("execute_intervals")
    execute_correlations = base_attr.correlations_in_intervals(con, execute_intervals)
    summary["kernel_types"] = base_attr.kernel_type_summary(con, summary["measure"], execute_correlations)
    summary["memory"] = base_attr.memory_summary(con, summary["measure"], execute_intervals)
    summary["benchmark_target"] = metrics.get("benchmark_target")
    summary["stage2_plugin_context"] = stage2_plugin_context_summary(summary)
    plugin_layers = [row for row in summary["stage2_plugin_context"]["layers"] if row["component"] == "relu_linear_att_plugin"]
    summary["plugin_kernel_types"] = plugin_kernel_types(con, summary["measure"], plugin_layers)
    summary["baseline_comparison"] = baseline_comparison(summary, args.baseline_summary_json)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(summary, args.sqlite, args.metrics, args.top_k), encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Phase 3 Plugin Nsight attribution summary written: {args.out_md}")


if __name__ == "__main__":
    main()
