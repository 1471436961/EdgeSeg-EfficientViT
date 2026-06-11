#!/usr/bin/env python3
"""Summarize TensorRT Nsight Systems SQLite attribution.

The script uses Nsight's TensorRT/NVTX layer ranges as structural boundaries
and attributes CUDA kernel time by joining CUDA runtime launches to kernels via
correlationId. It does not use NVTX range duration as GPU time.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EXCLUDED_NVTX_PREFIXES = ("trt/", "myelin")
EXCLUDED_NVTX_NAMES = {"ExecutionContext::enqueue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TensorRT Nsight SQLite attribution.")
    parser.add_argument("--sqlite", required=True, type=Path, help="Nsight-exported SQLite file.")
    parser.add_argument("--metrics", required=True, type=Path, help="benchmark_trt_engine.py JSON metadata.")
    parser.add_argument("--out-md", required=True, type=Path, help="Markdown summary output.")
    parser.add_argument("--out-json", default=None, type=Path, help="Optional JSON summary output.")
    parser.add_argument("--top-k", default=25, type=int, help="Number of layer/kernel rows to show.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def string_value_expr(column: str) -> str:
    return f"(select value from StringIds where id={column})"


def nvtx_name_expr() -> str:
    return "coalesce(text, (select value from StringIds where id=textId))"


def kernel_name_expr() -> str:
    return "coalesce((select value from StringIds where id=demangledName), (select value from StringIds where id=shortName))"


def kernel_durations_by_correlation(con: sqlite3.Connection) -> Dict[int, int]:
    durations: Dict[int, int] = {}
    rows = con.execute(
        """
        select correlationId, end - start
        from CUPTI_ACTIVITY_KIND_KERNEL
        where correlationId is not null
        """
    )
    for correlation_id, duration_ns in rows:
        durations[int(correlation_id)] = durations.get(int(correlation_id), 0) + int(duration_ns)
    return durations


def nvtx_intervals(con: sqlite3.Connection, name: str) -> List[Tuple[int, int]]:
    expr = nvtx_name_expr()
    rows = con.execute(
        f"""
        select start, end
        from NVTX_EVENTS
        where {expr} = ? and end is not null
        order by start
        """,
        (name,),
    )
    return [(int(start), int(end)) for start, end in rows]


def measured_intervals(intervals: Sequence[Tuple[int, int]], warmup: int, measure: int) -> List[Tuple[int, int]]:
    if len(intervals) >= warmup + measure:
        return list(intervals[warmup : warmup + measure])
    return list(intervals)


def runtime_correlation_ids_in_interval(con: sqlite3.Connection, start: int, end: int) -> Iterable[int]:
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


def nvtx_names_with_counts(con: sqlite3.Connection) -> List[Tuple[str, int]]:
    expr = nvtx_name_expr()
    rows = con.execute(
        f"""
        select {expr} as name, count(*) as count
        from NVTX_EVENTS
        where name is not null and end is not null
        group by name
        order by count desc, name
        """
    )
    return [(str(name), int(count)) for name, count in rows if name is not None]


def is_layer_range(name: str, count: int, warmup: int, measure: int) -> bool:
    if count < warmup + measure:
        return False
    if name in EXCLUDED_NVTX_NAMES or name.startswith(EXCLUDED_NVTX_PREFIXES):
        return False
    return (
        "/" in name
        or name.startswith("PWN")
        or name.startswith("Reformatting")
        or name.startswith("(Unnamed Layer")
    )


def normalize_layer_path(name: str) -> str:
    if " to " in name:
        return name.split(" to ", 1)[1]
    return name


def group_name(name: str) -> str:
    text = normalize_layer_path(name)
    if "/backbone/input_stem" in text:
        return "stem"
    match = re.search(r"/backbone/stages\.(\d+)", text)
    if match:
        return f"stage{match.group(1)}"
    if "/head/" in text:
        return "head"
    if text.startswith("(Unnamed Layer"):
        return "constant/unnamed"
    return "other"


def stage2_context_component(name: str) -> str:
    """Heuristic TensorRT layer-name mapping for stage2 LiteMLA context."""
    text = normalize_layer_path(name)
    if "/backbone/stages.2/" not in text or "/context_module/" not in text:
        return ""
    if "/qkv/" in text:
        return "qkv"
    if "/aggreg." in text:
        return "aggregation"
    if "/kernel_func" in text and "Relu" in text:
        return "relu_qk"
    if "Reformatting" in text:
        return "reformat"
    if "/proj/" in text:
        return "proj_add"
    if "/Div" in text or "/Add" in text:
        return "norm_add_div"
    if "MatMul" in text:
        return "matmul"
    if "Cast" in text:
        return "cast"
    if "Pad" in text:
        return "pad"
    if "Reshape" in text or "[Shuffle]" in text:
        return "reshape_shuffle"
    return "other_context"


def layer_attribution(con: sqlite3.Connection, metrics: Dict[str, Any]) -> Dict[str, Any]:
    warmup = int(metrics["timing"]["warmup"])
    measure = int(metrics["timing"]["measure"])
    latency_mean_ms = float(metrics["timing"]["latency_ms"]["mean"])
    latency_p50_ms = float(metrics["timing"]["latency_ms"]["p50"])
    kernel_durations = kernel_durations_by_correlation(con)

    execute_intervals = measured_intervals(nvtx_intervals(con, "trt/execute"), warmup, measure)
    execute_total_ns = 0
    execute_launches = 0
    for start, end in execute_intervals:
        for correlation_id in runtime_correlation_ids_in_interval(con, start, end):
            execute_launches += 1
            execute_total_ns += kernel_durations.get(correlation_id, 0)

    layers: List[Dict[str, Any]] = []
    for name, count in nvtx_names_with_counts(con):
        if not is_layer_range(name, count, warmup, measure):
            continue
        intervals = measured_intervals(nvtx_intervals(con, name), warmup, measure)
        total_ns = 0
        launches = 0
        matched_launches = 0
        for start, end in intervals:
            for correlation_id in runtime_correlation_ids_in_interval(con, start, end):
                launches += 1
                duration = kernel_durations.get(correlation_id)
                if duration is None:
                    continue
                matched_launches += 1
                total_ns += duration
        avg_ms = total_ns / 1e6 / len(intervals) if intervals else 0.0
        layers.append(
            {
                "name": name,
                "group": group_name(name),
                "count": len(intervals),
                "launches_per_iter": launches / len(intervals) if intervals else 0.0,
                "matched_launches_per_iter": matched_launches / len(intervals) if intervals else 0.0,
                "avg_kernel_ms": avg_ms,
            }
        )

    layer_total_avg_ms = sum(row["avg_kernel_ms"] for row in layers)
    for row in layers:
        row["share_of_layer_attributed_pct"] = (
            100.0 * row["avg_kernel_ms"] / layer_total_avg_ms if layer_total_avg_ms else 0.0
        )
        row["share_of_execute_kernel_pct"] = (
            100.0 * row["avg_kernel_ms"] / (execute_total_ns / 1e6 / len(execute_intervals))
            if execute_intervals and execute_total_ns
            else 0.0
        )
        row["share_of_latency_mean_pct"] = 100.0 * row["avg_kernel_ms"] / latency_mean_ms if latency_mean_ms else 0.0

    groups: Dict[str, Dict[str, Any]] = {}
    for row in layers:
        group = groups.setdefault(
            row["group"],
            {
                "group": row["group"],
                "layer_count": 0,
                "avg_kernel_ms": 0.0,
                "launches_per_iter": 0.0,
            },
        )
        group["layer_count"] += 1
        group["avg_kernel_ms"] += row["avg_kernel_ms"]
        group["launches_per_iter"] += row["launches_per_iter"]

    execute_avg_ms = execute_total_ns / 1e6 / len(execute_intervals) if execute_intervals else 0.0
    for row in groups.values():
        row["share_of_layer_attributed_pct"] = (
            100.0 * row["avg_kernel_ms"] / layer_total_avg_ms if layer_total_avg_ms else 0.0
        )
        row["share_of_execute_kernel_pct"] = 100.0 * row["avg_kernel_ms"] / execute_avg_ms if execute_avg_ms else 0.0
        row["share_of_latency_mean_pct"] = 100.0 * row["avg_kernel_ms"] / latency_mean_ms if latency_mean_ms else 0.0

    return {
        "warmup": warmup,
        "measure": measure,
        "precision": metrics.get("precision"),
        "latency_mean_ms": latency_mean_ms,
        "latency_p50_ms": latency_p50_ms,
        "execute_count": len(execute_intervals),
        "execute_intervals": execute_intervals,
        "execute_kernel_avg_ms": execute_avg_ms,
        "execute_launches_per_iter": execute_launches / len(execute_intervals) if execute_intervals else 0.0,
        "layer_attributed_avg_ms": layer_total_avg_ms,
        "layer_attribution_vs_execute_pct": 100.0 * layer_total_avg_ms / execute_avg_ms if execute_avg_ms else 0.0,
        "groups": sorted(groups.values(), key=lambda row: row["avg_kernel_ms"], reverse=True),
        "layers": sorted(layers, key=lambda row: row["avg_kernel_ms"], reverse=True),
    }


def stage2_context_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
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

    block_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
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

    components = sorted(component_map.values(), key=lambda row: row["avg_kernel_ms"], reverse=True)
    component_by_name = {row["component"]: row for row in components}

    def candidate_boundary(name: str, includes: Sequence[str]) -> Dict[str, Any]:
        rows = [component_by_name[item] for item in includes if item in component_by_name]
        return {
            "candidate": name,
            "includes": list(includes),
            "avg_kernel_ms": sum(row["avg_kernel_ms"] for row in rows),
            "launches_per_iter": sum(row["launches_per_iter"] for row in rows),
            "share_of_execute_kernel_pct": sum(row["share_of_execute_kernel_pct"] for row in rows),
            "share_of_latency_mean_pct": sum(row["share_of_latency_mean_pct"] for row in rows),
        }

    candidate_boundaries = [
        candidate_boundary("attention_core", ["relu_qk", "pad", "matmul", "norm_add_div"]),
        candidate_boundary("aggregation_only", ["aggregation"]),
        candidate_boundary(
            "aggregation_plus_attention_core",
            ["aggregation", "relu_qk", "pad", "matmul", "norm_add_div"],
        ),
        candidate_boundary("qkv_proj_overhead", ["qkv", "proj_add"]),
        candidate_boundary(
            "full_stage2_context",
            [
                "qkv",
                "aggregation",
                "relu_qk",
                "pad",
                "matmul",
                "norm_add_div",
                "proj_add",
                "cast",
                "reshape_shuffle",
            ],
        ),
    ]

    return {
        "total_avg_kernel_ms": sum(row["avg_kernel_ms"] for row in rows),
        "total_launches_per_iter": sum(row["launches_per_iter"] for row in rows),
        "total_share_of_execute_kernel_pct": sum(row["share_of_execute_kernel_pct"] for row in rows),
        "total_share_of_latency_mean_pct": sum(row["share_of_latency_mean_pct"] for row in rows),
        "components": components,
        "candidate_boundaries": sorted(candidate_boundaries, key=lambda row: row["avg_kernel_ms"], reverse=True),
        "blocks": sorted(block_map.values(), key=lambda row: row["block"]),
        "layers": sorted(rows, key=lambda row: row["avg_kernel_ms"], reverse=True),
    }


def correlations_in_intervals(con: sqlite3.Connection, intervals: Sequence[Tuple[int, int]]) -> List[int]:
    correlations: List[int] = []
    for start, end in intervals:
        correlations.extend(runtime_correlation_ids_in_interval(con, start, end))
    return correlations


def kernel_type_summary(
    con: sqlite3.Connection, measure: int, correlations: Sequence[int]
) -> List[Dict[str, Any]]:
    if not correlations:
        return []
    expr = kernel_name_expr()
    placeholders = ",".join("?" for _ in correlations)
    rows = con.execute(
        f"""
        select {expr} as name, count(*) as count, sum(end - start) as total_ns
        from CUPTI_ACTIVITY_KIND_KERNEL
        where correlationId in ({placeholders})
        group by name
        order by total_ns desc
        """,
        tuple(correlations),
    )
    out = []
    for name, count, total_ns in rows:
        total_ms = float(total_ns) / 1e6
        out.append(
            {
                "name": str(name),
                "count": int(count),
                "avg_kernel_ms_per_iter": total_ms / measure if measure else 0.0,
                "total_ms": total_ms,
            }
        )
    total = sum(row["total_ms"] for row in out)
    for row in out:
        row["share_pct"] = 100.0 * row["total_ms"] / total if total else 0.0
    return out


def event_summary_in_intervals(
    con: sqlite3.Connection, table: str, measure: int, intervals: Sequence[Tuple[int, int]]
) -> Tuple[int, float, int]:
    total_count = 0
    total_ns = 0
    total_bytes = 0
    for start, end in intervals:
        count, ns, bytes_ = con.execute(
            f"""
            select count(*), coalesce(sum(end - start), 0), coalesce(sum(bytes), 0)
            from {table}
            where start >= ? and start <= ?
            """,
            (start, end),
        ).fetchone()
        total_count += int(count)
        total_ns += int(ns)
        total_bytes += int(bytes_)
    return total_count, float(total_ns) / 1e6 / measure if measure else 0.0, total_bytes


def memory_summary(
    con: sqlite3.Connection, measure: int, intervals: Sequence[Tuple[int, int]]
) -> Dict[str, Any]:
    memcpy_count, memcpy_ms, memcpy_bytes = event_summary_in_intervals(
        con, "CUPTI_ACTIVITY_KIND_MEMCPY", measure, intervals
    )
    memset_count, memset_ms, memset_bytes = event_summary_in_intervals(
        con, "CUPTI_ACTIVITY_KIND_MEMSET", measure, intervals
    )
    return {
        "memcpy_count": memcpy_count,
        "memcpy_avg_ms_per_iter": memcpy_ms,
        "memcpy_bytes": memcpy_bytes,
        "memset_count": memset_count,
        "memset_avg_ms_per_iter": memset_ms,
        "memset_bytes": memset_bytes,
    }


def render_markdown(summary: Dict[str, Any], sqlite_path: Path, metrics_path: Path, top_k: int) -> str:
    lines = [
        "# TensorRT Nsight Attribution Summary",
        "",
        f"- SQLite: `{sqlite_path.as_posix()}`",
        f"- Metrics: `{metrics_path.as_posix()}`",
        f"- Precision: `{summary['precision']}`",
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

    stage2_context = summary["stage2_context"]
    lines.extend(
        [
            "",
            "## Stage2 Context Runtime Detail",
            "",
            f"- Total stage2 context kernel avg: {stage2_context['total_avg_kernel_ms']:.3f} ms / iter",
            f"- Total stage2 context launches: {stage2_context['total_launches_per_iter']:.1f} / iter",
            f"- Share of execute kernel time: {stage2_context['total_share_of_execute_kernel_pct']:.2f}%",
            "",
            "Component mapping is inferred from TensorRT layer names. It is a runtime attribution summary, not EngineInspector tactic metadata.",
            "`attention_core` is a TensorRT-side proxy for residual paths inside Phase 1 `relu_linear_att`; it does not replace the Phase 1 Plan D MVP candidates (`relu_linear_att-only` / `aggregation-only`) or the Phase 1 main performance boundary (`aggregation + cat + relu_linear_att`).",
            "",
            "### Stage2 Context Components",
            "",
            "| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stage2_context["components"]:
        lines.append(
            f"| `{row['component']}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} | "
            f"{row['layer_count']} |"
        )

    lines.extend(
        [
            "",
            "### Stage2 Context Candidate Boundaries",
            "",
            "| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in stage2_context["candidate_boundaries"]:
        includes = " + ".join(f"`{item}`" for item in row["includes"])
        lines.append(
            f"| `{row['candidate']}` | {includes} | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            "### Stage2 Context Blocks",
            "",
            "| Block | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stage2_context["blocks"]:
        lines.append(
            f"| `{row['block']}` | {row['avg_kernel_ms']:.3f} | "
            f"{row['share_of_execute_kernel_pct']:.2f}% | {row['launches_per_iter']:.1f} | "
            f"{row['layer_count']} |"
        )

    lines.extend(
        [
            "",
            "### Stage2 Context Layer Rows",
            "",
            "| Rank | Block | Component | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(stage2_context["layers"], 1):
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| {rank} | `{row['block']}` | `{row['component']}` | `{name}` | "
            f"{row['avg_kernel_ms']:.3f} | {row['share_of_execute_kernel_pct']:.2f}% | "
            f"{row['launches_per_iter']:.1f} |"
        )

    lines.extend(
        [
            "",
            f"## Top {top_k} CUDA Kernel Names",
            "",
            "| Rank | Kernel | Avg ms / iter | Share | Count |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["kernel_types"][:top_k], 1):
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| {rank} | `{name}` | {row['avg_kernel_ms_per_iter']:.3f} | "
            f"{row['share_pct']:.2f}% | {row['count']} |"
        )

    mem = summary["memory"]
    lines.extend(
        [
            "",
            "## Memory Activity",
            "",
            "| Type | Count | Avg ms / iter | Bytes |",
            "|---|---:|---:|---:|",
            f"| Memcpy | {mem['memcpy_count']} | {mem['memcpy_avg_ms_per_iter']:.3f} | {mem['memcpy_bytes']} |",
            f"| Memset | {mem['memset_count']} | {mem['memset_avg_ms_per_iter']:.3f} | {mem['memset_bytes']} |",
            "",
            "## Interpretation Notes",
            "",
            "- This summary uses TensorRT-emitted layer NVTX ranges, not PyTorch module hooks.",
            "- Group names are inferred from ONNX-like layer paths such as `/backbone/stages.2/...`.",
            "- This can answer residual hotspot trends after TensorRT, but it is not a one-to-one replay of Phase 1 Plan B/C/D ranges.",
            "- `attention_core` / `aggregation_plus_attention_core` are TensorRT-side residual-runtime proxy boundaries; they should be mapped back to Phase 1 Plan D as `relu_linear_att` internal residual paths and `aggregation + cat + relu_linear_att`, not treated as renamed Phase 1 MVP definitions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics = load_json(args.metrics)
    con = sqlite3.connect(args.sqlite)
    summary = layer_attribution(con, metrics)
    execute_intervals = summary.pop("execute_intervals")
    execute_correlations = correlations_in_intervals(con, execute_intervals)
    summary["kernel_types"] = kernel_type_summary(con, summary["measure"], execute_correlations)
    summary["memory"] = memory_summary(con, summary["measure"], execute_intervals)
    summary["stage2_context"] = stage2_context_summary(summary)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(summary, args.sqlite, args.metrics, args.top_k), encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"TensorRT Nsight attribution summary written: {args.out_md}")


if __name__ == "__main__":
    main()
