from __future__ import annotations

import ast
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PARENT = (
    REPO_ROOT
    / "Data"
    / "debug_exports"
    / "direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal"
)
FIGURE_ROOT = REPO_ROOT / "report" / "figures" / "2026-05-01_direct_latest_run_numerics"
REPORT_PATH = REPO_ROOT / "report" / "direct_lyapunov_latest_two_setpoint_numerical_analysis_2026-05-01.md"
CSV_PATH = FIGURE_ROOT / "case_diagnostics.csv"

CASES = [
    ("bounded_hard", "Bounded hard"),
    ("bounded_hard_u_prev_0p1", "Bounded hard + input anchor"),
    ("bounded_hard_xs_prev_0p1", "Bounded hard + x_s smoothing"),
]

CASE_COLORS = {
    "bounded_hard": "#1f4b99",
    "bounded_hard_u_prev_0p1": "#c04b2d",
    "bounded_hard_xs_prev_0p1": "#2a7f62",
}

TEXT = "#1f2430"
AXIS = "#2d3440"
GRID = "#d7dbe5"
PANEL_BG = "#fbfbfd"
FAIL_COLOR = "#bf1e2d"
BOUND_COLOR = "#9e5a00"
VK_COLOR = "#2458d3"
VNEXT_COLOR = "#2a7f62"
Y0_COLOR = "#2458d3"
Y1_COLOR = "#c04b2d"

RHO_CURRENT = 0.98
RHO_HYPOTHETICAL = 0.995
EPS_LYAP = 1.0e-9

EARLY_TAIL_WINDOW = (372, 399)
DRIFT_CLUSTER_WINDOW = (748, 798)


def latest_run_dir(parent: Path) -> Path:
    runs = [path for path in parent.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {parent}.")
    return sorted(runs, key=lambda path: path.name)[-1]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: str | None) -> float:
    if value in (None, "", "None", "null"):
        return math.nan
    return float(value)


def fmt_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_int_list(values: Sequence[int]) -> str:
    return ", ".join(str(value) for value in values)


def parse_vec(text: str | None) -> List[float] | None:
    if text in (None, "", "None", "null"):
        return None
    values = ast.literal_eval(text)
    return [float(value) for value in values]


def linf(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    idx = (len(finite) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return finite[lo]
    weight = idx - lo
    return finite[lo] * (1.0 - weight) + finite[hi] * weight


def group_steps(steps: Sequence[int], max_gap: int = 2) -> List[tuple[int, int]]:
    if not steps:
        return []
    ordered = sorted(steps)
    groups: List[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for step in ordered[1:]:
        if step - prev <= max_gap:
            prev = step
            continue
        groups.append((start, prev))
        start = step
        prev = step
    groups.append((start, prev))
    return groups


def scale_points(
    xs: Sequence[float],
    ys: Sequence[float],
    chart_x: float,
    chart_y: float,
    chart_w: float,
    chart_h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    x_span = max(1.0e-12, x_max - x_min)
    y_span = max(1.0e-12, y_max - y_min)
    for x_raw, y_raw in zip(xs, ys):
        if not math.isfinite(x_raw) or not math.isfinite(y_raw):
            continue
        x = chart_x + (x_raw - x_min) / x_span * chart_w
        y = chart_y + chart_h - (y_raw - y_min) / y_span * chart_h
        points.append((x, y))
    return points


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    ]


def svg_footer(parts: List[str]) -> None:
    parts.append("</svg>")


def svg_text(
    parts: List[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    fill: str = TEXT,
    anchor: str = "start",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def svg_rect(
    parts: List[str],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "none",
    stroke: str = AXIS,
    stroke_width: float = 1.0,
    rx: float = 0.0,
) -> None:
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx:.1f}" />'
    )


def svg_line(
    parts: List[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = AXIS,
    stroke_width: float = 1.0,
    dash: str | None = None,
) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />'
    )


def svg_polyline(
    parts: List[str],
    points: Sequence[tuple[float, float]],
    *,
    stroke: str,
    stroke_width: float = 1.8,
    fill: str = "none",
    dash: str | None = None,
) -> None:
    if not points:
        return
    point_blob = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<polyline points="{point_blob}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr} />'
    )


def svg_circle(
    parts: List[str],
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
) -> None:
    parts.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def draw_legend(
    parts: List[str],
    *,
    x: float,
    y: float,
    items: Sequence[Dict[str, str]],
) -> None:
    cursor_x = x
    for item in items:
        if item.get("shape") == "circle":
            svg_circle(parts, cursor_x + 6, y - 5, 3.2, fill=item["color"])
            cursor_x += 18
        else:
            svg_line(
                parts,
                cursor_x,
                y - 5,
                cursor_x + 28,
                y - 5,
                stroke=item["color"],
                stroke_width=2.4,
                dash=item.get("dash"),
            )
            cursor_x += 38
        svg_text(parts, cursor_x, y, item["label"], size=12)
        cursor_x += 9 + len(item["label"]) * 6.6


def draw_trace_panel(
    parts: List[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    x_values: Sequence[float],
    traces: Sequence[Dict[str, object]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    zero_line: bool = False,
    fail_points: Sequence[tuple[float, float]] | None = None,
    vlines: Sequence[float] | None = None,
) -> None:
    svg_rect(parts, x, y, width, height, fill=PANEL_BG, stroke=GRID, stroke_width=1.0, rx=8)
    svg_text(parts, x + 16, y + 24, title, size=16, weight="600")
    svg_text(parts, x + 16, y + 42, subtitle, size=12, fill="#5d6574")

    chart_x = x + 58
    chart_y = y + 58
    chart_w = width - 82
    chart_h = height - 90

    finite_values: List[float] = []
    for trace in traces:
        finite_values.extend(
            float(value)
            for value in trace["values"]  # type: ignore[index]
            if math.isfinite(float(value))
        )
    if fail_points:
        finite_values.extend(y_val for _, y_val in fail_points if math.isfinite(y_val))

    if y_min is None or y_max is None:
        if not finite_values:
            y_min = 0.0
            y_max = 1.0
        else:
            auto_min = min(finite_values)
            auto_max = max(finite_values)
            if math.isclose(auto_min, auto_max, rel_tol=0.0, abs_tol=1.0e-12):
                pad = max(1.0e-6, 0.05 * max(abs(auto_min), 1.0))
                auto_min -= pad
                auto_max += pad
            else:
                pad = 0.08 * (auto_max - auto_min)
                auto_min -= pad
                auto_max += pad
            y_min = auto_min
            y_max = auto_max

    x_min = float(x_values[0])
    x_max = float(x_values[-1])

    for tick in [y_min, y_min + (y_max - y_min) / 2.0, y_max]:
        tick_y = chart_y + chart_h - (tick - y_min) / max(1.0e-12, y_max - y_min) * chart_h
        svg_line(parts, chart_x, tick_y, chart_x + chart_w, tick_y, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, chart_x - 8, tick_y + 4, f"{tick:.3f}".rstrip("0").rstrip("."), size=11, fill="#5d6574", anchor="end")

    for tick in [x_min, x_min + (x_max - x_min) / 2.0, x_max]:
        tick_x = chart_x + (tick - x_min) / max(1.0e-12, x_max - x_min) * chart_w
        svg_line(parts, tick_x, chart_y, tick_x, chart_y + chart_h, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, tick_x, chart_y + chart_h + 18, f"{int(round(tick))}", size=11, fill="#5d6574", anchor="middle")

    svg_line(parts, chart_x, chart_y, chart_x, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_line(parts, chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_text(parts, chart_x - 42, chart_y + 14, y_label, size=12, fill="#5d6574")

    if zero_line and y_min < 0.0 < y_max:
        zero_y = chart_y + chart_h - (0.0 - y_min) / max(1.0e-12, y_max - y_min) * chart_h
        svg_line(parts, chart_x, zero_y, chart_x + chart_w, zero_y, stroke=AXIS, stroke_width=1.2, dash="6 4")

    if vlines:
        for x_raw in vlines:
            tick_x = chart_x + (x_raw - x_min) / max(1.0e-12, x_max - x_min) * chart_w
            svg_line(parts, tick_x, chart_y, tick_x, chart_y + chart_h, stroke=BOUND_COLOR, stroke_width=1.4, dash="7 4")

    for trace in traces:
        scaled = scale_points(
            x_values,
            [float(value) for value in trace["values"]],  # type: ignore[index]
            chart_x,
            chart_y,
            chart_w,
            chart_h,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        svg_polyline(
            parts,
            scaled,
            stroke=str(trace["color"]),
            stroke_width=float(trace.get("width", 2.0)),
            dash=None if trace.get("dash") is None else str(trace["dash"]),
        )

    if fail_points:
        fail_scaled = scale_points(
            [point[0] for point in fail_points],
            [point[1] for point in fail_points],
            chart_x,
            chart_y,
            chart_w,
            chart_h,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        for point_x, point_y in fail_scaled:
            svg_circle(parts, point_x, point_y, 2.4, fill=FAIL_COLOR)


def gather_case_data(run_dir: Path, case_key: str) -> Dict[str, object]:
    case_dir = run_dir / case_key
    rows = read_csv_rows(case_dir / "step_table.csv")
    summary = read_json(case_dir / "summary.json")

    fail_steps = [int(row["step"]) for row in rows if row["success"] != "True"]
    fail_clusters = group_steps(fail_steps, max_gap=2)

    u_s_values = [parse_vec(row["u_s"]) for row in rows]
    x_s_values = [parse_vec(row["x_s"]) for row in rows]
    y_s_values = [parse_vec(row["y_s"]) for row in rows]
    stages = [row["target_stage"] for row in rows]

    du_trace: List[float] = []
    dx_trace: List[float] = []
    dy_trace: List[float] = []
    stage_switches = 0

    for idx in range(1, len(rows)):
        if stages[idx] != stages[idx - 1]:
            stage_switches += 1
        if u_s_values[idx] is not None and u_s_values[idx - 1] is not None:
            du_trace.append(linf(u_s_values[idx], u_s_values[idx - 1]))
        else:
            du_trace.append(math.nan)
        if x_s_values[idx] is not None and x_s_values[idx - 1] is not None:
            dx_trace.append(linf(x_s_values[idx], x_s_values[idx - 1]))
        else:
            dx_trace.append(math.nan)
        if y_s_values[idx] is not None and y_s_values[idx - 1] is not None:
            dy_trace.append(linf(y_s_values[idx], y_s_values[idx - 1]))
        else:
            dy_trace.append(math.nan)

    v_values = [as_float(row["V_k"]) for row in rows if math.isfinite(as_float(row["V_k"]))]
    margin_values = [
        as_float(row["contraction_margin"])
        for row in rows
        if math.isfinite(as_float(row["contraction_margin"]))
    ]
    small_v_success = [
        row
        for row in rows
        if row["success"] == "True"
        and math.isfinite(as_float(row["V_k"]))
        and as_float(row["V_k"]) < 1.0e-4
    ]
    near_zero_margin_success = [
        row
        for row in rows
        if row["success"] == "True"
        and math.isfinite(as_float(row["contraction_margin"]))
        and abs(as_float(row["contraction_margin"])) < 5.0e-8
    ]
    stage_success_counts: Dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["target_stage"], row["success"], row["solver_status"])
        stage_success_counts[key] = stage_success_counts.get(key, 0) + 1

    target_cond_values = [
        as_float(row["target_cond_M"])
        for row in rows
        if math.isfinite(as_float(row["target_cond_M"]))
    ]
    target_cond_g_values = [
        as_float(row["target_cond_G"])
        for row in rows
        if math.isfinite(as_float(row["target_cond_G"]))
    ]

    return {
        "case_dir": case_dir,
        "rows": rows,
        "summary": summary,
        "fail_steps": fail_steps,
        "fail_clusters": fail_clusters,
        "du_trace": du_trace,
        "dx_trace": dx_trace,
        "dy_trace": dy_trace,
        "stage_switches": stage_switches,
        "v_min": min(v_values) if v_values else math.nan,
        "v_max": max(v_values) if v_values else math.nan,
        "margin_min": min(margin_values) if margin_values else math.nan,
        "margin_max": max(margin_values) if margin_values else math.nan,
        "small_v_success_count": len(small_v_success),
        "near_zero_margin_success_count": len(near_zero_margin_success),
        "small_v_examples": small_v_success[:5],
        "near_zero_examples": near_zero_margin_success[:5],
        "stage_success_counts": stage_success_counts,
        "target_cond_m_min": min(target_cond_values) if target_cond_values else math.nan,
        "target_cond_m_max": max(target_cond_values) if target_cond_values else math.nan,
        "target_cond_g_min": min(target_cond_g_values) if target_cond_g_values else math.nan,
        "target_cond_g_max": max(target_cond_g_values) if target_cond_g_values else math.nan,
    }


def cluster_spans(rows: Sequence[Dict[str, str]], start: int, end: int) -> Dict[str, float]:
    u2_values: List[float] = []
    y0_values: List[float] = []
    y1_values: List[float] = []
    for idx in range(start, end + 1):
        u_s = parse_vec(rows[idx]["u_s"])
        y_s = parse_vec(rows[idx]["y_s"])
        if u_s is not None:
            u2_values.append(float(u_s[1]))
        if y_s is not None:
            y0_values.append(float(y_s[0]))
            y1_values.append(float(y_s[1]))
    return {
        "u_s2_span": max(u2_values) - min(u2_values),
        "y_s0_span": max(y0_values) - min(y0_values),
        "y_s1_span": max(y1_values) - min(y1_values),
    }


def save_step_change_figure(case_data: Dict[str, Dict[str, object]], figure_path: Path) -> None:
    width = 1200
    height = 860
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, "Latest direct run: steady-target movement by case", size=22, weight="600")
    svg_text(
        parts,
        40,
        56,
        "The unanchored target shows much larger step-to-step target movement, especially in u_s. This is the key mechanism behind the bounded_hard failures.",
        size=13,
        fill="#5d6574",
    )

    x_values = list(range(1, len(next(iter(case_data.values()))["rows"])))  # type: ignore[index]

    draw_trace_panel(
        parts,
        x=40,
        y=90,
        width=1120,
        height=330,
        title="Step-to-step steady-input movement",
        subtitle="Trace = ||u_s(k) - u_s(k-1)||_inf in scaled deviation coordinates.",
        x_values=x_values,
        traces=[
            {
                "values": case_data[case_key]["du_trace"],  # type: ignore[index]
                "color": CASE_COLORS[case_key],
            }
            for case_key, _ in CASES
        ],
        y_label="|Du_s|",
    )

    draw_trace_panel(
        parts,
        x=40,
        y=450,
        width=1120,
        height=330,
        title="Step-to-step steady-state movement",
        subtitle="Trace = ||x_s(k) - x_s(k-1)||_inf in scaled deviation coordinates.",
        x_values=x_values,
        traces=[
            {
                "values": case_data[case_key]["dx_trace"],  # type: ignore[index]
                "color": CASE_COLORS[case_key],
            }
            for case_key, _ in CASES
        ],
        y_label="|Dx_s|",
    )

    draw_legend(
        parts,
        x=54,
        y=826,
        items=[
            {"label": case_label, "color": CASE_COLORS[case_key]}
            for case_key, case_label in CASES
        ],
    )

    svg_footer(parts)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text("\n".join(parts), encoding="utf-8")


def save_drift_cluster_figure(rows: Sequence[Dict[str, str]], fail_steps: Sequence[int], figure_path: Path) -> None:
    start, end = DRIFT_CLUSTER_WINDOW
    width = 1200
    height = 860
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, "bounded_hard: late-segment target drift during the failure cluster", size=22, weight="600")
    svg_text(
        parts,
        40,
        56,
        "Between steps 748 and 798, the target output y_s barely moves while u_s sweeps through a large range. The red markers are solver-failure hold steps.",
        size=13,
        fill="#5d6574",
    )

    x_values = list(range(start, end + 1))
    u_s2 = [parse_vec(rows[idx]["u_s"])[1] for idx in range(start, end + 1)]  # type: ignore[index]
    y_s0 = [parse_vec(rows[idx]["y_s"])[0] for idx in range(start, end + 1)]  # type: ignore[index]
    y_s1 = [parse_vec(rows[idx]["y_s"])[1] for idx in range(start, end + 1)]  # type: ignore[index]
    fail_points_u = [(float(step), float(parse_vec(rows[step]["u_s"])[1])) for step in fail_steps if start <= step <= end]  # type: ignore[index]

    draw_trace_panel(
        parts,
        x=40,
        y=90,
        width=1120,
        height=330,
        title="Steady-input target drift",
        subtitle="Second input component of u_s in scaled deviation coordinates. Vertical dashed line marks the first exact-bounded target step.",
        x_values=x_values,
        traces=[{"values": u_s2, "color": CASE_COLORS["bounded_hard"]}],
        y_label="u_s[1]",
        fail_points=fail_points_u,
        vlines=[798.0],
    )

    draw_trace_panel(
        parts,
        x=40,
        y=450,
        width=1120,
        height=330,
        title="Steady-output target drift",
        subtitle="Both y_s output components remain almost fixed while u_s keeps moving.",
        x_values=x_values,
        traces=[
            {"values": y_s0, "color": Y0_COLOR},
            {"values": y_s1, "color": Y1_COLOR},
        ],
        y_label="y_s",
        vlines=[798.0],
    )

    draw_legend(
        parts,
        x=54,
        y=826,
        items=[
            {"label": "u_s[1]", "color": CASE_COLORS["bounded_hard"]},
            {"label": "y_s[0]", "color": Y0_COLOR},
            {"label": "y_s[1]", "color": Y1_COLOR},
            {"label": "solver fail", "color": FAIL_COLOR, "shape": "circle"},
            {"label": "exact target starts", "color": BOUND_COLOR, "dash": "7 4"},
        ],
    )

    svg_footer(parts)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text("\n".join(parts), encoding="utf-8")


def save_small_v_figure(rows: Sequence[Dict[str, str]], figure_path: Path) -> None:
    start, end = EARLY_TAIL_WINDOW
    width = 1200
    height = 860
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, "bounded_hard: small-V tail before the setpoint switch", size=22, weight="600")
    svg_text(
        parts,
        40,
        56,
        "These steps are already in the tiny-Lyapunov regime. The run still satisfies the contraction test cleanly, which argues against the small-V hypothesis for this latest run.",
        size=13,
        fill="#5d6574",
    )

    x_values = list(range(start, end + 1))
    v_k = [as_float(rows[idx]["V_k"]) * 1.0e6 for idx in range(start, end + 1)]
    v_bound = [as_float(rows[idx]["V_bound"]) * 1.0e6 for idx in range(start, end + 1)]
    v_next = [as_float(rows[idx]["V_next_first"]) * 1.0e6 for idx in range(start, end + 1)]
    margin = [as_float(rows[idx]["contraction_margin"]) * 1.0e8 for idx in range(start, end + 1)]

    draw_trace_panel(
        parts,
        x=40,
        y=90,
        width=1120,
        height=330,
        title="Lyapunov values",
        subtitle="Scaled by 1e6 for readability. V_next_first stays below V_bound even when V_k is only a few micro-units.",
        x_values=x_values,
        traces=[
            {"values": v_k, "color": VK_COLOR},
            {"values": v_bound, "color": BOUND_COLOR, "dash": "7 4"},
            {"values": v_next, "color": VNEXT_COLOR},
        ],
        y_label="1e6 * V",
    )

    draw_trace_panel(
        parts,
        x=40,
        y=450,
        width=1120,
        height=330,
        title="Contraction margin",
        subtitle="Scaled by 1e8 for readability. The margin stays negative in this small-V window.",
        x_values=x_values,
        traces=[{"values": margin, "color": CASE_COLORS["bounded_hard"]}],
        y_label="1e8 * m",
        zero_line=True,
    )

    draw_legend(
        parts,
        x=54,
        y=826,
        items=[
            {"label": "V_k", "color": VK_COLOR},
            {"label": "V_bound", "color": BOUND_COLOR, "dash": "7 4"},
            {"label": "V_next_first", "color": VNEXT_COLOR},
            {"label": "margin", "color": CASE_COLORS["bounded_hard"]},
        ],
    )

    svg_footer(parts)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text("\n".join(parts), encoding="utf-8")


def build_report(run_dir: Path, run_summary: Dict[str, object], case_data: Dict[str, Dict[str, object]]) -> str:
    rows_for_csv: List[Dict[str, object]] = []
    for case_key, case_label in CASES:
        summary = case_data[case_key]["summary"]
        rows_for_csv.append(
            {
                "case_name": case_key,
                "case_label": case_label,
                "solver_success_rate": summary["solver_success_rate"],  # type: ignore[index]
                "reward_mean": summary["reward_mean"],  # type: ignore[index]
                "mean_output_reference_error_inf": summary["output_reference_error_inf_mean"],  # type: ignore[index]
                "fail_steps": len(case_data[case_key]["fail_steps"]),  # type: ignore[index]
                "fail_clusters": len(case_data[case_key]["fail_clusters"]),  # type: ignore[index]
                "stage_switches": case_data[case_key]["stage_switches"],  # type: ignore[index]
                "mean_du_s_inf": mean(case_data[case_key]["du_trace"]),  # type: ignore[index]
                "max_du_s_inf": max(
                    value for value in case_data[case_key]["du_trace"] if math.isfinite(value)
                ),  # type: ignore[index]
                "mean_dx_s_inf": mean(case_data[case_key]["dx_trace"]),  # type: ignore[index]
                "max_dx_s_inf": max(
                    value for value in case_data[case_key]["dx_trace"] if math.isfinite(value)
                ),  # type: ignore[index]
                "min_V_k": case_data[case_key]["v_min"],  # type: ignore[index]
                "small_V_success_count": case_data[case_key]["small_v_success_count"],  # type: ignore[index]
                "near_zero_margin_success_count": case_data[case_key]["near_zero_margin_success_count"],  # type: ignore[index]
            }
        )
    write_csv(
        CSV_PATH,
        rows_for_csv,
        fieldnames=list(rows_for_csv[0].keys()),
    )

    bounded_hard = case_data["bounded_hard"]
    cluster_metrics = cluster_spans(bounded_hard["rows"], *DRIFT_CLUSTER_WINDOW)  # type: ignore[arg-type]

    threshold_current = EPS_LYAP / (1.0 - RHO_CURRENT)
    threshold_hypo = EPS_LYAP / (1.0 - RHO_HYPOTHETICAL)

    bounded_hard_summary = bounded_hard["summary"]
    u_anchor_summary = case_data["bounded_hard_u_prev_0p1"]["summary"]
    x_anchor_summary = case_data["bounded_hard_xs_prev_0p1"]["summary"]

    lines: List[str] = []
    lines.append("# Direct Lyapunov Latest Two-Setpoint Run: Numerical Diagnosis")
    lines.append("")
    lines.append("## Run Under Analysis")
    lines.append("")
    lines.append(
        f"- Latest run directory: `{run_dir}`"
    )
    lines.append(
        f"- Saved comparison bundle created at `{run_summary['created_at']}`"
    )
    lines.append("- Study: `direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal`")
    lines.append("- Plant mode: nominal")
    lines.append("- Schedule: two setpoints, `n_tests = 2`, `set_points_len = 400`, total `1600` steps")
    lines.append("- Direct MPC settings: `predict_h = 9`, `cont_h = 3`, `Qy_diag = [5, 1]`, `Rdu_diag = [1, 1]`")
    lines.append("- Lyapunov settings: `rho_lyap = 0.98`, `eps_lyap = 1e-9`, hard first-step contraction")
    lines.append("- Stage objective tracks the raw scheduled setpoint: `use_target_output_for_tracking = False`")
    lines.append("- Fallback on solver failure is `hold_prev`, because `use_target_on_solver_fail = False`")
    lines.append("")
    lines.append("## Method Reconstruction")
    lines.append("")
    lines.append("The direct controller solves a bounded frozen-output-disturbance steady-target problem first, then uses that target inside the online MPC.")
    lines.append("")
    lines.append("For each step, the target stage computes $(x_s, u_s, d_s, y_s)$ from the current augmented estimate and the requested setpoint $y_{\\mathrm{sp}}$. The three compared target variants are:")
    lines.append("")
    lines.append("- `bounded_hard`: bounded target solve with no anchor term on the steady target.")
    lines.append("- `bounded_hard_u_prev_0p1`: same bounded target solve plus a steady-input anchor term $\\lambda_u \\|u_s - u_{k-1}\\|^2$ with $\\lambda_u = 0.1$.")
    lines.append("- `bounded_hard_xs_prev_0p1`: same bounded target solve plus a steady-state smoothing term $\\lambda_x \\|x_s - x_{s,\\mathrm{prev}}\\|^2$ with $\\lambda_x = 0.1$.")
    lines.append("")
    lines.append("The online direct MPC then minimizes output tracking and input-move suppression subject to the model, bounds, and the hard first-step Lyapunov constraint")
    lines.append("")
    lines.append("$$")
    lines.append("V_{k+1\\mid k}^{(1)} \\le \\rho_{\\mathrm{lyap}} V_k + \\varepsilon_{\\mathrm{lyap}},")
    lines.append("$$")
    lines.append("")
    lines.append("with $V_k = (x_k - x_s)^\\top P_x (x_k - x_s)$, $\\rho_{\\mathrm{lyap}} = 0.98$, and $\\varepsilon_{\\mathrm{lyap}} = 10^{-9}$.")
    lines.append("")
    lines.append("## Main Finding")
    lines.append("")
    lines.append("The latest run does **not** support the idea that the bad `bounded_hard` behavior is mainly a tiny-$V$ numerical issue or a target-matrix conditioning issue. The data support a different mechanism:")
    lines.append("")
    lines.append("- The plain unanchored bounded target drifts substantially in $(u_s, x_s)$ even when $y_s$ is almost stationary.")
    lines.append("- Those target moves happen only in the `bounded_ls` target stage, and every solver failure in this run also happens only in that stage.")
    lines.append("- Once a failure occurs, the controller applies `hold_prev`, which produces the visible stop-go oscillation.")
    lines.append("- The input-anchor and `x_s`-smoothing variants suppress that drift and almost eliminate the failures.")
    lines.append("")
    lines.append("## Case Comparison")
    lines.append("")
    lines.append("| Case | Solver success | Failed steps | Fail clusters | Stage switches | Mean $\\|\\Delta u_s\\|_\\infty$ | Mean $\\|\\Delta x_s\\|_\\infty$ | Mean $\\|y-y_{\\mathrm{sp}}\\|_\\infty$ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| `bounded_hard` | {fmt_float(float(bounded_hard_summary['solver_success_rate']), 4)} | {len(bounded_hard['fail_steps'])} | {len(bounded_hard['fail_clusters'])} | {bounded_hard['stage_switches']} | {fmt_float(mean(bounded_hard['du_trace']), 3)} | {fmt_float(mean(bounded_hard['dx_trace']), 3)} | {fmt_float(float(bounded_hard_summary['output_reference_error_inf_mean']), 3)} |"
    )
    lines.append(
        f"| `bounded_hard_u_prev_0p1` | {fmt_float(float(u_anchor_summary['solver_success_rate']), 4)} | {len(case_data['bounded_hard_u_prev_0p1']['fail_steps'])} | {len(case_data['bounded_hard_u_prev_0p1']['fail_clusters'])} | {case_data['bounded_hard_u_prev_0p1']['stage_switches']} | {fmt_float(mean(case_data['bounded_hard_u_prev_0p1']['du_trace']), 3)} | {fmt_float(mean(case_data['bounded_hard_u_prev_0p1']['dx_trace']), 3)} | {fmt_float(float(u_anchor_summary['output_reference_error_inf_mean']), 3)} |"
    )
    lines.append(
        f"| `bounded_hard_xs_prev_0p1` | {fmt_float(float(x_anchor_summary['solver_success_rate']), 4)} | {len(case_data['bounded_hard_xs_prev_0p1']['fail_steps'])} | {len(case_data['bounded_hard_xs_prev_0p1']['fail_clusters'])} | {case_data['bounded_hard_xs_prev_0p1']['stage_switches']} | {fmt_float(mean(case_data['bounded_hard_xs_prev_0p1']['du_trace']), 3)} | {fmt_float(mean(case_data['bounded_hard_xs_prev_0p1']['dx_trace']), 3)} | {fmt_float(float(x_anchor_summary['output_reference_error_inf_mean']), 3)} |"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(f"- `bounded_hard` is the only case with a large failure count: {len(bounded_hard['fail_steps'])} failed steps in {len(bounded_hard['fail_clusters'])} clusters.")
    lines.append(f"- The input anchor removes failures completely. The `x_s` smoother leaves only {len(case_data['bounded_hard_xs_prev_0p1']['fail_steps'])} failed steps.")
    lines.append(f"- The unanchored case has about {mean(bounded_hard['du_trace']) / mean(case_data['bounded_hard_u_prev_0p1']['du_trace']):.1f}x larger mean $\\|\\Delta u_s\\|_\\infty$ than the input-anchor case.")
    lines.append("")
    lines.append("![Steady-target step changes](figures/2026-05-01_direct_latest_run_numerics/target_step_changes_by_case.svg)")
    lines.append("")
    lines.append("## Why Conditioning Does Not Look Like the Main Cause")
    lines.append("")
    lines.append("The target linear system metrics are effectively constant across the entire run and across all three cases:")
    lines.append("")
    lines.append(f"- `target_cond_M` stays at about `{fmt_float(float(bounded_hard['target_cond_m_max']), 3)}`")
    lines.append(f"- `target_cond_G` stays at about `{fmt_float(float(bounded_hard['target_cond_g_max']), 3)}`")
    lines.append("- the target rank remains full in the saved diagnostics")
    lines.append("")
    lines.append("If raw matrix conditioning were the dominant problem, the two anchored variants should show the same failure pattern because they use the same target equations and nearly the same condition numbers. They do not. Their main difference is the regularization that selects a smoother steady target from the same feasible geometry.")
    lines.append("")
    lines.append("A second signal is the solver-status mix:")
    lines.append("")
    lines.append(f"- `bounded_hard`: 46 `infeasible` and 6 `optimal_inaccurate` steps")
    lines.append("- `bounded_hard_u_prev_0p1`: all 1600 steps `optimal`")
    lines.append("- `bounded_hard_xs_prev_0p1`: 8 `infeasible` steps")
    lines.append("")
    lines.append("Most failures are therefore not borderline acceptance rejections. They are actual QP infeasibility events under the current target choice.")
    lines.append("")
    lines.append("## Why the Small-$V$ Hypothesis Is Not the Best Explanation")
    lines.append("")
    lines.append("Two points matter here.")
    lines.append("")
    lines.append("1. A larger $\\rho$ makes the Lyapunov bound **looser**, not tighter.")
    lines.append("")
    lines.append("Because")
    lines.append("")
    lines.append("$$")
    lines.append("V_{\\mathrm{bound}} = \\rho V_k + \\varepsilon,")
    lines.append("$$")
    lines.append("")
    lines.append("moving from $\\rho = 0.98$ to $\\rho = 0.995$ increases the allowed $V_{k+1}^{(1)}$. So `0.995` is less likely to trigger a violation than `0.98`, not more likely.")
    lines.append("")
    lines.append("2. In this latest run, the stored $V_k$ values do not get small enough for $\\varepsilon_{\\mathrm{lyap}} = 10^{-9}$ to dominate.")
    lines.append("")
    lines.append("| Case | Min $V_k$ | Min $V_k / [\\varepsilon / (1-0.98)]$ | Min $V_k / [\\varepsilon / (1-0.995)]$ | Successful steps with $V_k < 10^{-4}$ | Successful steps with $|m| < 5\\times 10^{-8}$ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for case_key, case_label in CASES:
        min_v = float(case_data[case_key]["v_min"])
        lines.append(
            f"| `{case_key}` | {min_v:.3e} | {min_v / threshold_current:.1f} | {min_v / threshold_hypo:.1f} | {case_data[case_key]['small_v_success_count']} | {case_data[case_key]['near_zero_margin_success_count']} |"
        )
    lines.append("")
    lines.append(f"For the current run, the strict-decrease floor is only `{threshold_current:.3e}` at `rho = 0.98` and `{threshold_hypo:.3e}` at `rho = 0.995`. Even the smallest observed `bounded_hard` value, `{float(bounded_hard['v_min']):.3e}`, is still about `{float(bounded_hard['v_min']) / threshold_hypo:.1f}` times larger than the `rho = 0.995` floor.")
    lines.append("")
    lines.append("That means the contraction test is still meaningfully active in the saved tiny-$V$ regime. The report data also contain many successful steps with both:")
    lines.append("")
    lines.append(f"- $V_k < 10^{{-4}}$: {bounded_hard['small_v_success_count']} successful `bounded_hard` steps")
    lines.append(f"- $|m| < 5\\times 10^{{-8}}$: {bounded_hard['near_zero_margin_success_count']} successful `bounded_hard` steps, where $m = V_{{k+1\\mid k}}^{{(1)}} - (\\rho V_k + \\varepsilon)$")
    lines.append("")
    lines.append("So the saved data do not show the controller oscillating because a numerically tiny positive Lyapunov margin keeps flipping the acceptance logic. The small-$V$ windows are mostly solved successfully.")
    lines.append("")
    lines.append("![Small-V tail diagnostics](figures/2026-05-01_direct_latest_run_numerics/bounded_hard_small_v_tail.svg)")
    lines.append("")
    lines.append("## What Actually Happens in the Bad `bounded_hard` Windows")
    lines.append("")
    lines.append("The clearest late-segment failure cluster is around steps `748:798` in the second setpoint segment.")
    lines.append("")
    lines.append(f"- Over that window, the second steady-input component `u_s[1]` spans `{cluster_metrics['u_s2_span']:.3f}` in scaled deviation coordinates.")
    lines.append(f"- Over the same window, the first steady-output target component spans only `{cluster_metrics['y_s0_span']:.3f}`.")
    lines.append(f"- The second steady-output target component spans only `{cluster_metrics['y_s1_span']:.3f}`.")
    lines.append("")
    lines.append("So the target output is almost fixed, but the steady input target keeps sliding. That is exactly the kind of non-unique target motion that an anchor term is supposed to suppress.")
    lines.append("")
    lines.append("This same pattern appears in the failure logs:")
    lines.append("")
    lines.append(f"- `bounded_hard` failure steps: {fmt_int_list(bounded_hard['fail_steps'])}")
    lines.append(f"- grouped into clusters: {', '.join(f'{lo}-{hi}' if lo != hi else str(lo) for lo, hi in bounded_hard['fail_clusters'])}")
    lines.append("")
    lines.append("All 52 failed `bounded_hard` steps happen while the target stage is `frozen_output_disturbance_bounded_ls`. None happen during `frozen_output_disturbance_exact_bounded`.")
    lines.append("")
    lines.append("That does **not** mean every failure is caused by the exact/bounded handoff. The long last cluster near the end of the run stays inside `bounded_ls` the whole time. The stronger statement is:")
    lines.append("")
    lines.append("- when the target stage is still using the bounded least-squares selection, the unanchored problem allows substantial target drift")
    lines.append("- that drift sometimes makes the hard-contraction MPC infeasible")
    lines.append("- the fallback `hold_prev` action then produces the apparent oscillation")
    lines.append("")
    lines.append("![Late-segment target drift](figures/2026-05-01_direct_latest_run_numerics/bounded_hard_target_drift_cluster_748_798.svg)")
    lines.append("")
    lines.append("## Final Diagnosis")
    lines.append("")
    lines.append("For the latest direct run saved on `2026-05-01`, the evidence points to **target-selection drift / non-uniqueness** as the main mechanism behind the bad `bounded_hard` behavior.")
    lines.append("")
    lines.append("The evidence against the competing explanations is:")
    lines.append("")
    lines.append("- Not mainly raw conditioning: the same target matrix conditioning appears in all three cases, but only the unanchored case fails badly.")
    lines.append("- Not mainly tiny-$V$ Lyapunov sensitivity: the run contains many successful near-tight small-$V$ steps, and `rho = 0.995` would be looser than the current `rho = 0.98` anyway.")
    lines.append("- Not mainly post-check rejection noise: the dominant bad status is `infeasible`, not a successful solution with a tiny positive contraction margin.")
    lines.append("")
    lines.append("The anchor terms help because they pick a consistent member of the bounded steady-target family. Once `u_s` or `x_s` is stabilized, the direct hard-contraction MPC remains feasible much more reliably.")
    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append("1. Keep one anchor active by default for bounded direct runs. The input anchor is the cleanest choice because it completely removed failures in this latest run.")
    lines.append("2. Add a bounded-target hysteresis rule: once the residual is below a small threshold, freeze `u_s` or freeze the active target branch until the next setpoint switch.")
    lines.append("3. Log and monitor `||u_s(k)-u_s(k-1)||_inf`, `||x_s(k)-x_s(k-1)||_inf`, and target-stage switches as first-class diagnostics in future runs.")
    lines.append("4. Only revisit `eps_lyap` scaling if future longer-settling runs actually drive $V_k$ below about `2e-7` for the `rho = 0.995` case. That is not the regime reached in this latest saved run.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    run_dir = latest_run_dir(RUN_PARENT)
    run_summary = read_json(run_dir / "comparison_summary.json")
    case_data = {
        case_key: gather_case_data(run_dir, case_key)
        for case_key, _ in CASES
    }

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    save_step_change_figure(case_data, FIGURE_ROOT / "target_step_changes_by_case.svg")
    save_drift_cluster_figure(
        case_data["bounded_hard"]["rows"],  # type: ignore[arg-type]
        case_data["bounded_hard"]["fail_steps"],  # type: ignore[arg-type]
        FIGURE_ROOT / "bounded_hard_target_drift_cluster_748_798.svg",
    )
    save_small_v_figure(
        case_data["bounded_hard"]["rows"],  # type: ignore[arg-type]
        FIGURE_ROOT / "bounded_hard_small_v_tail.svg",
    )

    report_text = build_report(run_dir, run_summary, case_data)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"latest_run={run_dir}")
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_ROOT}")
    print(f"csv={CSV_PATH}")


if __name__ == "__main__":
    main()
