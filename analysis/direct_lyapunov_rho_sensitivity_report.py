from __future__ import annotations

import ast
import csv
import io
import math
import struct
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    REPO_ROOT
    / "Data"
    / "debug_exports"
    / "direct_lyapunov_mpc_bounded_three_scenario_single_setpoint_nominal"
)
FIGURE_ROOT = (
    REPO_ROOT / "report" / "figures" / "2026-05-01_direct_three_method_rho_sensitivity"
)

RUNS = [
    {"rho_label": "0.95", "rho": 0.95, "run_id": "20260501_001425"},
    {"rho_label": "0.98", "rho": 0.98, "run_id": "20260501_003638"},
    {"rho_label": "0.985", "rho": 0.985, "run_id": "20260501_002805"},
    {"rho_label": "0.99", "rho": 0.99, "run_id": "20260501_001948"},
]

DUPLICATE_RUNS = [
    {
        "rho_label": "0.98",
        "preferred_run_id": "20260501_003638",
        "duplicate_run_id": "20260501_000956",
    }
]

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

PANEL_BG = "#fbfbfd"
GRID = "#d7dbe5"
AXIS = "#2d3440"
TEXT = "#1f2430"
BOUND_COLOR = "#b35c00"
RATIO_COLOR = "#2458d3"
FAIL_COLOR = "#bf1e2d"
SETPOINT_PHYS = [4.5, 324.0]
TRANSIENT_WINDOW = 250
TAIL_START = 1800
TAIL_COUNT = 100
RHO_COLORS = {
    "0.95": "#bf1e2d",
    "0.98": "#c96b17",
    "0.985": "#2a7f62",
    "0.99": "#2458d3",
}


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


def as_float(value: str) -> float:
    return float(value) if value not in ("", "None", "null") else math.nan


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    weight = idx - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def prod(shape: Sequence[int]) -> int:
    out = 1
    for value in shape:
        out *= int(value)
    return out


def reshape_flat(values: Sequence[object], shape: Sequence[int]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values[: shape[0]])
    stride = prod(shape[1:])
    return [
        reshape_flat(values[idx * stride : (idx + 1) * stride], shape[1:])
        for idx in range(shape[0])
    ]


def read_npy_bytes(blob: bytes) -> object:
    handle = io.BytesIO(blob)
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("Unsupported NPY payload.")
    major = handle.read(1)[0]
    _minor = handle.read(1)[0]
    if major == 1:
        header_len = struct.unpack("<H", handle.read(2))[0]
    else:
        header_len = struct.unpack("<I", handle.read(4))[0]
    header = ast.literal_eval(handle.read(header_len).decode("latin1"))
    descr = str(header["descr"])
    shape = tuple(int(dim) for dim in header["shape"])
    raw = handle.read()
    count = prod(shape) if shape else 1

    if descr == "<f8":
        values = [pair[0] for pair in struct.iter_unpack("<d", raw)]
    elif descr == "<i8":
        values = [pair[0] for pair in struct.iter_unpack("<q", raw)]
    elif descr == "|b1":
        values = [bool(pair[0]) for pair in struct.iter_unpack("<?", raw)]
    else:
        raise ValueError(f"Unsupported NPY dtype {descr!r}.")

    if len(values) != count:
        raise ValueError(f"Unexpected NPY payload length for shape {shape!r}.")
    return reshape_flat(values, shape)


def read_npz_members(path: Path, members: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    with ZipFile(path) as archive:
        for member in members:
            out[member[:-4]] = read_npy_bytes(archive.read(member))
    return out


def fmt_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def downsample_indices(length: int, max_points: int = 600) -> List[int]:
    if length <= max_points:
        return list(range(length))
    stride = max(1, math.ceil(length / max_points))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def infer_rho_from_first_step(step_rows: Sequence[Dict[str, str]]) -> float:
    first = step_rows[0]
    v_k = as_float(first["V_k"])
    v_bound = as_float(first["V_bound"])
    if not math.isfinite(v_k) or abs(v_k) < 1e-15:
        raise ValueError("Cannot infer rho from first step because V_k is invalid.")
    return v_bound / v_k


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
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    for x_raw, y_raw in zip(xs, ys):
        if not math.isfinite(x_raw) or not math.isfinite(y_raw):
            continue
        x = chart_x + (x_raw - x_min) / x_span * chart_w
        y = chart_y + chart_h - (y_raw - y_min) / y_span * chart_h
        points.append((x, y))
    return points


def matrix_column(matrix: object, idx: int) -> List[float]:
    return [float(row[idx]) for row in matrix]  # type: ignore[index]


def vector_slice(values: object, start: int = 0, end: int | None = None) -> List[float]:
    seq = values[start:end] if end is not None else values[start:]  # type: ignore[index]
    return [float(value) for value in seq]


def tail_values(values: Sequence[float], count: int = TAIL_COUNT) -> List[float]:
    if len(values) <= count:
        return list(values)
    return list(values[-count:])


def mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def mean_abs(values: Sequence[float]) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def trace_bounds(traces: Sequence[Sequence[float]], refs: Sequence[float] | None = None) -> tuple[float, float]:
    finite: List[float] = []
    for trace in traces:
        finite.extend(value for value in trace if math.isfinite(value))
    if refs:
        finite.extend(value for value in refs if math.isfinite(value))
    if not finite:
        return (0.0, 1.0)
    y_min = min(finite)
    y_max = max(finite)
    if math.isclose(y_min, y_max, rel_tol=0.0, abs_tol=1e-12):
        pad = max(1e-3, abs(y_min) * 0.05, 0.05)
        return (y_min - pad, y_max + pad)
    pad = 0.08 * (y_max - y_min)
    return (y_min - pad, y_max + pad)


def draw_general_panel(
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
    y_min: float,
    y_max: float,
    zero_line: bool = False,
) -> None:
    svg_rect(parts, x, y, width, height, fill=PANEL_BG, stroke=GRID, stroke_width=1.0, rx=8)
    svg_text(parts, x + 16, y + 24, title, size=16, weight="600")
    svg_text(parts, x + 16, y + 42, subtitle, size=12, fill="#5d6574")

    chart_x = x + 58
    chart_y = y + 58
    chart_w = width - 82
    chart_h = height - 90

    for tick in [y_min, y_min + (y_max - y_min) / 2.0, y_max]:
        tick_y = chart_y + chart_h - (tick - y_min) / max(1e-12, y_max - y_min) * chart_h
        svg_line(parts, chart_x, tick_y, chart_x + chart_w, tick_y, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, chart_x - 8, tick_y + 4, f"{tick:.3f}".rstrip("0").rstrip("."), size=11, fill="#5d6574", anchor="end")

    x_min = x_values[0]
    x_max = x_values[-1]
    x_ticks = [x_min, x_min + (x_max - x_min) / 2.0, x_max]
    for tick in x_ticks:
        tick_x = chart_x + (tick - x_min) / max(1e-12, x_max - x_min) * chart_w
        svg_line(parts, tick_x, chart_y, tick_x, chart_y + chart_h, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, tick_x, chart_y + chart_h + 18, f"{int(round(tick))}", size=11, fill="#5d6574", anchor="middle")

    svg_line(parts, chart_x, chart_y, chart_x, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_line(parts, chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_text(parts, chart_x - 42, chart_y + 14, y_label, size=12, fill="#5d6574")

    if zero_line and y_min < 0.0 < y_max:
        zero_y = chart_y + chart_h - (0.0 - y_min) / max(1e-12, y_max - y_min) * chart_h
        svg_line(parts, chart_x, zero_y, chart_x + chart_w, zero_y, stroke=AXIS, stroke_width=1.2, dash="6 4")

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


def draw_time_panel(
    parts: List[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    x_values: Sequence[float],
    ratio_values: Sequence[float],
    rho: float,
    fail_x: Sequence[float],
    fail_ratio: Sequence[float],
) -> None:
    svg_rect(parts, x, y, width, height, fill=PANEL_BG, stroke=GRID, stroke_width=1.0, rx=8)

    title_y = y + 24
    subtitle_y = y + 42
    chart_x = x + 58
    chart_y = y + 58
    chart_w = width - 86
    chart_h = height - 84

    finite_ratios = [value for value in ratio_values if math.isfinite(value)]
    upper = max(rho + 0.03, percentile(finite_ratios, 0.98) * 1.03 if finite_ratios else rho + 0.03)
    upper = min(upper, 1.25)
    upper = max(upper, 1.02)
    lower = 0.0
    clipped_high = sum(1 for value in finite_ratios if value > upper)

    svg_text(parts, x + 16, title_y, title, size=16, weight="600")
    svg_text(parts, x + 16, subtitle_y, subtitle + f" | y-max={upper:.2f} | clipped={clipped_high}", size=12, fill="#5d6574")

    y_ticks = [lower, upper / 2.0, upper]
    for tick in y_ticks:
        tick_y = chart_y + chart_h - (tick - lower) / max(1e-12, upper - lower) * chart_h
        svg_line(parts, chart_x, tick_y, chart_x + chart_w, tick_y, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, chart_x - 10, tick_y + 4, f"{tick:.2f}", size=11, fill="#5d6574", anchor="end")

    x_ticks = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
    for tick in x_ticks:
        tick_x = chart_x + (tick - 0.0) / 2000.0 * chart_w
        svg_line(parts, tick_x, chart_y, tick_x, chart_y + chart_h, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, tick_x, chart_y + chart_h + 18, f"{int(tick)}", size=11, fill="#5d6574", anchor="middle")

    svg_line(parts, chart_x, chart_y, chart_x, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_line(parts, chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)

    sampled = downsample_indices(len(x_values))
    x_sampled = [x_values[i] for i in sampled]
    ratio_sampled = [min(ratio_values[i], upper) if math.isfinite(ratio_values[i]) else math.nan for i in sampled]
    bound_sampled = [rho for _ in sampled]

    ratio_points = scale_points(
        x_sampled,
        ratio_sampled,
        chart_x,
        chart_y,
        chart_w,
        chart_h,
        0.0,
        2000.0,
        lower,
        upper,
    )
    bound_points = scale_points(
        x_sampled,
        bound_sampled,
        chart_x,
        chart_y,
        chart_w,
        chart_h,
        0.0,
        2000.0,
        lower,
        upper,
    )
    svg_polyline(parts, bound_points, stroke=BOUND_COLOR, stroke_width=1.8)
    svg_polyline(parts, ratio_points, stroke=RATIO_COLOR, stroke_width=1.8)

    fail_points = scale_points(
        fail_x,
        [min(value, upper) for value in fail_ratio],
        chart_x,
        chart_y,
        chart_w,
        chart_h,
        0.0,
        2000.0,
        lower,
        upper,
    )
    for point_x, point_y in fail_points:
        svg_circle(parts, point_x, point_y, 2.2, fill=FAIL_COLOR)

    legend_y = y + height - 18
    svg_line(parts, x + 16, legend_y - 4, x + 40, legend_y - 4, stroke=RATIO_COLOR, stroke_width=2.0)
    svg_text(parts, x + 46, legend_y, "V_next / V_k", size=12)
    svg_line(parts, x + 168, legend_y - 4, x + 192, legend_y - 4, stroke=BOUND_COLOR, stroke_width=2.0)
    svg_text(parts, x + 198, legend_y, "rho", size=12)
    svg_circle(parts, x + 268, legend_y - 4, 2.4, fill=FAIL_COLOR)
    svg_text(parts, x + 280, legend_y, "violations", size=12)


def save_contraction_figure(
    case_key: str,
    case_label: str,
    run_payloads: Sequence[Dict[str, object]],
) -> Path:
    width = 1200
    height = 1120
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, f"{case_label}: Lyapunov contraction ratio across rho", size=22, weight="600")
    svg_text(
        parts,
        40,
        56,
        "Each panel plots V_next_first / V_k against the run-specific rho line. Red markers indicate first-step contraction violations.",
        size=13,
        fill="#5d6574",
    )

    panel_w = width - 80
    panel_h = 235
    panel_x = 40
    panel_y = 80

    for idx, payload in enumerate(run_payloads):
        step_rows = payload["step_rows"]
        x_values = [float(row["step"]) for row in step_rows]
        ratio_values: List[float] = []
        fail_x: List[float] = []
        fail_ratio: List[float] = []
        for row in step_rows:
            v_k = as_float(row["V_k"])
            v_next = as_float(row["V_next_first"])
            ratio = math.nan
            if math.isfinite(v_k) and abs(v_k) > 1e-15 and math.isfinite(v_next):
                ratio = v_next / v_k
            ratio_values.append(ratio)
            satisfied = row["first_step_contraction_satisfied"].strip().lower() == "true"
            if not satisfied and math.isfinite(ratio):
                fail_x.append(float(row["step"]))
                fail_ratio.append(ratio)

        summary = payload["summary"]
        subtitle = (
            f"rho={payload['rho_label']} | solver={summary['solver_success_rate']:.2%} | "
            f"hard={summary['hard_contraction_rate']:.2%} | bounded-LS={summary['bounded_solution_used_steps']}"
        )
        draw_time_panel(
            parts,
            x=panel_x,
            y=panel_y + idx * (panel_h + 18),
            width=panel_w,
            height=panel_h,
            title=f"Run {payload['run_id']}",
            subtitle=subtitle,
            x_values=x_values,
            ratio_values=ratio_values,
            rho=payload["rho"],
            fail_x=fail_x,
            fail_ratio=fail_ratio,
        )

    svg_footer(parts)
    out_path = FIGURE_ROOT / f"{case_key}_contraction_ratio_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def draw_metric_panel(
    parts: List[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    data_by_case: Dict[str, List[tuple[float, float]]],
    y_min: float,
    y_max: float,
    percent_axis: bool = False,
) -> None:
    svg_rect(parts, x, y, width, height, fill=PANEL_BG, stroke=GRID, stroke_width=1.0, rx=8)
    svg_text(parts, x + 16, y + 24, title, size=16, weight="600")
    svg_text(parts, x + 16, y + 42, y_label, size=12, fill="#5d6574")

    chart_x = x + 54
    chart_y = y + 58
    chart_w = width - 70
    chart_h = height - 90

    x_min = 0.95
    x_max = 0.99
    y_ticks = [y_min, y_min + (y_max - y_min) / 2.0, y_max]
    for tick in y_ticks:
        tick_y = chart_y + chart_h - (tick - y_min) / max(1e-12, y_max - y_min) * chart_h
        svg_line(parts, chart_x, tick_y, chart_x + chart_w, tick_y, stroke=GRID, stroke_width=1.0, dash="4 4")
        label = f"{tick * 100:.1f}%" if percent_axis else f"{tick:.2f}"
        svg_text(parts, chart_x - 8, tick_y + 4, label, size=11, fill="#5d6574", anchor="end")

    x_ticks = [0.95, 0.98, 0.985, 0.99]
    for tick in x_ticks:
        tick_x = chart_x + (tick - x_min) / max(1e-12, x_max - x_min) * chart_w
        svg_line(parts, tick_x, chart_y, tick_x, chart_y + chart_h, stroke=GRID, stroke_width=1.0, dash="4 4")
        svg_text(parts, tick_x, chart_y + chart_h + 18, f"{tick:.3f}".rstrip("0").rstrip("."), size=11, fill="#5d6574", anchor="middle")

    svg_line(parts, chart_x, chart_y, chart_x, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)
    svg_line(parts, chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke=AXIS, stroke_width=1.2)

    for case_key, points in data_by_case.items():
        xs = [pair[0] for pair in points]
        ys = [pair[1] for pair in points]
        scaled = scale_points(xs, ys, chart_x, chart_y, chart_w, chart_h, x_min, x_max, y_min, y_max)
        svg_polyline(parts, scaled, stroke=CASE_COLORS[case_key], stroke_width=2.0)
        for px, py in scaled:
            svg_circle(parts, px, py, 2.6, fill=CASE_COLORS[case_key])


def save_method_metric_figure(summary_rows: Sequence[Dict[str, object]]) -> Path:
    width = 1280
    height = 840
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, "Three-method rho sensitivity summary", size=22, weight="600")
    svg_text(
        parts,
        40,
        56,
        "Metrics are drawn from the four unique direct-MPC bundles. The 0.98 sweep uses the latest duplicate export 20260501_003638.",
        size=13,
        fill="#5d6574",
    )

    by_case: Dict[str, List[Dict[str, object]]] = {case_key: [] for case_key, _ in CASES}
    for row in summary_rows:
        by_case[str(row["case_name"])].append(row)
    for rows in by_case.values():
        rows.sort(key=lambda item: float(item["rho"]))

    def metric_points(metric: str) -> Dict[str, List[tuple[float, float]]]:
        return {
            case_key: [(float(row["rho"]), float(row[metric])) for row in rows]
            for case_key, rows in by_case.items()
        }

    draw_metric_panel(
        parts,
        x=40,
        y=82,
        width=580,
        height=300,
        title="Output RMSE mean",
        y_label="Lower is better",
        data_by_case=metric_points("output_rmse_mean"),
        y_min=0.0,
        y_max=1.35,
    )
    draw_metric_panel(
        parts,
        x=660,
        y=82,
        width=580,
        height=300,
        title="Solver success rate",
        y_label="Higher is better",
        data_by_case=metric_points("solver_success_rate"),
        y_min=0.90,
        y_max=1.00,
        percent_axis=True,
    )
    draw_metric_panel(
        parts,
        x=40,
        y=420,
        width=580,
        height=300,
        title="Target-reference inf mean",
        y_label="Lower is better",
        data_by_case=metric_points("target_reference_error_inf_mean"),
        y_min=0.0,
        y_max=2.5,
    )
    draw_metric_panel(
        parts,
        x=660,
        y=420,
        width=580,
        height=300,
        title="Bounded-LS steps",
        y_label="Fewer fallback-target steps is better",
        data_by_case=metric_points("bounded_solution_used_steps"),
        y_min=0.0,
        y_max=2000.0,
    )

    legend_x = 54
    legend_y = 770
    for idx, (case_key, case_label) in enumerate(CASES):
        x0 = legend_x + idx * 370
        svg_line(parts, x0, legend_y - 6, x0 + 28, legend_y - 6, stroke=CASE_COLORS[case_key], stroke_width=2.2)
        svg_text(parts, x0 + 38, legend_y, case_label, size=12)

    svg_footer(parts)
    out_path = FIGURE_ROOT / "method_metrics_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def payload_series(payload: Dict[str, object], key: str, component: int | None = None) -> List[float]:
    data = payload["arrays"][key]  # type: ignore[index]
    if component is None:
        return vector_slice(data)  # type: ignore[arg-type]
    return matrix_column(data, component)  # type: ignore[arg-type]


def save_output_overlay_figure(
    case_key: str,
    case_label: str,
    run_payloads: Sequence[Dict[str, object]],
) -> Path:
    width = 1280
    height = 760
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, f"{case_label}: output tracking and steady targets across rho", size=22, weight="600")
    svg_text(parts, 40, 56, "Solid lines are plant outputs y. Dashed lines are steady targets y_s. Black dashed line is the raw setpoint y_sp.", size=13, fill="#5d6574")

    x_values = [float(step) for step in range(TRANSIENT_WINDOW + 1)]
    for output_idx in range(2):
        traces: List[Dict[str, object]] = []
        y_traces: List[List[float]] = []
        refs = [SETPOINT_PHYS[output_idx]]
        for payload in run_payloads:
            rho_label = str(payload["rho_label"])
            color = RHO_COLORS[rho_label]
            y_series = payload_series(payload, "y_system", output_idx)[: TRANSIENT_WINDOW + 1]
            y_s_series = payload_series(payload, "y_target_phys_store", output_idx)[: TRANSIENT_WINDOW + 1]
            traces.append({"values": y_series, "color": color, "width": 2.2})
            traces.append({"values": y_s_series, "color": color, "width": 1.8, "dash": "8 5"})
            y_traces.extend([y_series, y_s_series])
        y_min, y_max = trace_bounds(y_traces, refs=refs)
        setpoint_trace = {"values": [SETPOINT_PHYS[output_idx]] * len(x_values), "color": AXIS, "width": 1.6, "dash": "4 4"}
        draw_general_panel(
            parts,
            x=40,
            y=88 + output_idx * 282,
            width=1200,
            height=250,
            title=f"Output y[{output_idx}]",
            subtitle=f"Transient window 0-{TRANSIENT_WINDOW} steps",
            x_values=x_values,
            traces=[setpoint_trace, *traces],
            y_label="physical",
            y_min=y_min,
            y_max=y_max,
        )

    legend_y = 690
    svg_line(parts, 54, legend_y - 4, 82, legend_y - 4, stroke=AXIS, stroke_width=1.8, dash="4 4")
    svg_text(parts, 92, legend_y, "y_sp", size=12)
    svg_line(parts, 156, legend_y - 4, 184, legend_y - 4, stroke=AXIS, stroke_width=2.0)
    svg_text(parts, 194, legend_y, "solid = y", size=12)
    svg_line(parts, 290, legend_y - 4, 318, legend_y - 4, stroke=AXIS, stroke_width=2.0, dash="8 5")
    svg_text(parts, 328, legend_y, "dashed = y_s", size=12)
    for idx, rho_label in enumerate(["0.95", "0.98", "0.985", "0.99"]):
        x0 = 470 + idx * 165
        svg_line(parts, x0, legend_y - 4, x0 + 28, legend_y - 4, stroke=RHO_COLORS[rho_label], stroke_width=2.4)
        svg_text(parts, x0 + 38, legend_y, f"rho={rho_label}", size=12)

    svg_footer(parts)
    out_path = FIGURE_ROOT / f"{case_key}_outputs_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def save_input_overlay_figure(
    case_key: str,
    case_label: str,
    run_payloads: Sequence[Dict[str, object]],
) -> Path:
    width = 1280
    height = 760
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, f"{case_label}: applied inputs and steady input targets across rho", size=22, weight="600")
    svg_text(parts, 40, 56, "Solid lines are applied physical inputs u. Dashed lines are steady targets u_s.", size=13, fill="#5d6574")

    x_values = [float(step) for step in range(TRANSIENT_WINDOW)]
    for input_idx in range(2):
        traces: List[Dict[str, object]] = []
        y_traces: List[List[float]] = []
        for payload in run_payloads:
            rho_label = str(payload["rho_label"])
            color = RHO_COLORS[rho_label]
            u_series = payload_series(payload, "u_applied_phys", input_idx)[:TRANSIENT_WINDOW]
            u_s_series = payload_series(payload, "u_target_phys_store", input_idx)[:TRANSIENT_WINDOW]
            traces.append({"values": u_series, "color": color, "width": 2.2})
            traces.append({"values": u_s_series, "color": color, "width": 1.8, "dash": "8 5"})
            y_traces.extend([u_series, u_s_series])
        y_min, y_max = trace_bounds(y_traces)
        draw_general_panel(
            parts,
            x=40,
            y=88 + input_idx * 282,
            width=1200,
            height=250,
            title=f"Input u[{input_idx}]",
            subtitle=f"Transient window 0-{TRANSIENT_WINDOW - 1} steps",
            x_values=x_values,
            traces=traces,
            y_label="physical",
            y_min=y_min,
            y_max=y_max,
        )

    legend_y = 690
    svg_line(parts, 54, legend_y - 4, 82, legend_y - 4, stroke=AXIS, stroke_width=2.0)
    svg_text(parts, 92, legend_y, "solid = u", size=12)
    svg_line(parts, 176, legend_y - 4, 204, legend_y - 4, stroke=AXIS, stroke_width=2.0, dash="8 5")
    svg_text(parts, 214, legend_y, "dashed = u_s", size=12)
    for idx, rho_label in enumerate(["0.95", "0.98", "0.985", "0.99"]):
        x0 = 390 + idx * 165
        svg_line(parts, x0, legend_y - 4, x0 + 28, legend_y - 4, stroke=RHO_COLORS[rho_label], stroke_width=2.4)
        svg_text(parts, x0 + 38, legend_y, f"rho={rho_label}", size=12)

    svg_footer(parts)
    out_path = FIGURE_ROOT / f"{case_key}_inputs_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def save_disturbance_overlay_figure(
    case_key: str,
    case_label: str,
    run_payloads: Sequence[Dict[str, object]],
) -> Path:
    width = 1280
    height = 760
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, f"{case_label}: steady disturbance targets d_s across rho", size=22, weight="600")
    svg_text(parts, 40, 56, "These are the saved steady disturbance targets used by the frozen-output-disturbance target solve.", size=13, fill="#5d6574")

    x_values = [float(step) for step in range(TRANSIENT_WINDOW)]
    for disturbance_idx in range(2):
        traces: List[Dict[str, object]] = []
        y_traces: List[List[float]] = []
        for payload in run_payloads:
            rho_label = str(payload["rho_label"])
            series = payload_series(payload, "d_target_store", disturbance_idx)[:TRANSIENT_WINDOW]
            traces.append({"values": series, "color": RHO_COLORS[rho_label], "width": 2.2})
            y_traces.append(series)
        y_min, y_max = trace_bounds(y_traces)
        draw_general_panel(
            parts,
            x=40,
            y=88 + disturbance_idx * 282,
            width=1200,
            height=250,
            title=f"Disturbance target d_s[{disturbance_idx}]",
            subtitle=f"Transient window 0-{TRANSIENT_WINDOW - 1} steps",
            x_values=x_values,
            traces=traces,
            y_label="target",
            y_min=y_min,
            y_max=y_max,
        )

    legend_y = 690
    for idx, rho_label in enumerate(["0.95", "0.98", "0.985", "0.99"]):
        x0 = 120 + idx * 165
        svg_line(parts, x0, legend_y - 4, x0 + 28, legend_y - 4, stroke=RHO_COLORS[rho_label], stroke_width=2.4)
        svg_text(parts, x0 + 38, legend_y, f"rho={rho_label}", size=12)

    svg_footer(parts)
    out_path = FIGURE_ROOT / f"{case_key}_disturbance_targets_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def save_delta_v_figure(
    case_key: str,
    case_label: str,
    run_payloads: Sequence[Dict[str, object]],
) -> Path:
    width = 1280
    height = 760
    parts = svg_header(width, height)
    svg_rect(parts, 0, 0, width, height, fill="#ffffff", stroke="none")
    svg_text(parts, 40, 34, f"{case_label}: predicted Lyapunov decrement across rho", size=22, weight="600")
    svg_text(parts, 40, 56, "Each curve plots ΔV_pred = V_next_first - V_k. The expected pattern is negative during the transient and near zero in the tail.", size=13, fill="#5d6574")

    windows = [
        ("Transient ΔV_pred", "Steps 0-250", 0, TRANSIENT_WINDOW + 1),
        ("Tail ΔV_pred", f"Steps {TAIL_START}-{TAIL_START + TAIL_COUNT - 1}", TAIL_START, TAIL_START + TAIL_COUNT),
    ]
    for panel_idx, (title, subtitle, start, end) in enumerate(windows):
        x_values = [float(step) for step in range(start, end)]
        traces: List[Dict[str, object]] = []
        y_traces: List[List[float]] = []
        for payload in run_payloads:
            rho_label = str(payload["rho_label"])
            v_k = payload_series(payload, "V_k")
            v_next = payload_series(payload, "V_next_first")
            delta_v = [v_next[idx] - v_k[idx] for idx in range(len(v_k))]
            panel_series = delta_v[start:end]
            traces.append({"values": panel_series, "color": RHO_COLORS[rho_label], "width": 2.2})
            y_traces.append(panel_series)
        y_min, y_max = trace_bounds(y_traces, refs=[0.0])
        draw_general_panel(
            parts,
            x=40,
            y=88 + panel_idx * 282,
            width=1200,
            height=250,
            title=title,
            subtitle=subtitle,
            x_values=x_values,
            traces=traces,
            y_label="ΔV",
            y_min=y_min,
            y_max=y_max,
            zero_line=True,
        )

    legend_y = 690
    svg_line(parts, 54, legend_y - 4, 82, legend_y - 4, stroke=AXIS, stroke_width=1.8, dash="6 4")
    svg_text(parts, 92, legend_y, "ΔV = 0", size=12)
    for idx, rho_label in enumerate(["0.95", "0.98", "0.985", "0.99"]):
        x0 = 220 + idx * 165
        svg_line(parts, x0, legend_y - 4, x0 + 28, legend_y - 4, stroke=RHO_COLORS[rho_label], stroke_width=2.4)
        svg_text(parts, x0 + 38, legend_y, f"rho={rho_label}", size=12)

    svg_footer(parts)
    out_path = FIGURE_ROOT / f"{case_key}_delta_v_by_rho.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    mapping_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    detailed_rows: List[Dict[str, object]] = []
    tail_target_rows: List[Dict[str, object]] = []
    contraction_payloads: Dict[str, List[Dict[str, object]]] = {case_key: [] for case_key, _ in CASES}
    npz_members = [
        "y_system.npy",
        "u_applied_phys.npy",
        "u_target_phys_store.npy",
        "x_target_store.npy",
        "d_target_store.npy",
        "V_k.npy",
        "V_next_first.npy",
        "y_target_phys_store.npy",
    ]

    for run_spec in RUNS:
        run_id = str(run_spec["run_id"])
        rho_label = str(run_spec["rho_label"])
        run_root = DATA_ROOT / run_id
        inferred = math.nan

        for case_key, case_label in CASES:
            step_table_path = run_root / case_key / "step_table.csv"
            comparison_table_path = run_root / "comparison_table.csv"
            arrays_path = run_root / case_key / "arrays.npz"

            summary_rows_raw = read_csv_rows(comparison_table_path)
            comparison_row = next(row for row in summary_rows_raw if row["case_name"] == case_key)
            step_rows = read_csv_rows(step_table_path)
            arrays = read_npz_members(arrays_path, npz_members)
            inferred = infer_rho_from_first_step(step_rows)
            y_system = arrays["y_system"]
            u_target_phys = arrays["u_target_phys_store"]
            x_target = arrays["x_target_store"]
            d_target = arrays["d_target_store"]
            v_k = vector_slice(arrays["V_k"])
            v_next = vector_slice(arrays["V_next_first"])
            delta_v_pred = [v_next[idx] - v_k[idx] for idx in range(len(v_k))]
            tail_y0_error = [row[0] - SETPOINT_PHYS[0] for row in y_system[-TAIL_COUNT:]]  # type: ignore[index]
            tail_y1_error = [row[1] - SETPOINT_PHYS[1] for row in y_system[-TAIL_COUNT:]]  # type: ignore[index]

            summary_record = {
                "run_id": run_id,
                "rho": float(run_spec["rho"]),
                "rho_label": rho_label,
                "rho_inferred": inferred,
                "case_name": case_key,
                "case_label": case_label,
                "reward_mean": float(comparison_row["reward_mean"]),
                "output0_rmse": float(comparison_row["output0_rmse"]),
                "output1_rmse": float(comparison_row["output1_rmse"]),
                "output_rmse_mean": float(comparison_row["output_rmse_mean"]),
                "solver_success_rate": float(comparison_row["solver_success_rate"]),
                "hard_contraction_rate": float(comparison_row["hard_contraction_rate"]),
                "target_reference_error_inf_mean": float(comparison_row["target_reference_error_inf_mean"]),
                "target_us_u_ref_inf_mean": float(comparison_row["target_us_u_ref_inf_mean"]),
                "target_xs_x_ref_inf_mean": float(comparison_row["target_xs_x_ref_inf_mean"]),
                "bounded_solution_used_steps": int(float(comparison_row["bounded_solution_used_steps"])),
                "exact_target_within_bounds_steps": int(float(comparison_row["exact_target_within_bounds_steps"])),
                "contraction_margin_min": min(as_float(row["contraction_margin"]) for row in step_rows if row["contraction_margin"] != ""),
                "violation_steps": sum(
                    1
                    for row in step_rows
                    if row["first_step_contraction_satisfied"].strip().lower() != "true"
                ),
            }
            summary_rows.append(summary_record)
            detailed_rows.append(
                {
                    **summary_record,
                    "delta_v_pred_min": min(delta_v_pred),
                    "delta_v_pred_tail_mean": mean(tail_values(delta_v_pred)),
                    "delta_v_pred_tail_abs_mean": mean_abs(tail_values(delta_v_pred)),
                    "tail_y0_error_mean": mean(tail_y0_error),
                    "tail_y1_error_mean": mean(tail_y1_error),
                    "tail_y0_error_abs_mean": mean_abs(tail_y0_error),
                    "tail_y1_error_abs_mean": mean_abs(tail_y1_error),
                }
            )
            tail_target_rows.append(
                {
                    "run_id": run_id,
                    "rho": float(run_spec["rho"]),
                    "rho_label": rho_label,
                    "case_name": case_key,
                    "case_label": case_label,
                    "u_s_phys_0_tail_mean": mean(matrix_column(u_target_phys, 0)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "u_s_phys_1_tail_mean": mean(matrix_column(u_target_phys, 1)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "d_s_0_tail_mean": mean(matrix_column(d_target, 0)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "d_s_1_tail_mean": mean(matrix_column(d_target, 1)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_0_tail_mean": mean(matrix_column(x_target, 0)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_1_tail_mean": mean(matrix_column(x_target, 1)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_2_tail_mean": mean(matrix_column(x_target, 2)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_3_tail_mean": mean(matrix_column(x_target, 3)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_4_tail_mean": mean(matrix_column(x_target, 4)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_5_tail_mean": mean(matrix_column(x_target, 5)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                    "x_s_6_tail_mean": mean(matrix_column(x_target, 6)[-TAIL_COUNT:]),  # type: ignore[arg-type]
                }
            )

            contraction_payloads[case_key].append(
                {
                    "run_id": run_id,
                    "rho": float(run_spec["rho"]),
                    "rho_label": rho_label,
                    "summary": summary_record,
                    "step_rows": step_rows,
                    "arrays": arrays,
                }
            )

        mapping_rows.append(
            {
                "rho_label": rho_label,
                "rho_configured": run_spec["rho"],
                "rho_inferred": inferred,
                "run_id": run_id,
                "note": "",
            }
        )

    for duplicate in DUPLICATE_RUNS:
        mapping_rows.append(
            {
                "rho_label": duplicate["rho_label"],
                "rho_configured": float(duplicate["rho_label"]),
                "rho_inferred": float(duplicate["rho_label"]),
                "run_id": duplicate["duplicate_run_id"],
                "note": f"Duplicate export of {duplicate['preferred_run_id']}; not used in report figures.",
            }
        )

    summary_fieldnames = [
        "run_id",
        "rho",
        "rho_label",
        "rho_inferred",
        "case_name",
        "case_label",
        "reward_mean",
        "output0_rmse",
        "output1_rmse",
        "output_rmse_mean",
        "solver_success_rate",
        "hard_contraction_rate",
        "target_reference_error_inf_mean",
        "target_us_u_ref_inf_mean",
        "target_xs_x_ref_inf_mean",
        "bounded_solution_used_steps",
        "exact_target_within_bounds_steps",
        "contraction_margin_min",
        "violation_steps",
    ]
    write_csv(FIGURE_ROOT / "rho_sweep_summary.csv", summary_rows, summary_fieldnames)
    write_csv(
        FIGURE_ROOT / "rho_sweep_detailed_summary.csv",
        detailed_rows,
        [
            *summary_fieldnames,
            "delta_v_pred_min",
            "delta_v_pred_tail_mean",
            "delta_v_pred_tail_abs_mean",
            "tail_y0_error_mean",
            "tail_y1_error_mean",
            "tail_y0_error_abs_mean",
            "tail_y1_error_abs_mean",
        ],
    )
    write_csv(
        FIGURE_ROOT / "rho_sweep_tail_targets.csv",
        tail_target_rows,
        [
            "run_id",
            "rho",
            "rho_label",
            "case_name",
            "case_label",
            "u_s_phys_0_tail_mean",
            "u_s_phys_1_tail_mean",
            "d_s_0_tail_mean",
            "d_s_1_tail_mean",
            "x_s_0_tail_mean",
            "x_s_1_tail_mean",
            "x_s_2_tail_mean",
            "x_s_3_tail_mean",
            "x_s_4_tail_mean",
            "x_s_5_tail_mean",
            "x_s_6_tail_mean",
        ],
    )
    write_csv(
        FIGURE_ROOT / "rho_run_mapping.csv",
        mapping_rows,
        ["rho_label", "rho_configured", "rho_inferred", "run_id", "note"],
    )

    for case_key, case_label in CASES:
        contraction_payloads[case_key].sort(key=lambda item: float(item["rho"]))
        save_contraction_figure(case_key, case_label, contraction_payloads[case_key])
        save_output_overlay_figure(case_key, case_label, contraction_payloads[case_key])
        save_input_overlay_figure(case_key, case_label, contraction_payloads[case_key])
        save_disturbance_overlay_figure(case_key, case_label, contraction_payloads[case_key])
        save_delta_v_figure(case_key, case_label, contraction_payloads[case_key])

    save_method_metric_figure(summary_rows)


if __name__ == "__main__":
    main()
