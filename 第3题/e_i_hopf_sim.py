"""Question 3: Hopf analysis and numerical verification for an E-I network.

The model is
    tau_E dv_E/dt = -v_E + [M_EE v_E - M_EI v_I + h_E]_+
    tau_I dv_I/dt = -v_I + [M_IE v_E - M_II v_I + h_I]_+.

The script scans tau_I and h_E, integrates the ODE, and writes CSV/PNG
artifacts used by the TeX report.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass
class EIParams:
    tau_e: float = 1.0
    tau_i: float = 1.0
    m_ee: float = 2.0
    m_ei: float = 1.5
    m_ie: float = 2.0
    m_ii: float = 1.0
    h_e: float = 0.25
    h_i: float = 0.20


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def vector_field(state: np.ndarray, params: EIParams) -> np.ndarray:
    v_e, v_i = state
    z_e = params.m_ee * v_e - params.m_ei * v_i + params.h_e
    z_i = params.m_ie * v_e - params.m_ii * v_i + params.h_i
    return np.array(
        [
            (-v_e + relu(z_e)) / params.tau_e,
            (-v_i + relu(z_i)) / params.tau_i,
        ],
        dtype=float,
    )


def rk4_step(state: np.ndarray, dt: float, params: EIParams) -> np.ndarray:
    k1 = vector_field(state, params)
    k2 = vector_field(state + 0.5 * dt * k1, params)
    k3 = vector_field(state + 0.5 * dt * k2, params)
    k4 = vector_field(state + dt * k3, params)
    next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # The continuous system keeps nonnegative rates nonnegative.  This guard
    # only removes tiny numerical undershoots caused by finite time steps.
    return np.maximum(next_state, 0.0)


def simulate(
    params: EIParams,
    t_end: float,
    dt: float,
    initial: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(t_end / dt)) + 1
    times = np.linspace(0.0, dt * (n_steps - 1), n_steps)
    states = np.zeros((n_steps, 2), dtype=float)
    states[0] = np.asarray(initial, dtype=float)

    for k in range(1, n_steps):
        states[k] = rk4_step(states[k - 1], dt, params)
    return times, states


def active_equilibrium(params: EIParams) -> tuple[np.ndarray, bool]:
    """Equilibrium in the active-active linear region."""
    k = params.m_ei * params.m_ie - (params.m_ee - 1.0) * (params.m_ii + 1.0)
    v_e = ((params.m_ii + 1.0) * params.h_e - params.m_ei * params.h_i) / k
    v_i = (params.m_ie * params.h_e - (params.m_ee - 1.0) * params.h_i) / k
    state = np.array([v_e, v_i], dtype=float)
    z_e = params.m_ee * v_e - params.m_ei * v_i + params.h_e
    z_i = params.m_ie * v_e - params.m_ii * v_i + params.h_i
    active = bool(v_e > 0.0 and v_i > 0.0 and z_e > 0.0 and z_i > 0.0)
    return state, active


def jacobian_active(params: EIParams) -> np.ndarray:
    return np.array(
        [
            [(params.m_ee - 1.0) / params.tau_e, -params.m_ei / params.tau_e],
            [params.m_ie / params.tau_i, -(params.m_ii + 1.0) / params.tau_i],
        ],
        dtype=float,
    )


def trace_det(params: EIParams) -> tuple[float, float]:
    jac = jacobian_active(params)
    return float(np.trace(jac)), float(np.linalg.det(jac))


def hopf_tau_i(params: EIParams) -> float:
    if params.m_ee <= 1.0:
        return float("nan")
    return params.tau_e * (params.m_ii + 1.0) / (params.m_ee - 1.0)


def amplitude_after_transient(states: np.ndarray, transient_fraction: float = 0.55) -> tuple[float, float, float]:
    start = int(len(states) * transient_fraction)
    tail = states[start:]
    amp_e = float(tail[:, 0].max() - tail[:, 0].min())
    amp_i = float(tail[:, 1].max() - tail[:, 1].min())
    return amp_e, amp_i, float(np.std(tail[:, 0]))


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scan_tau_i(
    base: EIParams,
    output_dir: Path,
    dt: float,
    t_end: float,
    initial: tuple[float, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    tau_values = np.linspace(0.8, 4.0, 33)
    for tau_i in tau_values:
        params = EIParams(**{**asdict(base), "tau_i": float(tau_i)})
        eq, active = active_equilibrium(params)
        trace, det = trace_det(params)
        eig = np.linalg.eigvals(jacobian_active(params))
        times, states = simulate(params, t_end=t_end, dt=dt, initial=initial)
        amp_e, amp_i, std_e = amplitude_after_transient(states)
        rows.append(
            {
                "tau_i": float(tau_i),
                "trace": trace,
                "determinant": det,
                "eig_real_max": float(np.max(np.real(eig))),
                "eig_imag_abs": float(np.max(np.abs(np.imag(eig)))),
                "eq_v_e": float(eq[0]),
                "eq_v_i": float(eq[1]),
                "active_equilibrium": int(active),
                "amp_v_e": amp_e,
                "amp_v_i": amp_i,
                "std_v_e": std_e,
            }
        )

    write_csv(output_dir / "tau_i_scan.csv", rows)

    for tau_i, name in [(1.5, "stable"), (2.8, "oscillatory")]:
        params = EIParams(**{**asdict(base), "tau_i": tau_i})
        times, states = simulate(params, t_end=t_end, dt=dt, initial=initial)
        save_time_series(
            times,
            states,
            output_dir / f"time_series_tau_{name}.png",
            title=f"tau_I={tau_i:.2f} ({name})",
        )
        save_phase_plot(
            states,
            output_dir / f"phase_tau_{name}.png",
            title=f"phase portrait: tau_I={tau_i:.2f}",
        )
    return rows


def scan_h_e(
    base: EIParams,
    output_dir: Path,
    dt: float,
    t_end: float,
    initial: tuple[float, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    h_values = np.linspace(0.05, 0.60, 34)
    tau_i = 1.5
    for h_e in h_values:
        params = EIParams(**{**asdict(base), "tau_i": tau_i, "h_e": float(h_e)})
        eq, active = active_equilibrium(params)
        trace, det = trace_det(params)
        eig = np.linalg.eigvals(jacobian_active(params))
        times, states = simulate(params, t_end=t_end, dt=dt, initial=initial)
        amp_e, amp_i, std_e = amplitude_after_transient(states)
        rows.append(
            {
                "h_e": float(h_e),
                "tau_i": tau_i,
                "trace": trace,
                "determinant": det,
                "eig_real_max": float(np.max(np.real(eig))),
                "eig_imag_abs": float(np.max(np.abs(np.imag(eig)))),
                "eq_v_e": float(eq[0]),
                "eq_v_i": float(eq[1]),
                "active_equilibrium": int(active),
                "amp_v_e": amp_e,
                "amp_v_i": amp_i,
                "std_v_e": std_e,
            }
        )

    write_csv(output_dir / "h_e_scan.csv", rows)
    return rows


def scale_points(
    x: np.ndarray,
    y: np.ndarray,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    xmin, xmax = x_range or (float(x.min()), float(x.max()))
    ymin, ymax = y_range or (float(y.min()), float(y.max()))
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    return [
        (
            x0 + (float(xi) - xmin) / (xmax - xmin) * (x1 - x0),
            y1 - (float(yi) - ymin) / (ymax - ymin) * (y1 - y0),
        )
        for xi, yi in zip(x, y)
    ]


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    font = ImageFont.load_default()
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(30, 30, 30))
    draw.text((x0 + (x1 - x0) // 2 - 80, 14), title, fill=(0, 0, 0), font=font)
    draw.text((x0 + (x1 - x0) // 2 - 20, y1 + 30), xlabel, fill=(0, 0, 0), font=font)
    draw.text((15, y0 + (y1 - y0) // 2), ylabel, fill=(0, 0, 0), font=font)


def save_tau_scan_plot(rows: list[dict[str, float | int | str]], hopf: float, path: Path) -> None:
    if Image is None:
        return
    width, height = 900, 540
    box = (75, 45, 855, 455)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_axes(draw, box, "tau_I scan: eigenvalue and oscillation amplitude", "tau_I", "scaled value")

    tau = np.array([float(r["tau_i"]) for r in rows])
    amp = np.array([float(r["amp_v_e"]) for r in rows])
    real = np.array([float(r["eig_real_max"]) for r in rows])
    real_scaled = (real - real.min()) / max(1e-12, real.max() - real.min()) * max(amp.max(), 1.0)
    ymax = max(float(amp.max()), float(real_scaled.max()), 1.0) * 1.1
    xr = (float(tau.min()), float(tau.max()))
    yr = (0.0, ymax)

    amp_pts = scale_points(tau, amp, box, xr, yr)
    real_pts = scale_points(tau, real_scaled, box, xr, yr)
    draw.line(amp_pts, fill=(40, 120, 220), width=3)
    draw.line(real_pts, fill=(220, 70, 60), width=3)

    x_h = scale_points(np.array([hopf]), np.array([0.0]), box, xr, yr)[0][0]
    draw.line((x_h, box[1], x_h, box[3]), fill=(50, 50, 50), width=2)
    font = ImageFont.load_default()
    draw.text((x_h + 5, box[1] + 10), f"Hopf tau_I={hopf:.2f}", fill=(50, 50, 50), font=font)
    draw.line((650, 70, 690, 70), fill=(40, 120, 220), width=3)
    draw.text((700, 63), "v_E amplitude", fill=(0, 0, 0), font=font)
    draw.line((650, 94, 690, 94), fill=(220, 70, 60), width=3)
    draw.text((700, 87), "scaled max Re(lambda)", fill=(0, 0, 0), font=font)
    image.save(path)


def save_h_scan_plot(rows: list[dict[str, float | int | str]], h_boundary: float, path: Path) -> None:
    if Image is None:
        return
    width, height = 900, 540
    box = (75, 45, 855, 455)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_axes(draw, box, "h_E scan: no Hopf in a fixed active region", "h_E", "value")

    h = np.array([float(r["h_e"]) for r in rows])
    amp = np.array([float(r["amp_v_e"]) for r in rows])
    real = np.array([float(r["eig_real_max"]) for r in rows])
    active = np.array([int(r["active_equilibrium"]) for r in rows])

    ymax = max(float(amp.max()), float(np.abs(real).max()), 1.0) * 1.1
    xr = (float(h.min()), float(h.max()))
    yr = (-0.25 * ymax, ymax)
    amp_pts = scale_points(h, amp, box, xr, yr)
    real_pts = scale_points(h, real, box, xr, yr)
    draw.line(amp_pts, fill=(40, 120, 220), width=3)
    draw.line(real_pts, fill=(220, 70, 60), width=3)

    # Mark active-active region threshold.
    x_b = scale_points(np.array([h_boundary]), np.array([0.0]), box, xr, yr)[0][0]
    draw.line((x_b, box[1], x_b, box[3]), fill=(50, 50, 50), width=2)
    font = ImageFont.load_default()
    draw.text((x_b + 5, box[1] + 12), "active boundary", fill=(50, 50, 50), font=font)
    draw.line((650, 70, 690, 70), fill=(40, 120, 220), width=3)
    draw.text((700, 63), "v_E amplitude", fill=(0, 0, 0), font=font)
    draw.line((650, 94, 690, 94), fill=(220, 70, 60), width=3)
    draw.text((700, 87), "max Re(lambda)", fill=(0, 0, 0), font=font)

    # Small ticks showing active equilibrium status.
    for hi, ac in zip(h, active):
        x = scale_points(np.array([hi]), np.array([0.0]), box, xr, yr)[0][0]
        color = (40, 150, 80) if ac else (180, 180, 180)
        draw.line((x, box[3] + 6, x, box[3] + 12), fill=color, width=2)
    image.save(path)


def save_time_series(times: np.ndarray, states: np.ndarray, path: Path, title: str) -> None:
    if Image is None:
        return
    width, height = 900, 520
    box = (75, 45, 855, 445)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_axes(draw, box, title, "time", "rate")

    tail_start = int(len(times) * 0.15)
    t = times[tail_start:]
    ve = states[tail_start:, 0]
    vi = states[tail_start:, 1]
    yr = (0.0, max(float(ve.max()), float(vi.max()), 0.1) * 1.12)
    xr = (float(t.min()), float(t.max()))
    draw.line(scale_points(t, ve, box, xr, yr), fill=(220, 70, 60), width=2)
    draw.line(scale_points(t, vi, box, xr, yr), fill=(40, 120, 220), width=2)
    font = ImageFont.load_default()
    draw.line((650, 70, 690, 70), fill=(220, 70, 60), width=2)
    draw.text((700, 63), "v_E", fill=(0, 0, 0), font=font)
    draw.line((650, 94, 690, 94), fill=(40, 120, 220), width=2)
    draw.text((700, 87), "v_I", fill=(0, 0, 0), font=font)
    image.save(path)


def save_phase_plot(states: np.ndarray, path: Path, title: str) -> None:
    if Image is None:
        return
    width, height = 620, 560
    box = (70, 45, 575, 500)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_axes(draw, box, title, "v_E", "v_I")

    start = int(len(states) * 0.15)
    data = states[start:]
    margin = 0.05
    xr = (max(0.0, float(data[:, 0].min()) - margin), float(data[:, 0].max()) + margin)
    yr = (max(0.0, float(data[:, 1].min()) - margin), float(data[:, 1].max()) + margin)
    pts = scale_points(data[:, 0], data[:, 1], box, xr, yr)
    if len(pts) > 1:
        draw.line(pts, fill=(80, 90, 190), width=2)
    # Mark last point.
    x, y = pts[-1]
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(220, 70, 60))
    image.save(path)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base = EIParams()
    hopf = hopf_tau_i(base)
    k = base.m_ei * base.m_ie - (base.m_ee - 1.0) * (base.m_ii + 1.0)
    h_active = max(
        base.m_ei * base.h_i / (base.m_ii + 1.0),
        (base.m_ee - 1.0) * base.h_i / base.m_ie,
    )

    tau_rows = scan_tau_i(base, output_dir, args.dt, args.t_end, (0.60, 0.10))
    h_rows = scan_h_e(base, output_dir, args.dt, args.t_end, (0.60, 0.10))

    save_tau_scan_plot(tau_rows, hopf, output_dir / "tau_i_scan.png")
    save_h_scan_plot(h_rows, h_active, output_dir / "h_e_scan.png")

    summary = {
        "base_parameters": asdict(base),
        "active_region_K": k,
        "tau_i_hopf": hopf,
        "omega_hopf": float(np.sqrt(k / (base.tau_e * hopf))),
        "h_e_active_boundary": h_active,
        "stable_example_tau_i": 1.5,
        "oscillatory_example_tau_i": 2.8,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Base parameters:")
    for key, value in asdict(base).items():
        print(f"  {key}: {value}")
    print(f"Active determinant numerator K = {k:.6f}")
    print(f"Hopf tau_I = {hopf:.6f}, omega = {summary['omega_hopf']:.6f}")
    print(f"h_E active-active boundary = {h_active:.6f}")
    print(f"Artifacts written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E-I network Hopf verification.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=240.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
