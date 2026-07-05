"""Question 1: LIF renewal moments and parameter construction.

The script constructs parameters (g_syn, mu, sigma) for a target inter-spike
interval mean/variance, then validates them by simulating the original LIF
model driven by an Ornstein-Uhlenbeck synaptic current.

Run:
    python lif_renewal_sim.py
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
class LIFConstants:
    g_l: float = 0.2
    tau_syn: float = 0.02
    v_reset: float = 0.0
    v_th: float = 1.0
    tau_ref: float = 0.05
    g_syn: float = 1.0


@dataclass
class ConstructedParameters:
    target_mean: float
    target_var: float
    a_eff: float
    b_eff: float
    g_syn: float
    mu: float
    sigma: float
    theory_fpt_mean: float
    theory_isi_mean: float
    theory_isi_var: float


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal linear system."""
    n = len(diag)
    c = upper.astype(float).copy()
    d = diag.astype(float).copy()
    b = rhs.astype(float).copy()
    a = lower.astype(float).copy()

    for i in range(1, n):
        factor = a[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        b[i] -= factor * b[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


def fpt_moments_ou(
    a_eff: float,
    b_eff: float,
    constants: LIFConstants,
    grid_size: int = 900,
) -> tuple[float, float]:
    """First two moments of hitting V_th for dV=(-gL*V+a)dt+b dW.

    The backward equations are solved on [x_min, V_th] with a reflecting lower
    boundary far below reset and an absorbing boundary at threshold.
    """
    if a_eff <= constants.g_l * constants.v_th:
        return float("inf"), float("inf")
    if b_eff <= 1e-10:
        mean = deterministic_hitting_time(a_eff, constants)
        return mean, mean * mean

    std_stationary = b_eff / np.sqrt(2.0 * constants.g_l)
    x_min = min(
        constants.v_reset - 7.0 * std_stationary - 1.0,
        -2.0,
    )
    x_max = constants.v_th
    x = np.linspace(x_min, x_max, grid_size)
    h = x[1] - x[0]
    d = 0.5 * b_eff * b_eff
    drift = -constants.g_l * x + a_eff

    diag = np.zeros(grid_size)
    lower = np.zeros(grid_size - 1)
    upper = np.zeros(grid_size - 1)

    # Reflecting lower boundary: u(x_0) - u(x_1) = 0.
    diag[0] = 1.0
    upper[0] = -1.0

    # Absorbing upper boundary: u(V_th)=0.
    diag[-1] = 1.0

    for i in range(1, grid_size - 1):
        lower[i - 1] = d / h**2 - drift[i] / (2.0 * h)
        diag[i] = -2.0 * d / h**2
        upper[i] = d / h**2 + drift[i] / (2.0 * h)

    rhs1 = -np.ones(grid_size)
    rhs1[0] = 0.0
    rhs1[-1] = 0.0
    u1 = solve_tridiagonal(lower, diag, upper, rhs1)

    rhs2 = -2.0 * u1
    rhs2[0] = 0.0
    rhs2[-1] = 0.0
    u2 = solve_tridiagonal(lower, diag, upper, rhs2)

    mean = float(np.interp(constants.v_reset, x, u1))
    second = float(np.interp(constants.v_reset, x, u2))
    return mean, second


def deterministic_hitting_time(a_eff: float, constants: LIFConstants) -> float:
    g_l = constants.g_l
    vr = constants.v_reset
    th = constants.v_th
    if g_l == 0.0:
        return (th - vr) / a_eff
    ratio = (a_eff - g_l * th) / (a_eff - g_l * vr)
    return -np.log(ratio) / g_l


def deterministic_a_for_mean(mean_fpt: float, constants: LIFConstants) -> float:
    g_l = constants.g_l
    vr = constants.v_reset
    th = constants.v_th
    if g_l == 0.0:
        return (th - vr) / mean_fpt
    e = np.exp(-g_l * mean_fpt)
    return g_l * (th - vr * e) / (1.0 - e)


def mean_for_a(a_eff: float, b_eff: float, constants: LIFConstants) -> float:
    return fpt_moments_ou(a_eff, b_eff, constants)[0]


def find_a_for_mean(b_eff: float, mean_fpt: float, constants: LIFConstants) -> float:
    lo = constants.g_l * constants.v_th + 1e-5
    hi = max(deterministic_a_for_mean(mean_fpt, constants) + 5.0 * b_eff, lo + 0.5)

    while mean_for_a(hi, b_eff, constants) > mean_fpt:
        hi *= 1.8

    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if mean_for_a(mid, b_eff, constants) > mean_fpt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def moments_for_b(b_eff: float, mean_fpt: float, constants: LIFConstants) -> tuple[float, float, float]:
    a_eff = find_a_for_mean(b_eff, mean_fpt, constants)
    m1, m2 = fpt_moments_ou(a_eff, b_eff, constants)
    return a_eff, m1, max(0.0, m2 - m1 * m1)


def construct_parameters(target_mean: float, target_var: float, constants: LIFConstants) -> ConstructedParameters:
    if target_mean <= constants.tau_ref:
        raise ValueError("target_mean must be larger than tau_ref.")
    if target_var <= 0.0:
        raise ValueError("target_var must be positive.")

    mean_fpt = target_mean - constants.tau_ref
    lo = 1e-4
    hi = max(0.05, np.sqrt(target_var))
    _, _, var_hi = moments_for_b(hi, mean_fpt, constants)
    while var_hi < target_var:
        hi *= 1.8
        _, _, var_hi = moments_for_b(hi, mean_fpt, constants)
        if hi > 10.0:
            raise RuntimeError("Could not bracket target variance.")

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        _, _, var_mid = moments_for_b(mid, mean_fpt, constants)
        if var_mid < target_var:
            lo = mid
        else:
            hi = mid

    b_eff = 0.5 * (lo + hi)
    a_eff, fpt_mean, fpt_var = moments_for_b(b_eff, mean_fpt, constants)
    g_syn = constants.g_syn
    return ConstructedParameters(
        target_mean=target_mean,
        target_var=target_var,
        a_eff=a_eff,
        b_eff=b_eff,
        g_syn=g_syn,
        mu=a_eff / g_syn,
        sigma=b_eff / g_syn,
        theory_fpt_mean=fpt_mean,
        theory_isi_mean=constants.tau_ref + fpt_mean,
        theory_isi_var=fpt_var,
    )


def simulate_isi(
    params: ConstructedParameters,
    constants: LIFConstants,
    n_intervals: int,
    dt: float,
    seed: int,
    max_time: float = 10.0,
) -> np.ndarray:
    """Vectorized Euler-Maruyama simulation of independent ISIs."""
    rng = np.random.default_rng(seed)
    mu = params.mu
    sigma = params.sigma
    g_syn = params.g_syn

    v = np.full(n_intervals, constants.v_reset, dtype=float)
    i_syn = rng.normal(mu, sigma / np.sqrt(2.0 * constants.tau_syn), size=n_intervals)
    active = np.ones(n_intervals, dtype=bool)
    t = np.zeros(n_intervals, dtype=float)
    isi = np.full(n_intervals, np.nan, dtype=float)
    sqrt_dt = np.sqrt(dt)
    max_steps = int(max_time / dt)

    for _ in range(max_steps):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        v_old = v[idx].copy()

        noise = rng.normal(size=len(idx))
        i_syn[idx] += ((mu - i_syn[idx]) / constants.tau_syn) * dt
        i_syn[idx] += (sigma / constants.tau_syn) * sqrt_dt * noise
        v[idx] += (-constants.g_l * v[idx] + g_syn * i_syn[idx]) * dt
        t[idx] += dt

        hit = v[idx] >= constants.v_th
        if np.any(hit):
            hit_idx = idx[hit]
            v_new = v[hit_idx]
            prev = v_old[hit]
            frac = np.clip((constants.v_th - prev) / (v_new - prev + 1e-12), 0.0, 1.0)
            crossing_time = t[hit_idx] - dt + frac * dt
            isi[hit_idx] = constants.tau_ref + crossing_time
            active[hit_idx] = False

    if np.any(active):
        raise RuntimeError(
            f"{int(active.sum())} simulated intervals did not hit threshold; increase max_time."
        )
    return isi


def renewal_count_statistics(
    isi_samples: np.ndarray,
    target_mean: float,
    target_var: float,
    out_csv: Path,
    t_max: float = 20.0,
    n_paths: int = 2500,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, t_max, 41)
    counts = np.zeros((n_paths, len(grid)), dtype=int)

    for path in range(n_paths):
        total = 0.0
        spike_times: list[float] = []
        while total <= t_max:
            total += float(rng.choice(isi_samples))
            if total <= t_max:
                spike_times.append(total)
        spike_times_np = np.asarray(spike_times)
        counts[path] = np.searchsorted(spike_times_np, grid, side="right")

    mean_count = counts.mean(axis=0)
    var_count = counts.var(axis=0, ddof=1)
    asym_mean = grid / target_mean
    asym_var = target_var * grid / (target_mean**3)

    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "empirical_mean", "empirical_var", "asymptotic_mean", "asymptotic_var"])
        for row in zip(grid, mean_count, var_count, asym_mean, asym_var):
            writer.writerow([f"{x:.8f}" for x in row])

    return grid, mean_count, var_count, asym_mean, asym_var


def save_isi_histogram(
    isi: np.ndarray,
    target_mean: float,
    simulated_mean: float,
    path: Path,
) -> None:
    if Image is None:
        return
    width, height = 860, 520
    ml, mr, mt, mb = 70, 35, 42, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pw = width - ml - mr
    ph = height - mt - mb

    bins = np.linspace(max(0.0, float(np.min(isi)) - 0.05), float(np.percentile(isi, 99.5)), 36)
    hist, edges = np.histogram(isi, bins=bins)
    max_hist = max(1, int(hist.max()))

    draw.rectangle((ml, mt, width - mr, height - mb), outline=(30, 30, 30))
    draw.text((width // 2 - 145, 14), "Inter-spike interval histogram", fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 18, height - 28), "T", fill=(0, 0, 0), font=font)
    draw.text((12, mt + ph // 2), "count", fill=(0, 0, 0), font=font)

    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        x0 = ml + (left - edges[0]) / (edges[-1] - edges[0]) * pw
        x1 = ml + (right - edges[0]) / (edges[-1] - edges[0]) * pw
        y0 = height - mb - count / max_hist * ph
        draw.rectangle((x0, y0, x1 - 1, height - mb), fill=(90, 150, 220), outline="white")

    def draw_vline(value: float, color: tuple[int, int, int], label: str, y_offset: int) -> None:
        x = ml + (value - edges[0]) / (edges[-1] - edges[0]) * pw
        draw.line((x, mt, x, height - mb), fill=color, width=3)
        draw.text((x + 4, mt + y_offset), label, fill=color, font=font)

    draw_vline(target_mean, (220, 70, 60), f"target mean={target_mean:.3f}", 12)
    draw_vline(simulated_mean, (30, 120, 60), f"sim mean={simulated_mean:.3f}", 30)
    image.save(path)


def save_renewal_plot(
    grid: np.ndarray,
    mean_count: np.ndarray,
    var_count: np.ndarray,
    asym_mean: np.ndarray,
    asym_var: np.ndarray,
    path: Path,
) -> None:
    if Image is None:
        return
    width, height = 900, 540
    ml, mr, mt, mb = 70, 45, 42, 62
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pw = width - ml - mr
    ph = height - mt - mb
    ymax = max(float(mean_count.max()), float(var_count.max()), float(asym_mean.max()), float(asym_var.max())) * 1.1

    draw.rectangle((ml, mt, width - mr, height - mb), outline=(30, 30, 30))
    draw.text((width // 2 - 125, 14), "Renewal count statistics", fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 18, height - 30), "time", fill=(0, 0, 0), font=font)
    draw.text((12, mt + ph // 2), "value", fill=(0, 0, 0), font=font)

    def to_points(values: np.ndarray) -> list[tuple[float, float]]:
        pts = []
        for t, value in zip(grid, values):
            x = ml + (t - grid[0]) / (grid[-1] - grid[0]) * pw
            y = height - mb - value / ymax * ph
            pts.append((x, y))
        return pts

    series = [
        ("E[N(t)] empirical", mean_count, (220, 70, 60), 0),
        ("E[N(t)] asymptotic", asym_mean, (220, 70, 60), 1),
        ("Var[N(t)] empirical", var_count, (40, 120, 220), 0),
        ("Var[N(t)] asymptotic", asym_var, (40, 120, 220), 1),
    ]
    for _, values, color, dashed in series:
        pts = to_points(values)
        if dashed:
            for p0, p1 in zip(pts[:-1:2], pts[1::2]):
                draw.line((p0, p1), fill=color, width=2)
        else:
            draw.line(pts, fill=color, width=3)

    lx, ly = width - mr - 205, mt + 12
    for k, (label, _, color, dashed) in enumerate(series):
        y = ly + 20 * k
        if dashed:
            draw.line((lx, y + 7, lx + 12, y + 7), fill=color, width=2)
            draw.line((lx + 18, y + 7, lx + 30, y + 7), fill=color, width=2)
        else:
            draw.line((lx, y + 7, lx + 30, y + 7), fill=color, width=3)
        draw.text((lx + 38, y), label, fill=(0, 0, 0), font=font)

    image.save(path)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    constants = LIFConstants(
        g_l=args.g_l,
        tau_syn=args.tau_syn,
        v_reset=args.v_reset,
        v_th=args.v_th,
        tau_ref=args.tau_ref,
        g_syn=args.g_syn,
    )
    params = construct_parameters(args.target_mean, args.target_var, constants)

    isi = simulate_isi(
        params,
        constants,
        n_intervals=args.n_intervals,
        dt=args.dt,
        seed=args.seed,
        max_time=args.max_time,
    )
    sim_mean = float(np.mean(isi))
    sim_var = float(np.var(isi, ddof=1))
    sim_cv = float(np.sqrt(sim_var) / sim_mean)

    np.savetxt(output_dir / "isi_samples.csv", isi, delimiter=",", header="isi", comments="")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        summary = {
            "constants": asdict(constants),
            "constructed_parameters": asdict(params),
            "simulation": {
                "n_intervals": args.n_intervals,
                "dt": args.dt,
                "mean": sim_mean,
                "variance": sim_var,
                "cv": sim_cv,
                "relative_mean_error": (sim_mean - args.target_mean) / args.target_mean,
                "relative_var_error": (sim_var - args.target_var) / args.target_var,
            },
        }
        json.dump(summary, file, indent=2, ensure_ascii=False)

    grid, mean_count, var_count, asym_mean, asym_var = renewal_count_statistics(
        isi,
        target_mean=sim_mean,
        target_var=sim_var,
        out_csv=output_dir / "renewal_counts.csv",
        t_max=args.renewal_t_max,
        n_paths=args.renewal_paths,
        seed=args.seed + 100,
    )

    save_isi_histogram(isi, args.target_mean, sim_mean, output_dir / "isi_histogram.png")
    save_renewal_plot(grid, mean_count, var_count, asym_mean, asym_var, output_dir / "renewal_counts.png")

    print("Constructed parameters")
    print(f"  g_syn = {params.g_syn:.6f}")
    print(f"  mu    = {params.mu:.6f}")
    print(f"  sigma = {params.sigma:.6f}")
    print("Moment comparison")
    print(f"  target mean / var      = {args.target_mean:.6f} / {args.target_var:.6f}")
    print(f"  diffusion mean / var   = {params.theory_isi_mean:.6f} / {params.theory_isi_var:.6f}")
    print(f"  simulated mean / var   = {sim_mean:.6f} / {sim_var:.6f}")
    print(f"  simulated CV           = {sim_cv:.6f}")
    print(f"Artifacts written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIF renewal simulation for question 1.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-mean", type=float, default=1.0)
    parser.add_argument("--target-var", type=float, default=0.08)
    parser.add_argument("--g-l", type=float, default=0.2)
    parser.add_argument("--tau-syn", type=float, default=0.02)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--v-th", type=float, default=1.0)
    parser.add_argument("--tau-ref", type=float, default=0.05)
    parser.add_argument("--g-syn", type=float, default=1.0)
    parser.add_argument("--n-intervals", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--max-time", type=float, default=8.0)
    parser.add_argument("--renewal-t-max", type=float, default=20.0)
    parser.add_argument("--renewal-paths", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
