"""Paired two-sided Student's t-tests between CNN-LSTM+MBFN and STL-Net (Table 11).

Reproduces the statistical analysis reported in Section 5.4 of the manuscript:
for each lead-time-metric combination, paired differences are computed across the
ten matched random seeds {0,1,2,3,4,5,6,7,42,2026} as

    d_i = E_i^{MBFN} - E_i^{STL-Net}

The mean paired difference, its 95% confidence interval

    d_bar +/- t_{0.975, 9} * s_d / sqrt(10)

and the exact two-sided p-value (df = 9) are reported.

Input: CSV with columns
    lead_time,metric,seed,mbfn_error,stlnet_error
where one row corresponds to one (lead time, metric, seed) combination.

Usage:
    python paired_ttest.py errors.csv
    python paired_ttest.py errors.csv --out table11.md
"""

import argparse
import math

import numpy as np

N_SEEDS = 10
DF = 9
T_CRIT_0975_9 = 2.2622  # t_{0.975, 9}; used for the CI when SciPy is unavailable


def t_cdf(t, df):
    """CDF of Student's t distribution via the regularized incomplete beta function."""
    from math import lgamma

    x = df / (df + t * t)
    # regularized incomplete beta I_x(df/2, 1/2) by continued fraction (Numerical Recipes)
    ib = _betacf(0.5 * df, 0.5, x)
    ib *= math.exp(0.5 * df * math.log(x) + 0.5 * math.log(1.0 - x)
                   - lgamma(0.5 * df) - lgamma(0.5) + lgamma(0.5 * df + 0.5))
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def _betacf(a, b, x, max_iter=200, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def paired_ttest(d):
    """Return (mean, sd, ci_half_width, two-sided p) for n=10 paired differences."""
    d = np.asarray(d, dtype=float)
    assert d.size == N_SEEDS, f"expected {N_SEEDS} paired differences, got {d.size}"
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(N_SEEDS)
    t_stat = mean / se
    try:
        from scipy import stats
        ci_half = float(stats.t.ppf(0.975, DF)) * se
        p = float(2.0 * stats.t.sf(abs(t_stat), DF))
    except ImportError:
        ci_half = T_CRIT_0975_9 * se
        p = 2.0 * (1.0 - t_cdf(abs(t_stat), DF))
    return mean, sd, ci_half, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="input CSV: lead_time,metric,seed,mbfn_error,stlnet_error")
    ap.add_argument("--out", default=None, help="output markdown file (default: stdout)")
    args = ap.parse_args()

    import csv as _csv
    groups = {}
    with open(args.csv) as f:
        for row in _csv.DictReader(f):
            key = (row["lead_time"], row["metric"])
            groups.setdefault(key, {})[int(row["seed"])] = (
                float(row["mbfn_error"]), float(row["stlnet_error"]))

    lines = [
        "| Lead time | Metric | Mean paired difference (m/s) | 95% CI (m/s) | Two-sided p-value |",
        "|---|---|---|---|---|",
    ]
    for (lead, metric) in sorted(groups, key=lambda k: (int(k[0][:-1]), k[1])):
        pairs = groups[(lead, metric)]
        seeds = sorted(pairs)
        assert len(seeds) == N_SEEDS, f"{lead} {metric}: expected {N_SEEDS} seeds"
        d = [pairs[s][0] - pairs[s][1] for s in seeds]
        mean, sd, ci_half, p = paired_ttest(d)
        lines.append(
            f"| {lead} | {metric} | {mean:+.4f} | [{mean - ci_half:.4f}, {mean + ci_half:.4f}] "
            f"| {p:.6f} |")

    note = ("\n\nNote: the paired difference is defined as CNN-LSTM+MBFN minus STL-Net; "
            "negative values indicate lower errors for CNN-LSTM+MBFN. Tests are two-sided with "
            "n = 10 matched random seeds and df = 9. The 95% CI is calculated as "
            "d_bar +/- t_{0.975,9} * s_d / sqrt(10).")
    out = "\n".join(lines) + note
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
