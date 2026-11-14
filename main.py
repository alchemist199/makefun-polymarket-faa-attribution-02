# SPDX-License-Identifier: MIT
# polymarket-faa-attribution-02 entry (Python 3.13, standard library only).
#
# Forecast: for each market of the case, for each of the 240 minutes after the
# checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of the market's Yes price.
#
# Model (original work, calibrated on the practice cases' history, not on their
# answers): a jump-aware random walk around the last price.
#
#   q_tau(h) = clamp(x + S_tau * sigma * sqrt(h), 0, 1)
#
# where x is the market's price at the checkpoint and sigma is the market's own
# realised 1-minute volatility over the last 120 minutes (population standard
# deviation of the 1-minute price changes), floored at 0.0004 and capped at
# 0.01 price units per square-root minute, times 0.85.
#
# Why this shape (S = [-1.55, -0.30, 0, +0.15, +1.05]):
# Polymarket Yes prices are step functions: most minutes they do not move at
# all, then they jump. So the conditional distribution of the h-step move is
# far more peaked than a Gaussian (the 25%/75% quantiles sit only ~0.3 sigma
# from the median, not 0.67) with fat tails (the 5%/95% quantiles sit ~1.5x /
# ~1.05x sigma*sqrt(h) out). The downside tail is fatter than the upside one:
# a price can crash toward 0 in a minute, while rises are slower. The median
# is the last price: 1-minute returns show no exploitable drift (adding a
# recent-trend drift or a mean-reversion pull both made the practice loss
# worse), so there is no drift term.
#
# Why a 120-minute volatility window: volatility is strongly non-stationary
# (markets are calm for hours, then jump). A short window tracks the current
# regime; longer windows (6h/24h/72h) drag in stale jumps from the market's
# early chaotic hours and overstate the spread. The window is short, but the
# floor (0.0004) keeps a market that was flat for two hours from getting a
# zero-width forecast, and the cap (0.01) keeps one huge recent jump from
# blowing the spread up. The model needs only the last 120 minutes, so it
# still works when a case holds much less than 72 hours of history (the final
# FAA markets are only ~16 hours old at the first checkpoint).
#
# Deterministic: pure arithmetic over the case file, no randomness, no clock,
# no network. Two runs on the same case write byte-identical output.
#
# Usage (local): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# The grader runs: python3 /bot/main.py, reading /input/<id>.in, writing
# /output/<id>.out.

import json
import math
import os
from os import path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standardised h-step quantile shape S_tau (see header): how far each quantile
# sits from the median, in units of sigma * sqrt(h).
SHAPE = [-1.55, -0.30, 0.0, 0.15, 1.05]

# Volatility window: the last 120 one-minute price changes.
VOL_WINDOW = 120
# Floor / cap on the raw realised sigma, in price units per square-root
# minute (applied before the shrinkage multiplier below).
SIGMA_FLOOR = 0.0004
SIGMA_CAP = 0.01
# Shrinkage on the realised volatility (it is jump-dominated, so slightly
# overstated for the quiet minutes that dominate the score).
SIGMA_MULT = 0.85
# Fallback sigma when a market has too little history to estimate one.
SIGMA_DEFAULT = 0.0025


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def realised_sigma(prices):
    """Population std of the last VOL_WINDOW 1-minute price changes,
    floored/capped, then shrunk (see header)."""
    n = len(prices)
    if n < 6:
        return SIGMA_DEFAULT
    start = max(0, n - 1 - VOL_WINDOW)
    changes = [prices[i + 1] - prices[i] for i in range(start, n - 1)]
    if len(changes) < 5:
        return SIGMA_DEFAULT
    mean = math.fsum(changes) / len(changes)
    var = math.fsum((c - mean) * (c - mean) for c in changes) / len(changes)
    sigma = math.sqrt(var)
    if sigma < SIGMA_FLOOR:
        sigma = SIGMA_FLOOR
    elif sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    return sigma * SIGMA_MULT


def forecast_market(prices, horizon):
    """240 rows of five quantiles for one market, median = last price."""
    if not prices:
        # No history at all: stay flat at 0.5 rather than failing the case.
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = prices[-1]
    sigma = realised_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = [price6(x + SHAPE[j] * spread) for j in range(5)]
        # Rounding can only tie values, but enforce non-decreasing anyway.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for market in case["markets"]:
            rows = forecast_market(market["prices"], horizon)
            forecasts.append({"id": market["id"], "q": rows})
        out_path = path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
