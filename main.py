#!/usr/bin/env python3
"""Reference model for roblox-up-and-coming-forecast-01 (PROTOTYPE for tuning only;
the real entry is main.py, written by the parent agent — mirror this logic).

Idea: per-game rank distribution diffused over the 24 hourly horizons of the
scored window, then averaged (one forecast serves all 24 slots, so the honest
forecast is the horizon-average). Components, all calibrated on the archive:
  * recency-weighted rank velocity (momentum, shrunk) with saturating drift;
  * Gaussian rank diffusion, sigma growing ~ sqrt(horizon);
  * top-10 exit / 11-30 entry mass transfer, rank-dependent;
  * #1 reign-age exit risk (a long #1 reign is fragile: DONT SPLASH vanished
    after ~6.5 days at #1) with the heir (#2) absorbing most of the mass;
  * per-place flattening (small for places 1-4, larger for 5-10) and an
    unnamed share left at every place to cover off-chart entrants.

Inputs: MAKEFUN_INPUT_DIR (history.ndjson, task.json) — window read from
task.json, never hardcoded. Output: MAKEFUN_OUTPUT_DIR/forecast.json.
Stdlib only, deterministic.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- parameters
PARAMS = {
    "VEL_WINDOW_H": 12.0,    # lookback (h) for velocity estimation (v2: short window)
    "VEL_HALF_LIFE_H": 6.0,  # exponential recency half-life for velocity fit
    "SHRINK": 0.3,           # momentum shrink (v2: short-window momentum helps)
    "DRIFT_TAU_H": 12.0,     # momentum saturation timescale (h)
    "DRIFT_MAX": 3.0,         # max total drift (ranks) over the window
    "DRIFT_DECAY_H": 0.0,     # drift decay timescale (h); 0 = no decay
    "SIGMA_H": 0.90,          # diffusion: ranks per sqrt(hour)
    "SIGMA_CAP": 5.0,         # max sigma (ranks)
    "SIGMA_TOP_MULT": 0.3,    # sigma multiplier for ranks 1-2 (sticky top)
    "SIGMA_MID_MULT": 0.8,    # sigma multiplier for ranks 3-4
    "SIGMA_MIN_TOP": 0.3,     # sigma floor for ranks 1-2
    "SIGMA_MIN": 0.6,         # sigma floor otherwise
    "EXIT_BASE_24H": 0.05,    # 24h P(leave top 10) at rank 1
    "EXIT_SLOPE_24H": 0.025,  # + per rank (rank 10 -> 0.275)
    "RANK1_EXIT_24H": 0.35,   # 24h P(#1 loses #1); reign-age adjusted in code
    "ENTRY_TOTAL_11_20": 3.0,  # expected top-10 entrants/24h from ranks 11-20
    "ENTRY_TOTAL_21_30": 0.12,  # expected top-10 entrants/24h from ranks 21-30
    "EXIT_TABLE": {1: 0.021, 2: 0.021, 3: 0.021, 4: 0.021, 5: 0.028, 6: 0.147, 7: 0.21, 8: 0.413, 9: 0.574, 10: 0.672},  # measured per-rank 24h exit, softened x0.7
    "RANK1_EXIT_CAP": 0.8,  # cap on reign-adjusted #1 exit/24h
    "ALPHA_TOP": 0.0,        # flat-mix weight, places 1-4
    "ALPHA_LOW": 0.25,        # flat-mix weight, places 5-10
    "UNNAMED": 0.25,          # per-place share left unnamed
    "CANDIDATE_TOP_N": 40,    # games considered, by adjusted last rank
    "MAXNAMED": 30,           # max named games per place in the output
    "MIN_P": 0.001,           # drop output rows below this p
    "RANK_GRID": 45,          # ranks modelled explicitly (1..45); rest = out
    # --- v2 additions (defaults = v1 behaviour unless noted) ---
    "PLAY_WINDOW_H": 24.0,    # lookback (h) for playing-count growth
    "PLAY_DRIFT": 0.0,        # ranks of drift per unit log-playing growth (v2: NOT used)
    "PLAY_DRIFT_MAX": 3.0,    # cap on |playing drift| (ranks)
    "AGE_YOUNG": 1.0,         # exit multiplier, age < 14d (v2: NOT used)
    "AGE_MID": 1.0,           # exit multiplier, age 21-25d (v2: NOT used)
    "AGE_OLD": 1.0,           # exit multiplier, age >= 25d (v2: NOT used)
    "LAND_SHIFT": 2.0,        # entrant landing mean = origin rank - LAND_SHIFT (v2: was 6.0)
    "LAND_SIGMA": 1.5,        # entrant landing spread (v2: was 2.0)
    "DETHRONE": None,         # {rank: share} for a dethroned #1 (+ "out"); None = v1 default
    "CREATED": None,          # {universe_id: ISO created} for the age hazard
}
RANK_GRID = PARAMS["RANK_GRID"]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    return reads, task


def collect_created(reads):
    """created dates from any read's details block (house recorder reads)."""
    created = {}
    for r in reads:
        det = r.get("details")
        if isinstance(det, dict):
            for k, v in det.items():
                if isinstance(v, dict) and v.get("created"):
                    try:
                        created[int(k)] = v["created"]
                    except (TypeError, ValueError):
                        pass
    return created


def playing_growth(pts, t0, window_h):
    """Log playing growth over the last window_h hours.
    pts: [(t, rank, playing)] sorted. Returns None if unusable."""
    recent = [(t, p) for t, _, p in pts
              if 0 <= (t0 - t).total_seconds() / 3600 <= window_h and p]
    if len(recent) < 2:
        return None
    p_now = recent[-1][1]
    p_old = recent[0][1]
    if not p_now or not p_old or p_now <= 0 or p_old <= 0:
        return None
    return math.log(p_now + 1) - math.log(p_old + 1)


def norm_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def velocity(times, ranks, t0, window_h, half_life_h):
    """Recency-weighted linear fit of rank vs time (hours before t0).
    Returns ranks/hour (negative = improving). None if <2 points in window."""
    pts = [(t, r) for t, r in zip(times, ranks) if 0 <= (t0 - t).total_seconds() / 3600 <= window_h]
    if len(pts) < 2:
        return None
    w = [0.5 ** (((t0 - t).total_seconds() / 3600) / half_life_h) for t, _ in pts]
    x = [-(t0 - t).total_seconds() / 3600 for t, _ in pts]  # hours ago (<=0)
    y = [r for _, r in pts]
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    den = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    if den <= 0:
        return None
    return sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y)) / den


def build_forecast(reads, task, P=None):
    P = dict(PARAMS) if P is None else {**PARAMS, **P}
    t_end = parse_time(task["window"]["from"])
    last = reads[-1]
    last_chart = sorted(last["up_and_coming"], key=lambda g: g["rank"])
    last_rank = {g["universe_id"]: g["rank"] for g in last_chart}
    last_time = parse_time(last["at"])

    # per-game rank history (time, rank, playing)
    hist = {}
    for r in reads:
        t = parse_time(r["at"])
        for g in r["up_and_coming"]:
            hist.setdefault(g["universe_id"], []).append((t, g["rank"], g.get("playing")))
    for u in hist:
        hist[u].sort()

    # created dates for the age hazard: from params (backtest) or details
    created = P.get("CREATED") or collect_created(reads)
    created_dt = {}
    for u, v in (created or {}).items():
        try:
            created_dt[int(u)] = parse_time(v) if isinstance(v, str) else v
        except (TypeError, ValueError):
            pass

    def age_mult(u):
        c = created_dt.get(int(u))
        if c is None:
            return 1.0
        age_d = (t_end - c).total_seconds() / 86400.0
        if age_d < 14:
            return P["AGE_YOUNG"]
        if age_d < 21:
            return 1.0
        if age_d < 25:
            return P["AGE_MID"]
        return P["AGE_OLD"]

    # reign age of the #1 game (hours it has held #1 consecutively)
    top1 = last_chart[0]["universe_id"]
    reign_h = 0.0
    for r in reversed(reads):
        if sorted(r["up_and_coming"], key=lambda g: g["rank"])[0]["universe_id"] == top1:
            reign_h = (t_end - parse_time(r["at"])).total_seconds() / 3600
        else:
            break
    # #1 24h exit prob grows with reign age: 0.07 fresh -> ~0.20 at 7 days
    rank1_exit_24h = min(P["RANK1_EXIT_CAP"], P["RANK1_EXIT_24H"] + 0.018 * max(0.0, reign_h - 24) / 24)

    # candidate set: last rank <= 40, or top-20 in last 48h, ranked by adjusted rank
    cand = set(u for u, r in last_rank.items() if r <= P["CANDIDATE_TOP_N"])
    for u, pts in hist.items():
        if any(r <= 20 and (t_end - t).total_seconds() / 3600 <= 48 for t, r, _ in pts):
            cand.add(u)
    # velocity + playing drift + adjusted rank for ordering
    vel = {}
    play_drift = {}
    adj = {}
    for u in cand:
        pts = hist.get(u, [])
        times = [t for t, _, _ in pts]
        ranks = [r for _, r, _ in pts]
        v = velocity(times, ranks, t_end, P["VEL_WINDOW_H"], P["VEL_HALF_LIFE_H"])
        vel[u] = v if v is not None else 0.0
        g = playing_growth(pts, t_end, P["PLAY_WINDOW_H"])
        play_drift[u] = (max(-P["PLAY_DRIFT_MAX"], min(P["PLAY_DRIFT_MAX"], -P["PLAY_DRIFT"] * g))
                         if g is not None else 0.0)
        r0 = last_rank.get(u, 40)
        adj[u] = r0 + vel[u] * P["SHRINK"] * P["DRIFT_TAU_H"] + play_drift[u]
    cand = sorted(cand, key=lambda u: adj[u])[: P["CANDIDATE_TOP_N"]]

    # per-game horizon-averaged rank distribution over 1..RANK_GRID (+ out)
    avg = {}
    horizons = list(range(1, task["window"]["slots"] + 1))
    for u in cand:
        r0 = last_rank.get(u, 40)
        v = vel[u]
        drift_total = max(-P["DRIFT_MAX"], min(P["DRIFT_MAX"], v * P["SHRINK"] * P["DRIFT_TAU_H"])) + play_drift[u]
        drift_total = max(-P["DRIFT_MAX"] - P["PLAY_DRIFT_MAX"],
                          min(P["DRIFT_MAX"] + P["PLAY_DRIFT_MAX"], drift_total))
        acc = [0.0] * (RANK_GRID + 1)  # index 0 unused; RANK_GRID = out bucket
        out_mass = 0.0
        for h in horizons:
            # saturating drift + diffusion; drift decays at long horizons
            # (momentum is a short-horizon signal) when DRIFT_DECAY_H > 0
            frac = 1 - math.exp(-h / P["DRIFT_TAU_H"])
            if P.get("DRIFT_DECAY_H"):
                frac *= math.exp(-h / P["DRIFT_DECAY_H"])
            mu = r0 + drift_total * frac
            mu = max(1.0, min(float(RANK_GRID), mu))
            mult = (P["SIGMA_TOP_MULT"] if r0 <= 2 else
                    P["SIGMA_MID_MULT"] if r0 <= 4 else 1.0)
            smin = P["SIGMA_MIN_TOP"] if r0 <= 2 else P["SIGMA_MIN"]
            sigma = min(P["SIGMA_H"] * math.sqrt(h) * mult, P["SIGMA_CAP"])
            sigma = max(sigma, smin)
            dist = [0.0] * (RANK_GRID + 1)
            tot = 0.0
            for k in range(1, RANK_GRID + 1):
                # integrate the normal over [k-0.5, k+0.5] approx by pdf at k
                dist[k] = norm_pdf(float(k), mu, sigma)
                tot += dist[k]
            # mass beyond the grid -> out (rank > 45)
            p_out = max(0.0, 1.0 - tot)
            # exit/entry transfer for this horizon (scale 24h rates to h hours)
            f = h / 24.0
            if r0 <= 10:
                if P.get("EXIT_TABLE"):
                    p_exit = (rank1_exit_24h if r0 == 1 else
                              P["EXIT_TABLE"].get(r0, P["EXIT_BASE_24H"] + P["EXIT_SLOPE_24H"] * (r0 - 1)))
                else:
                    p_exit = (rank1_exit_24h if r0 == 1 else
                              min(0.6, P["EXIT_BASE_24H"] + P["EXIT_SLOPE_24H"] * (r0 - 1)))
                if r0 != 1:
                    p_exit = min(0.98, p_exit * age_mult(u))
                p_exit = min(0.98, p_exit)
                e = 1 - (1 - p_exit) ** f
                moved = sum(dist[1:11]) * e
                for k in range(1, 11):
                    dist[k] *= (1 - e)
                if r0 == 1:
                    # a dethroned #1 usually falls to #2 (measured 7/8) or #3
                    deth = P.get("DETHRONE") or {2: 0.28, 3: 0.18, 4: 0.14, 5: 0.10}
                    out_share = 1.0 - sum(deth.values())
                    for k, w in deth.items():
                        dist[k] += moved * w
                    p_out += moved * out_share
                else:
                    # spread over ranks 11-30, decaying weights summing to 1
                    ws = {k: 0.5 ** ((k - 11) / 4.0) for k in range(11, 31)}
                    sw = sum(ws.values())
                    for k, w in ws.items():
                        dist[k] += moved * w / sw
            else:
                # Entry: per-game 24h entry prob decays with origin rank
                # (rank 11 ~2x rank 20); total expected entrants from 11-20
                # ~= 1.44/24h, from 21-30 ~= 0.12/24h (archive 24h pairs).
                if r0 <= 20:
                    w = 1.0 / (r0 - 9)
                    norm = sum(1.0 / (r - 9) for r in range(11, 21))
                    p_entry = min(0.9, P["ENTRY_TOTAL_11_20"] * w / norm)
                else:
                    w = 1.0 / (r0 - 9)
                    norm = sum(1.0 / (r - 9) for r in range(21, 31))
                    p_entry = min(0.9, P["ENTRY_TOTAL_21_30"] * w / norm)
                e = 1 - (1 - p_entry) ** f
                moved = sum(dist[11:]) * e + p_out * e
                for k in range(11, RANK_GRID + 1):
                    dist[k] *= (1 - e)
                p_out *= (1 - e)
                # Entrants land near origin rank - LAND_SHIFT (measured: 11-13 -> ~9)
                mu_land = max(2.0, r0 - P["LAND_SHIFT"])
                lw = {}
                ltot = 0.0
                for k in range(1, 11):
                    lw[k] = norm_pdf(float(k), mu_land, P["LAND_SIGMA"])
                    ltot += lw[k]
                for k in range(1, 11):
                    dist[k] += moved * lw[k] / ltot
            # renormalise (pdf approximation + transfers)
            s = sum(dist[1:]) + p_out
            if s > 0:
                for k in range(1, RANK_GRID + 1):
                    acc[k] += dist[k] / s
                out_mass += p_out / s
        n_h = len(horizons)
        avg[u] = ([a / n_h for a in acc], out_mass / n_h)

    # per-place probabilities from averaged distributions
    places = task["places"]
    probs = {k: {} for k in range(1, places + 1)}
    for u in cand:
        dist, _ = avg[u]
        for k in range(1, places + 1):
            probs[k][u] = dist[k]
    # heir adjustment for place 1: if the #1 exits, #2/#3 absorb most of it
    # (already partly in the diffusion; keep as is -- no extra step)
    # flatten + unnamed share per place
    out_rows = []
    for k in range(1, places + 1):
        alpha = P["ALPHA_TOP"] if k <= 4 else P["ALPHA_LOW"]
        cands = sorted(probs[k], key=lambda u: -probs[k][u])[: P["MAXNAMED"]]
        named_sum = sum(probs[k][u] for u in cands)
        # scale named mass to (1 - UNNAMED), then flat-mix toward uniform
        target = 1.0 - P["UNNAMED"]
        uni = target / len(cands) if cands else 0.0
        kept = []
        for u in cands:
            base = probs[k][u] / named_sum * target if named_sum > 0 else uni
            p = (1 - alpha) * base + alpha * uni
            if p >= P["MIN_P"]:
                kept.append([int(u), p])
        # Renormalise after the MIN_P cut so the named rows at this place
        # sum to exactly 1 - UNNAMED (the cut otherwise leaves places 1-4
        # a touch below target — measured 0.7477-0.7489 instead of 0.75 —
        # inflating the unnamed share).
        kept_sum = sum(p for _, p in kept)
        if kept_sum > 0:
            scale = target / kept_sum
            for row in kept:
                row[1] = round(row[1] * scale, 6)
        for u, p in kept:
            out_rows.append({"universe_id": u, "place": k, "p": p})
    # top10 by expected rank (lowest mean rank over averaged distribution)
    def mean_rank(u):
        dist, out = avg[u]
        m = sum(k * dist[k] for k in range(1, RANK_GRID + 1)) + out * 50.0
        return m
    top10 = [int(u) for u in sorted(cand, key=mean_rank)[:places]]
    return {"probabilities": out_rows, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    params = {}
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
    reads, task = load_history(input_dir)
    fc = build_forecast(reads, task, params)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(fc) + "\n")
    print(f"wrote {len(fc['probabilities'])} rows, top10={fc['top10']}")


if __name__ == "__main__":
    main()
