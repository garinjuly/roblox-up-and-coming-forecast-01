# Roblox Up-and-Coming Forecast — v2

Forecast Roblox's **Up-and-Coming** chart top 10 — a probability for each game
at each place — for every hour of the 72 hours after the make.fun deadline.
Entry for the make.fun competition
[`roblox-up-and-coming-forecast-01`](https://make.fun/c/roblox-up-and-coming-forecast-01)
(Predict Roblox's Up-and-Coming top 10; $500 USDC in prizes).

This is v2 of the entry (v1: submission `01a1cd3f-eabe-7150-b011-5038b25b85ff`,
practice −35.93). Two measured improvements over v1; everything else is the
same rank-diffusion core.

## Method

Rank-diffusion forecast, Python 3.13 standard library only, deterministic,
no network. `main.py` reads `$MAKEFUN_INPUT_DIR/history.ndjson` + `task.json`
(the scored window is read from `task.json`, never hardcoded) and writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` (one forecast serving all 72 hourly slots:
the average of the per-hour rank distributions).

- Per-game Gaussian rank distribution diffused over horizons h = 1..72h
  (sigma growing ~ sqrt(h); ranks 1–2 diffuse much slower — the top is sticky).
- Top-10 exit / rank 11–30 entry mass transfer, rank-dependent, scaled from
  measured 24h rates. Measured on the archive: ranks 1–5 almost never leave
  the top 10, rank 6 ~0.21/24h, rank 7 ~0.30, rank 8 ~0.59, rank 9 ~0.82,
  rank 10 ~0.96 (softened ×0.7 — the raw 6.4h-gap extrapolation overstates
  exits). Entries run ~5/day into the top 10, almost all from ranks 11–20.
- #1 reign-age exit risk: a long reign at #1 is fragile; a dethroned #1
  usually falls to ranks 2–5 (70%) or leaves the chart (30%).
- **v2 — entry landing calibrated:** entrants from ranks 11–13 land at place
  ~9 on average (measured on 43 entrants), not ~5 as v1 assumed. Landing
  mean is now origin rank − 2 (was −6), spread 1.5 (was 2.0). Biggest single
  win: +0.44 nats backtest, +1.30 practice.
- **v2 — short-window momentum:** rank velocity over the last 12h (6h
  half-life), shrink 0.3. Long-window momentum (48h) does not help;
  anti-momentum hurts. Momentum is a short-horizon signal.
- Tested and NOT used: playing-count growth drift (wins the backtest but
  loses the practice window — it extrapolates too far at 48–72h horizons);
  age/tenure exit multipliers (measured exit rates do rise with age, but
  multiplying the exit table by age hurts the backtest); dethrone
  concentrated at #2 (measured 7/8 landings) — hurts the backtest at long
  horizons.
- Playing counts, votes, visits and favorites are NOT used as features.
- Per place, named probabilities sum to 0.75; 0.25 is left unnamed to cover
  off-chart entrants, and places 5–10 are flattened toward uniform — this
  chart churns hard (a median 6 of 10 places change between consecutive
  reads).

## Validation

- Backtest, 24 cutoffs inside the 34-read kit history (72h ahead each):
  **mean log score −37.56** vs v1's −38.28 (+0.73; wins 22/24 cutoffs, wins
  both split-halves, wins every horizon bucket 0–24h / 24–48h / 48–72h).
  House starter baseline on the same cutoffs: −46.54.
- Practice window (10 public-archive reads, 8–10 Nov 2026):
  **−34.93** vs v1's −35.93 (+1.00), the starter's −43.51 and the best
  house entry's −37.2 (+2.27).
- Local run: ~0.4 s, deterministic (two runs byte-identical), 241 rows,
  14 KB.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

## License

MIT — see `LICENSE`.
