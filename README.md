# Ecosystem Compass

High-level comparison of technology equity **ecosystems** — 11 of them, from
Semiconductors to Energy & Power Infrastructure — ranked against each other on
return, risk-adjusted return, and breadth, with drill-down into the constituents
behind any ecosystem.

Reoriented from the per-name Risk & Momentum analyzer: same resilient yfinance
backbone (batch quote → per-ticker fallback, disk cache, curl_cffi impersonation,
FX-to-USD market caps, per-session returns), but the output is cross-ecosystem
comparison rather than portfolio allocation.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you get

- **Leaderboard** — ecosystems ranked by whichever metric you choose (cap- or
  equal-weighted return, excess vs benchmark, Sharpe, volatility, drawdown,
  momentum, breadth, or total market cap). Leader is highlighted; the SPY line
  is drawn on return-based rankings.
- **Risk vs return centroids** — each ecosystem as one bubble (size ≈ total
  market cap), with the benchmark's coordinates as the quadrant origin.
- **Breadth** — share of constituents with a positive return. A high
  cap-weighted return next to low breadth flags a "few mega-caps carrying it"
  ecosystem, which the `CW − EW (pp)` column also captures.
- **Scorecard** — full per-ecosystem table with a red→green gradient on the key
  columns; downloadable as CSV.
- **Drill-down** — pick an ecosystem for its constituent table, a by-sector
  breakdown, and a leaders/laggards list.

## Filters

Ecosystems compared · period · rank metric · countries · min market cap ·
min session coverage (screens out very recent IPOs) · risk-free rate · force
refresh / clear cache.

## Files

- `app.py` — Streamlit UI
- `data_layer.py` — registry + fetching + per-ticker metrics
- `aggregate.py` — ecosystem roll-up (cap- and equal-weighted)

## Notes

Cap-weighted aggregates convert local market caps to USD before weighting.
Returns are computed on each ticker's own trading sessions (no forward-fill
across foreign-market holidays, which would otherwise deflate volatility for
non-US names). Not investment advice — a screening tool on free, best-effort
market data.
