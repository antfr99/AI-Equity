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

- **Timeline** — each ecosystem as an indexed price series (base 100) over the
  window, cap- or equal-weighted, with SPY dashed on top. Weights renormalize
  daily across whichever names have data, so a late IPO joins the index cleanly
  instead of distorting it.
- **Leaderboard** — ecosystems ranked by whichever metric you choose (cap- or
  equal-weighted return, excess vs benchmark, Sharpe, volatility, drawdown,
  momentum, breadth, or total market cap). Leader is highlighted; the SPY line
  is drawn on return-based rankings.
- **Risk vs return centroids** — each ecosystem as one bubble (size ≈ total
  market cap), with the benchmark's coordinates as the quadrant origin.
- **Breadth** — share of constituents with a positive return. A high
  cap-weighted return next to low breadth flags a "few mega-caps carrying it"
  ecosystem, which the `CW − EW (pp)` column also captures.
- **Valuation vs momentum** — median P/E against % above the 50-day SMA, so you
  can spot cheap-and-rising vs expensive-and-fading ecosystems at a glance.
- **Scorecard** — full per-ecosystem table with a red→green gradient on the key
  columns; downloadable as CSV.
- **Drill-down** — pick an ecosystem for its constituent table, a **return
  drivers** chart (cap-weighted contribution = weight × return, so you see which
  names actually moved the aggregate), a by-sector breakdown, and a
  leaders/laggards list.

The table coloring is computed in pure Python (no matplotlib dependency), so it
renders on a bare Streamlit Cloud install.

## Theme

`.streamlit/config.toml` pins a clean white/light theme (`base = "light"`).
Deploy it alongside `app.py` — dotfolders sometimes get skipped by drag-and-drop
uploads or `.gitignore` templates, so confirm `.streamlit/config.toml` is
actually present in your repo, and do a full app reboot (not just a rerun)
after adding or changing it, since theme config is only read at process start.

## Filters

Ecosystems compared · period · rank metric · countries · min market cap ·
min session coverage (screens out very recent IPOs) · risk-free rate · force
refresh / clear cache.

## Files

- `app.py` — Streamlit UI
- `data_layer.py` — registry + fetching + per-ticker metrics
- `aggregate.py` — ecosystem roll-up, timeline index, contribution breakdown
- `.streamlit/config.toml` — pins the dark theme (keep this next to `app.py`;
  without it the top bar and widget accents fall back to Streamlit defaults)

## Notes

Cap-weighted aggregates convert local market caps to USD before weighting.
Returns are computed on each ticker's own trading sessions (no forward-fill
across foreign-market holidays, which would otherwise deflate volatility for
non-US names). Not investment advice — a screening tool on free, best-effort
market data.
