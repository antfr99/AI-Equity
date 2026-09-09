"""
Ecosystem-level aggregation.

Takes the per-ticker metrics frame and rolls it up to one row per ecosystem,
with both cap-weighted and equal-weighted aggregates plus breadth statistics.
Cap-weighting answers "how did the money in this ecosystem do?"; equal-weighting
answers "how did the typical name do?" — they diverge sharply when a couple of
mega-caps dominate, which is exactly the insight this view exists to surface.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_layer import BENCHMARK_TICKER


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float(v.dropna().mean()) if v.notna().any() else np.nan
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def build_constituents(metrics: pd.DataFrame,
                       fundamentals: pd.DataFrame,
                       ticker_eco: Dict[str, str],
                       name_fn, sector_fn, country_fn) -> pd.DataFrame:
    """Flatten metrics into a per-constituent table tagged with ecosystem,
    sector, country and market cap. Benchmark is dropped."""
    if metrics.empty:
        return pd.DataFrame()
    df = metrics.drop(index=[BENCHMARK_TICKER], errors="ignore").copy()
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"index": "Ticker"})
    df["Ecosystem"] = df["Ticker"].map(ticker_eco).fillna("—")
    df["Company"] = df.apply(lambda r: name_fn(r["Ecosystem"], r["Ticker"]), axis=1)
    df["Sector"] = df.apply(lambda r: sector_fn(r["Ecosystem"], r["Ticker"]), axis=1)
    df["Country"] = df.apply(lambda r: country_fn(r["Ecosystem"], r["Ticker"]), axis=1)

    if "Market Cap ($B)" in fundamentals.columns:
        caps = pd.to_numeric(fundamentals["Market Cap ($B)"], errors="coerce")
        df["Market Cap ($B)"] = df["Ticker"].map(caps.to_dict())
    else:
        df["Market Cap ($B)"] = np.nan

    lead = ["Ticker", "Company", "Ecosystem", "Sector", "Country", "Market Cap ($B)"]
    ordered = lead + [c for c in df.columns if c not in lead]
    return df[ordered]


def aggregate_ecosystems(constituents: pd.DataFrame,
                         bench_row: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per ecosystem. Returns/valuation are cap-weighted; a parallel
    equal-weighted return is kept so the UI can show dispersion between the two.
    Breadth = share of names beating the benchmark."""
    if constituents.empty:
        return pd.DataFrame()

    bench_ret = float(bench_row["Total Return (%)"]) if bench_row is not None else np.nan
    rows = []
    for eco, g in constituents.groupby("Ecosystem"):
        caps = pd.to_numeric(g["Market Cap ($B)"], errors="coerce")
        ret = pd.to_numeric(g["Total Return (%)"], errors="coerce")

        n = int(ret.notna().sum())
        beat = int((ret > bench_ret).sum()) if not np.isnan(bench_ret) else np.nan
        positive = int((ret > 0).sum())

        rows.append({
            "Ecosystem": eco,
            "Names": n,
            "Return CW (%)": _wmean(ret, caps),
            "Return EW (%)": float(ret.mean()) if n else np.nan,
            "Median Return (%)": float(ret.median()) if n else np.nan,
            "Best (%)": float(ret.max()) if n else np.nan,
            "Worst (%)": float(ret.min()) if n else np.nan,
            "Dispersion (pp)": float(ret.max() - ret.min()) if n else np.nan,
            "Volatility (Ann %)": _wmean(g["Volatility (Ann %)"], caps),
            "Sharpe": _wmean(g["Sharpe Ratio"], caps),
            "Max Drawdown (%)": _wmean(g["Max Drawdown (%)"], caps),
            "Beta": _wmean(g["Beta"], caps),
            "Momentum (%)": _wmean(g["SMA Deviation (%)"], caps),
            "P/E (median)": float(pd.to_numeric(
                g["P/E Ratio"].where(g["P/E Ratio"] > 0), errors="coerce").median()),
            "Mkt Cap ($B)": float(caps.sum()) if caps.notna().any() else np.nan,
            "Breadth (beat bmk)": beat,
            "% Positive": round(100.0 * positive / n, 1) if n else np.nan,
        })

    agg = pd.DataFrame(rows)
    if not np.isnan(bench_ret):
        agg["Excess CW (pp)"] = agg["Return CW (%)"] - bench_ret
        agg["% Beating Bmk"] = (100.0 * agg["Breadth (beat bmk)"] / agg["Names"]).round(1)
    agg["CW − EW (pp)"] = (agg["Return CW (%)"] - agg["Return EW (%)"]).round(2)
    return agg.round(2)


def ecosystem_index_series(prices: pd.DataFrame,
                           constituents: pd.DataFrame,
                           weighting: str = "cap") -> pd.DataFrame:
    """Build one indexed price series (base 100) per ecosystem over time.

    Each constituent is rebased to 100 at the first date it has data, then
    combined per ecosystem — cap-weighted (money's-eye view) or equal-weighted
    (typical-name view). Weights are static, taken from current market cap, and
    renormalized each day across whichever names have data that day, so a late
    IPO simply joins the index when it starts trading rather than distorting it.
    """
    if prices.empty or constituents.empty:
        return pd.DataFrame()

    caps = (pd.to_numeric(constituents.set_index("Ticker")["Market Cap ($B)"],
                          errors="coerce")
            if "Market Cap ($B)" in constituents.columns else pd.Series(dtype=float))

    out: Dict[str, pd.Series] = {}
    for eco, g in constituents.groupby("Ecosystem"):
        members = [t for t in g["Ticker"] if t in prices.columns]
        if not members:
            continue
        block = prices[members].apply(pd.to_numeric, errors="coerce")
        # rebase each column to 100 at its own first valid observation
        rebased = block.divide(block.apply(lambda c: c[c.notna()].iloc[0]
                                           if c.notna().any() else np.nan)) * 100

        if weighting == "cap" and caps.reindex(members).notna().any():
            w = caps.reindex(members).fillna(0.0).clip(lower=0)
            if float(w.sum()) <= 0:
                w = pd.Series(1.0, index=members)
        else:
            w = pd.Series(1.0, index=members)

        wmat = pd.DataFrame(np.tile(w.values, (len(rebased), 1)),
                            index=rebased.index, columns=members)
        wmat = wmat.where(rebased.notna())               # only weight live names
        wmat = wmat.divide(wmat.sum(axis=1), axis=0)      # renormalize daily
        out[eco] = (rebased * wmat).sum(axis=1, min_count=1)

    return pd.DataFrame(out).dropna(how="all")


def contribution_breakdown(constituents: pd.DataFrame, ecosystem: str,
                           top_n: int = 8) -> pd.DataFrame:
    """Cap-weighted return contribution of each name within one ecosystem:
    weight × return. Shows what actually drove (or dragged) the aggregate."""
    g = constituents[constituents["Ecosystem"] == ecosystem].copy()
    if g.empty:
        return pd.DataFrame()
    caps = pd.to_numeric(g["Market Cap ($B)"], errors="coerce")
    ret = pd.to_numeric(g["Total Return (%)"], errors="coerce")
    mask = caps.notna() & ret.notna() & (caps > 0)
    if not mask.any():
        return pd.DataFrame()
    w = caps.where(mask, 0.0)
    w = w / w.sum()
    g["Contribution (pp)"] = (w * ret).round(2)
    g = g.dropna(subset=["Contribution (pp)"]).sort_values(
        "Contribution (pp)", ascending=False)
    if len(g) > top_n:
        head = g.head(top_n)[["Company", "Contribution (pp)"]]
        rest = pd.DataFrame([{"Company": f"Other {len(g) - top_n}",
                              "Contribution (pp)": round(
                                  g["Contribution (pp)"].iloc[top_n:].sum(), 2)}])
        return pd.concat([head, rest], ignore_index=True)
    return g[["Company", "Contribution (pp)"]].reset_index(drop=True)


def rank_metric_options(has_bench: bool) -> List[str]:
    opts = [
        "Return CW (%)", "Return EW (%)", "Median Return (%)",
        "Sharpe", "Volatility (Ann %)", "Max Drawdown (%)",
        "Momentum (%)", "% Positive", "Mkt Cap ($B)",
    ]
    if has_bench:
        opts.insert(3, "Excess CW (pp)")
        opts.insert(4, "% Beating Bmk")
    return opts
