"""
Ecosystem Compass — high-level comparison of technology equity ecosystems.

Ranks 11 tech ecosystems against each other on return, risk-adjusted return and
breadth, then lets you drill into the constituents behind any ecosystem. Built
on the same resilient yfinance backbone as the per-name analyzer, reoriented
from portfolio allocation toward cross-ecosystem comparison.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_layer as dl
from aggregate import (aggregate_ecosystems, build_constituents,
                       rank_metric_options)

# ---------------------------------------------------------------------------
# Palette — dark analyst terminal
# ---------------------------------------------------------------------------
BASE   = "#0E1117"   # page
PANEL  = "#181D27"   # cards / plots
LINE   = "#2A3140"   # gridlines, borders
INK    = "#E6E9EF"   # primary text
MUTED  = "#8B93A7"   # secondary text
GAIN   = "#22C55E"   # up
LOSS   = "#F43F5E"   # down
ACCENT = "#F5B301"   # selection / highlight / benchmark
SEQ    = ["#5B8DEF", "#22C55E", "#F5B301", "#F43F5E", "#A78BFA", "#2DD4BF",
          "#FB923C", "#F472B6", "#94A3B8", "#60A5FA", "#34D399"]

st.set_page_config(page_title="Ecosystem Compass", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:{BASE}; color:{INK}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
  h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-0.01em; }}
  .block-container {{ padding-top:2.2rem; max-width:1500px; }}
  [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums;
      font-weight:700; font-size:1.7rem; }}
  [data-testid="stMetricLabel"] {{ color:{MUTED}; }}
  .lede {{ color:{MUTED}; font-size:1.02rem; line-height:1.5; max-width:70ch; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:999px;
      background:{LINE}; color:{INK}; font-size:0.78rem; margin-right:6px; }}
  .stDataFrame {{ border:1px solid {LINE}; border-radius:10px; }}
  hr {{ border-color:{LINE}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
  .stTabs [data-baseweb="tab"] {{ color:{MUTED}; }}
  .stTabs [aria-selected="true"] {{ color:{INK}; border-bottom-color:{ACCENT}!important; }}
  div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:10px; }}
</style>
""", unsafe_allow_html=True)


def _plotly_theme(fig, height=None, legend=True):
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=INK, family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)) if legend else dict(),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font_color=INK),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------------------
# Cached data pull
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_bundle(ecosystems, period, force, rf_pct):
    universe = dl.ticker_universe(ecosystems, include_benchmark=True)
    prices = dl.fetch_price_data(universe, period, force=force)
    if prices.empty or len(prices) < 2:
        return None
    live = list(prices.columns)
    fundamentals = dl.fetch_fundamentals(live, force=force)
    metrics = dl.calculate_metrics(prices, fundamentals, risk_free_rate=rf_pct / 100.0)
    return {"prices": prices, "fundamentals": fundamentals, "metrics": metrics,
            "universe": universe, "live": live}


def _fmt_signed(v, suffix="%"):
    if pd.isna(v):
        return "—"
    return f"{v:+.1f}{suffix}"


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### Filters")
    all_ecos = dl.ECOSYSTEM_CHOICES
    chosen_ecos = st.multiselect("Ecosystems to compare", all_ecos, default=all_ecos,
                                 help="Pick which technology ecosystems enter the comparison.")
    period = st.selectbox("Period", dl.PERIOD_CHOICES, index=4)  # 1y
    rank_by_placeholder = st.empty()

    with st.expander("Constituent filters", expanded=False):
        country_sel = st.multiselect("Countries", dl.all_countries(), default=[],
                                     help="Empty = all countries.")
        cap_floor = st.slider("Min market cap ($B)", 0, 500, 0, step=5,
                              help="Drop names below this cap from every calculation.")
        min_coverage = st.slider("Min session coverage (%)", 0, 100, 40, step=5,
                                 help="Drop names that traded fewer than this share of "
                                      "benchmark sessions — filters out very recent IPOs.")

    with st.expander("Assumptions & cache", expanded=False):
        rf_pct = st.number_input("Risk-free rate (% annual)", value=4.0, step=0.25,
                                 help="Subtracted from returns before Sharpe.")
        force = st.checkbox("Force refresh (bypass cache)", value=False)
        if st.button("Clear cache"):
            n = dl.clear_caches()
            st.cache_data.clear()
            st.success(f"Cleared {n} cached file(s).")

    run = st.button("Run comparison", type="primary", width='stretch')


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Ecosystem Compass")
st.markdown('<p class="lede">Where is capital being rewarded across the technology stack? '
            'Each ecosystem is ranked on how its money-weighted constituents performed, '
            'how much risk that took, and how broadly the strength was shared — then you '
            'can open any one to see the names underneath.</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if run:
    st.session_state.has_run = True

if not chosen_ecos:
    st.info("Pick at least one ecosystem in the sidebar to begin.")
    st.stop()

if not st.session_state.has_run:
    st.info("Set your filters in the sidebar, then press **Run comparison**. "
            "Results cache for 10 minutes, so re-runs are near-instant.")
    st.stop()

with st.spinner("Fetching prices and fundamentals…"):
    bundle = load_bundle(tuple(chosen_ecos), period, force, rf_pct)

if bundle is None:
    st.error("No price data came back for this selection. Yahoo Finance is likely "
             "rate-limiting — wait about a minute, then re-run with **Force refresh** on. "
             "If you're running this in a sandboxed environment, confirm the host can "
             "reach `query1.finance.yahoo.com` and `query2.finance.yahoo.com`.")
    st.stop()

metrics = bundle["metrics"]
fundamentals = bundle["fundamentals"]
if metrics.empty:
    st.error("Prices loaded but no ticker had enough history to score. Try a longer period.")
    st.stop()

# ---------------------------------------------------------------------------
# Build constituents + apply constituent filters
# ---------------------------------------------------------------------------
ticker_eco = dl.ticker_ecosystem_map()
constituents = build_constituents(
    metrics, fundamentals, ticker_eco,
    name_fn=dl.name_of, sector_fn=dl.sector_of, country_fn=dl.country_of,
)
# keep only chosen ecosystems (a shared ticker maps to its first ecosystem)
constituents = constituents[constituents["Ecosystem"].isin(chosen_ecos)].copy()

# constituent filters
if country_sel:
    constituents = constituents[constituents["Country"].isin(country_sel)]
if cap_floor > 0:
    constituents = constituents[
        pd.to_numeric(constituents["Market Cap ($B)"], errors="coerce").fillna(0) >= cap_floor]
if min_coverage > 0 and "Coverage (%)" in constituents.columns:
    constituents = constituents[
        pd.to_numeric(constituents["Coverage (%)"], errors="coerce").fillna(0) >= min_coverage]

if constituents.empty:
    st.warning("Every constituent was filtered out. Loosen the country / market-cap / "
               "coverage filters in the sidebar.")
    st.stop()

bench_row = metrics.loc[dl.BENCHMARK_TICKER] if dl.BENCHMARK_TICKER in metrics.index else None
bench_ret = float(bench_row["Total Return (%)"]) if bench_row is not None else np.nan

agg = aggregate_ecosystems(constituents, bench_row)
has_bench = "Excess CW (pp)" in agg.columns

# rank selector (populated now that we know columns)
with rank_by_placeholder:
    rank_by = st.selectbox("Rank ecosystems by", rank_metric_options(has_bench), index=0)

lower_is_better = rank_by in ("Volatility (Ann %)", "Max Drawdown (%)")
agg_sorted = agg.sort_values(rank_by, ascending=lower_is_better, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
period_label = {"1mo": "1 month", "3mo": "3 months", "6mo": "6 months",
                "ytd": "year to date", "1y": "1 year", "2y": "2 years"}.get(period, period)

lead = agg_sorted.iloc[0]
best_ret = agg.loc[agg["Return CW (%)"].idxmax()]
worst_ret = agg.loc[agg["Return CW (%)"].idxmin()]

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Top ecosystem · {rank_by}", lead["Ecosystem"],
          _fmt_signed(lead[rank_by], "" if "pp" in rank_by or rank_by in ("Sharpe", "Beta") else "%")
          if rank_by not in ("Mkt Cap ($B)", "Names") else f"{lead[rank_by]:,.0f}")
k2.metric("Strongest return (cap-wtd)", best_ret["Ecosystem"], _fmt_signed(best_ret["Return CW (%)"]))
k3.metric("Weakest return (cap-wtd)", worst_ret["Ecosystem"], _fmt_signed(worst_ret["Return CW (%)"]))
if not np.isnan(bench_ret):
    beat_count = int((agg["Return CW (%)"] > bench_ret).sum())
    k4.metric(f"Beat {dl.BENCHMARK_TICKER} ({period_label})",
              f"{beat_count} / {len(agg)}", _fmt_signed(bench_ret) + " bmk")
else:
    k4.metric("Ecosystems compared", f"{len(agg)}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO — ecosystem leaderboard
# ---------------------------------------------------------------------------
st.markdown(f"#### Leaderboard · ranked by {rank_by}")

vals = agg_sorted[rank_by]
if rank_by in ("Volatility (Ann %)", "Max Drawdown (%)", "P/E (median)", "Mkt Cap ($B)", "Names"):
    bar_colors = [ACCENT] + [MUTED] * (len(vals) - 1)  # highlight the leader only
else:
    bar_colors = [GAIN if v >= 0 else LOSS for v in vals]
    bar_colors[0] = ACCENT  # leader always highlighted

fig = go.Figure(go.Bar(
    x=vals, y=agg_sorted["Ecosystem"], orientation="h",
    marker=dict(color=bar_colors),
    text=[f"{v:+.1f}" if v == v else "—" for v in vals],
    textposition="outside", textfont=dict(color=INK),
    hovertemplate="<b>%{y}</b><br>" + rank_by + ": %{x:.2f}<extra></extra>",
))
if not np.isnan(bench_ret) and rank_by in ("Return CW (%)", "Return EW (%)", "Median Return (%)"):
    fig.add_vline(x=bench_ret, line=dict(color=ACCENT, dash="dot", width=1.5),
                  annotation_text=f"{dl.BENCHMARK_TICKER} {bench_ret:+.1f}%",
                  annotation_font_color=ACCENT, annotation_position="top")
fig.update_layout(yaxis=dict(autorange="reversed"))
fig = _plotly_theme(fig, height=max(320, 42 * len(agg_sorted)), legend=False)
st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Risk / return positioning + breadth
# ---------------------------------------------------------------------------
c_left, c_right = st.columns([3, 2], gap="large")

with c_left:
    st.markdown("#### Risk vs return — ecosystem centroids")
    d = agg.dropna(subset=["Volatility (Ann %)", "Return CW (%)"])
    caps = pd.to_numeric(d["Mkt Cap ($B)"], errors="coerce")
    max_cap = caps.max() if caps.notna().any() else None
    sizes = (18 + 46 * np.sqrt(caps / max_cap)).fillna(20) if max_cap else pd.Series(24, index=d.index)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=d["Volatility (Ann %)"], y=d["Return CW (%)"], mode="markers+text",
        text=d["Ecosystem"], textposition="top center", textfont=dict(color=MUTED, size=10),
        marker=dict(size=sizes, color=[SEQ[i % len(SEQ)] for i in range(len(d))],
                    line=dict(color=BASE, width=1.5), opacity=0.9),
        customdata=np.stack([d["Sharpe"], d["Names"]], axis=-1),
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>Vol %{x:.1f}%"
                      "<br>Sharpe %{customdata[0]:.2f}<br>%{customdata[1]} names<extra></extra>",
    ))
    if not np.isnan(bench_ret) and bench_row is not None and not pd.isna(bench_row.get("Volatility (Ann %)")):
        bx, by = float(bench_row["Volatility (Ann %)"]), bench_ret
        fig2.add_hline(y=by, line=dict(color=ACCENT, dash="dot", width=1))
        fig2.add_vline(x=bx, line=dict(color=ACCENT, dash="dot", width=1))
        fig2.add_annotation(x=bx, y=by, text=dl.BENCHMARK_TICKER, showarrow=False,
                            font=dict(color=ACCENT, size=11), yshift=10)
    fig2.update_xaxes(title="Annualized volatility (%)  →  more risk")
    fig2.update_yaxes(title="Cap-weighted return (%)")
    fig2 = _plotly_theme(fig2, height=430, legend=False)
    st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
    st.caption("Bubble size ≈ total market cap. Up-and-left of the dotted lines = beat the "
               "benchmark with less risk.")

with c_right:
    st.markdown("#### Breadth — how broad is the strength?")
    b = agg.sort_values("% Positive", ascending=True)
    fig3 = go.Figure(go.Bar(
        x=b["% Positive"], y=b["Ecosystem"], orientation="h",
        marker=dict(color=[GAIN if v >= 50 else LOSS for v in b["% Positive"]]),
        text=[f"{v:.0f}%" for v in b["% Positive"]], textposition="outside",
        textfont=dict(color=INK),
        hovertemplate="<b>%{y}</b><br>%{x:.0f}% of names positive<extra></extra>",
    ))
    fig3.add_vline(x=50, line=dict(color=MUTED, dash="dot", width=1))
    fig3 = _plotly_theme(fig3, height=430, legend=False)
    fig3.update_xaxes(title="% of constituents with a positive return", range=[0, 105])
    st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})
    st.caption("A high cap-weighted return with low breadth means a few mega-caps are "
               "carrying the ecosystem.")

# ---------------------------------------------------------------------------
# Ecosystem summary table
# ---------------------------------------------------------------------------
st.markdown("#### Ecosystem scorecard")
table_cols = ["Ecosystem", "Names", "Return CW (%)", "Return EW (%)", "CW − EW (pp)",
              "Median Return (%)", "Sharpe", "Volatility (Ann %)", "Max Drawdown (%)",
              "Beta", "Momentum (%)", "P/E (median)", "% Positive", "Mkt Cap ($B)"]
if has_bench:
    table_cols[3:3] = ["Excess CW (pp)", "% Beating Bmk"]
table_cols = [c for c in table_cols if c in agg_sorted.columns]
show = agg_sorted[table_cols].copy()

sty = show.style.format({c: "{:+.1f}" for c in show.columns
                         if c not in ("Ecosystem", "Names", "Sharpe", "Beta",
                                      "P/E (median)", "Mkt Cap ($B)", "% Positive",
                                      "% Beating Bmk")})
sty = sty.format({"Sharpe": "{:.2f}", "Beta": "{:.2f}", "P/E (median)": "{:.1f}",
                  "Mkt Cap ($B)": "{:,.0f}", "% Positive": "{:.0f}%",
                  **({"% Beating Bmk": "{:.0f}%"} if has_bench else {})}, na_rep="—")
grad_cols = [c for c in ("Return CW (%)", "Sharpe", "Momentum (%)") if c in show.columns]
if grad_cols:
    sty = sty.background_gradient(cmap="RdYlGn", subset=grad_cols, vmin=None, vmax=None)
st.dataframe(sty, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Drill into an ecosystem")

dd = st.selectbox("Ecosystem", agg_sorted["Ecosystem"].tolist(), index=0)
sub = constituents[constituents["Ecosystem"] == dd].copy()

if sub.empty:
    st.info("No constituents survive the current filters for this ecosystem.")
else:
    row = agg[agg["Ecosystem"] == dd].iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cap-wtd return", _fmt_signed(row["Return CW (%)"]))
    m2.metric("Equal-wtd return", _fmt_signed(row["Return EW (%)"]))
    m3.metric("Sharpe", f"{row['Sharpe']:.2f}" if not pd.isna(row["Sharpe"]) else "—")
    m4.metric("Median P/E", f"{row['P/E (median)']:.1f}" if not pd.isna(row["P/E (median)"]) else "—")
    m5.metric("Names", f"{int(row['Names'])}")

    tab1, tab2, tab3 = st.tabs(["Constituents", "By sector", "Leaders & laggards"])

    with tab1:
        cols = ["Ticker", "Company", "Sector", "Country", "Market Cap ($B)",
                "Total Return (%)", "Excess vs Bench (pp)", "Volatility (Ann %)",
                "Sharpe Ratio", "Max Drawdown (%)", "Beta", "SMA Deviation (%)", "P/E Ratio"]
        cols = [c for c in cols if c in sub.columns]
        cdf = sub[cols].sort_values("Total Return (%)", ascending=False, na_position="last")
        csty = cdf.style.format({
            "Market Cap ($B)": "{:,.1f}", "Total Return (%)": "{:+.1f}",
            "Excess vs Bench (pp)": "{:+.1f}", "Volatility (Ann %)": "{:.1f}",
            "Sharpe Ratio": "{:.2f}", "Max Drawdown (%)": "{:.1f}", "Beta": "{:.2f}",
            "SMA Deviation (%)": "{:+.1f}", "P/E Ratio": "{:.1f}",
        }, na_rep="—").background_gradient(
            cmap="RdYlGn", subset=[c for c in ("Total Return (%)", "Sharpe Ratio") if c in cdf.columns])
        st.dataframe(csty, width='stretch', hide_index=True)

    with tab2:
        by_sec = sub.groupby("Sector").agg(
            Names=("Ticker", "count"),
            Return=("Total Return (%)", "mean"),
            Sharpe=("Sharpe Ratio", "mean"),
            Cap=("Market Cap ($B)", "sum"),
        ).reset_index().sort_values("Return", ascending=False)
        figs = go.Figure(go.Bar(
            x=by_sec["Return"], y=by_sec["Sector"], orientation="h",
            marker=dict(color=[GAIN if v >= 0 else LOSS for v in by_sec["Return"]]),
            text=[f"{v:+.1f}%" for v in by_sec["Return"]], textposition="outside",
            textfont=dict(color=INK),
            customdata=by_sec["Names"],
            hovertemplate="<b>%{y}</b><br>Avg return %{x:.1f}%<br>%{customdata} names<extra></extra>",
        ))
        figs.update_layout(yaxis=dict(autorange="reversed"))
        figs = _plotly_theme(figs, height=max(260, 46 * len(by_sec)), legend=False)
        figs.update_xaxes(title="Average constituent return (%)")
        st.plotly_chart(figs, width='stretch', config={"displayModeBar": False})

    with tab3:
        ranked = sub.dropna(subset=["Total Return (%)"]).sort_values("Total Return (%)", ascending=False)
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**Leaders**")
            for _, r in ranked.head(5).iterrows():
                st.markdown(f"<span class='pill'>{r['Ticker']}</span> {r['Company']} "
                            f"<span style='color:{GAIN};font-weight:700'>"
                            f"{r['Total Return (%)']:+.1f}%</span>", unsafe_allow_html=True)
        with rc:
            st.markdown("**Laggards**")
            for _, r in ranked.tail(5).iloc[::-1].iterrows():
                col = LOSS if r["Total Return (%)"] < 0 else MUTED
                st.markdown(f"<span class='pill'>{r['Ticker']}</span> {r['Company']} "
                            f"<span style='color:{col};font-weight:700'>"
                            f"{r['Total Return (%)']:+.1f}%</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Export + footnote
# ---------------------------------------------------------------------------
st.markdown("---")
csv = agg_sorted.to_csv(index=False).encode()
st.download_button("Download ecosystem scorecard (CSV)", csv,
                   file_name=f"ecosystem_scorecard_{period}.csv", mime="text/csv")

st.caption(
    "Cap-weighted aggregates use market cap converted to USD; equal-weighted treats every "
    "name alike. Returns are computed on each ticker's own trading sessions (no forward-fill "
    "across foreign-market holidays). P/E is the median of resolved trailing/forward figures. "
    "Not investment advice — a screening tool on free, best-effort market data."
)
