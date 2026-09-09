"""
Ecosystem Compass — high-level comparison of technology equity ecosystems.

Ranks 11 tech ecosystems against each other on return, risk-adjusted return and
breadth, shows how they moved over time, and lets you drill into constituents.
Built on the same resilient yfinance backbone as the per-name analyzer,
reoriented from portfolio allocation toward cross-ecosystem comparison.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_layer as dl
from aggregate import (aggregate_ecosystems, build_constituents,
                       contribution_breakdown, ecosystem_index_series,
                       rank_metric_options)

# ---------------------------------------------------------------------------
# Palette — clean light
# ---------------------------------------------------------------------------
BASE   = "#FFFFFF"
PANEL  = "#FFFFFF"
PANEL2 = "#F3F4F6"
LINE   = "#E2E5EA"
INK    = "#111827"
MUTED  = "#6B7280"
GAIN   = "#15803D"
LOSS   = "#DC2626"
ACCENT = "#B45309"
SEQ    = ["#2563EB", "#15803D", "#B45309", "#DC2626", "#7C3AED", "#0891B2",
          "#EA580C", "#DB2777", "#4B5563", "#0284C7", "#059669"]

st.set_page_config(page_title="AI Ecosystem Compass", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:{BASE}; color:{INK}; }}
  header[data-testid="stHeader"] {{ background:{BASE}; border-bottom:1px solid {LINE}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL2}; border-right:1px solid {LINE}; }}
  h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-0.015em; }}
  .block-container {{ padding-top:1.6rem; max-width:1480px; }}
  [data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE};
      border-radius:12px; padding:14px 16px;
      box-shadow:0 1px 2px rgba(17,24,39,0.04); }}
  [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums;
      font-weight:700; font-size:1.55rem; color:{INK}; }}
  [data-testid="stMetricLabel"] {{ color:{MUTED}; font-size:0.82rem; }}
  .lede {{ color:{MUTED}; font-size:1.03rem; line-height:1.55; max-width:76ch; }}
  .sect {{ font-size:1.12rem; font-weight:650; margin:0.4rem 0 0.2rem 0; color:{INK}; }}
  .sub {{ color:{MUTED}; font-size:0.86rem; margin-bottom:0.5rem; }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:6px;
      background:{PANEL2}; color:{INK}; font-size:0.76rem; font-weight:600;
      font-variant-numeric:tabular-nums; margin-right:7px; letter-spacing:0.02em;
      border:1px solid {LINE}; }}
  .row {{ padding:5px 0; border-bottom:1px solid {LINE}; }}
  hr {{ border-color:{LINE}; margin:1.1rem 0; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
  .stTabs [data-baseweb="tab"] {{ color:{MUTED}; }}
  .stTabs [aria-selected="true"] {{ color:{INK}; border-bottom-color:{ACCENT}!important; }}
  div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:10px;
      background:{PANEL}; }}
  .stDataFrame {{ border:1px solid {LINE}; border-radius:10px; }}

  /* Multiselect tags → white with black text */
  section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
      background-color:{PANEL} !important;
      color:{INK} !important;
      border:1px solid {LINE} !important;
  }}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] span {{
      color:{INK} !important;
  }}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] [role="button"] svg,
  section[data-testid="stSidebar"] span[data-baseweb="tag"] svg {{
      fill:{INK} !important;
      color:{INK} !important;
  }}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] [role="button"]:hover {{
      background-color:{PANEL2} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _theme(fig, height=None, legend=False):
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=INK, family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=(dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                     orientation="v") if legend else dict()),
        hoverlabel=dict(bgcolor=PANEL2, bordercolor=LINE, font_color=INK),
        showlegend=legend,
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    return fig


def _hex(r, g, b):
    return f"#{int(max(0,min(255,r))):02x}{int(max(0,min(255,g))):02x}{int(max(0,min(255,b))):02x}"


def diverging_bg(series, center=0.0):
    """White→green / white→red cell backgrounds without matplotlib (Cloud has none)."""
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return ["" for _ in s]
    lo, hi = float(finite.min()), float(finite.max())
    neg_span = max(center - lo, 1e-9)
    pos_span = max(hi - center, 1e-9)
    base = (255, 255, 255)
    out = []
    for v in s:
        if not np.isfinite(v):
            out.append(""); continue
        if v >= center:
            t = min(v / pos_span, 1.0) * 0.75
            r, g, b = (base[0] + (209 - base[0]) * t, base[1] + (237 - base[1]) * t,
                       base[2] + (216 - base[2]) * t)
        else:
            t = min((center - v) / neg_span, 1.0) * 0.75
            r, g, b = (base[0] + (250 - base[0]) * t, base[1] + (214 - base[1]) * t,
                       base[2] + (214 - base[2]) * t)
        out.append(f"background-color:{_hex(r, g, b)};color:{INK}")
    return out


def _fmt_signed(v, suffix="%"):
    return "—" if pd.isna(v) else f"{v:+.1f}{suffix}"


@st.cache_data(show_spinner=False, ttl=600)
def load_bundle(ecosystems, period, force, rf_pct):
    universe = dl.ticker_universe(ecosystems, include_benchmark=True)
    prices = dl.fetch_price_data(universe, period, force=force)
    if prices.empty or len(prices) < 2:
        return None
    live = list(prices.columns)
    fundamentals = dl.fetch_fundamentals(live, force=force)
    metrics = dl.calculate_metrics(prices, fundamentals, risk_free_rate=rf_pct / 100.0)
    return {"prices": prices, "fundamentals": fundamentals, "metrics": metrics}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    all_ecos = dl.ECOSYSTEM_CHOICES
    chosen_ecos = st.multiselect("Ecosystems to compare", all_ecos, default=all_ecos)
    period = st.selectbox("Period", dl.PERIOD_CHOICES, index=4)
    rank_placeholder = st.empty()
    weight_mode = st.radio("Weighting", ["Cap-weighted", "Equal-weighted"], index=0,
                           help="Cap = money's-eye view; Equal = typical-name view. "
                                "Drives the timeline and the ranking's return columns.")

    with st.expander("Constituent filters", expanded=False):
        country_sel = st.multiselect("Countries", dl.all_countries(), default=[])
        cap_floor = st.slider("Min market cap ($B)", 0, 500, 0, step=5)
        min_coverage = st.slider("Min session coverage (%)", 0, 100, 40, step=5,
                                 help="Screens out very recent IPOs.")

    with st.expander("Assumptions & cache", expanded=False):
        rf_pct = st.number_input("Risk-free rate (% annual)", value=4.0, step=0.25)
        force = st.checkbox("Force refresh (bypass cache)", value=False)
        if st.button("Clear cache"):
            n = dl.clear_caches()
            st.cache_data.clear()
            st.success(f"Cleared {n} cached file(s).")

    run = st.button("Run comparison", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# AI Ecosystem Compass")
st.markdown('<p class="lede">Where is capital being rewarded across the technology stack? '
            'Each ecosystem is ranked on how its constituents performed, how much risk that '
            'took, and how broadly the strength was shared — then you can open any one to see '
            'the names underneath.</p>', unsafe_allow_html=True)
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
    st.error("No price data came back. Yahoo Finance is likely rate-limiting — wait ~60s "
             "and re-run with **Force refresh** on. In a sandboxed host, confirm it can reach "
             "`query1.finance.yahoo.com` / `query2.finance.yahoo.com`.")
    st.stop()

prices = bundle["prices"]
metrics = bundle["metrics"]
fundamentals = bundle["fundamentals"]
if metrics.empty:
    st.error("Prices loaded but no ticker had enough history to score. Try a longer period.")
    st.stop()

# ---------------------------------------------------------------------------
# Build + filter constituents
# ---------------------------------------------------------------------------
ticker_eco = dl.ticker_ecosystem_map()
constituents = build_constituents(metrics, fundamentals, ticker_eco,
                                  dl.name_of, dl.sector_of, dl.country_of)
constituents = constituents[constituents["Ecosystem"].isin(chosen_ecos)].copy()

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
weighting = "cap" if weight_mode == "Cap-weighted" else "equal"
ret_col = "Return CW (%)" if weighting == "cap" else "Return EW (%)"

with rank_placeholder:
    rank_by = st.selectbox("Rank ecosystems by", rank_metric_options(has_bench),
                           index=0 if weighting == "cap" else 1)

lower_better = rank_by in ("Volatility (Ann %)", "Max Drawdown (%)")
agg_sorted = agg.sort_values(rank_by, ascending=lower_better, na_position="last").reset_index(drop=True)

period_label = {"1mo": "1 month", "3mo": "3 months", "6mo": "6 months", "ytd": "year to date",
                "1y": "1 year", "2y": "2 years"}.get(period, period)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
lead = agg_sorted.iloc[0]
best = agg.loc[agg[ret_col].idxmax()]
worst = agg.loc[agg[ret_col].idxmin()]
is_ratio = rank_by in ("Sharpe", "Beta")
is_count = rank_by in ("Mkt Cap ($B)", "Names")

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Top · {rank_by}", lead["Ecosystem"],
          f"{lead[rank_by]:,.0f}" if is_count else
          (f"{lead[rank_by]:.2f}" if is_ratio else _fmt_signed(lead[rank_by], "" if "pp" in rank_by else "%")))
k2.metric("Strongest return", best["Ecosystem"], _fmt_signed(best[ret_col]))
k3.metric("Weakest return", worst["Ecosystem"], _fmt_signed(worst[ret_col]))
if not np.isnan(bench_ret):
    beat = int((agg[ret_col] > bench_ret).sum())
    k4.metric(f"Beat {dl.BENCHMARK_TICKER} · {period_label}", f"{beat} / {len(agg)}",
              _fmt_signed(bench_ret) + " bmk", delta_color="off")
else:
    k4.metric("Ecosystems", f"{len(agg)}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TIMELINE (hero) — indexed ecosystem performance over the window
# ---------------------------------------------------------------------------
st.markdown(f'<div class="sect">Performance over time · {weight_mode.lower()}, rebased to 100</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sub">Each line is an ecosystem index; the dashed line is '
            f'{dl.BENCHMARK_TICKER}. Hover for values.</div>', unsafe_allow_html=True)

idx_df = ecosystem_index_series(prices, constituents, weighting=weighting)
if not idx_df.empty:
    order = idx_df.iloc[-1].sort_values(ascending=False).index.tolist()
    figt = go.Figure()
    for i, eco in enumerate(order):
        s = idx_df[eco].dropna()
        figt.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=eco,
            line=dict(color=SEQ[i % len(SEQ)], width=2),
            hovertemplate=f"<b>{eco}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1f}}<extra></extra>"))
    if dl.BENCHMARK_TICKER in prices.columns:
        bpx = pd.to_numeric(prices[dl.BENCHMARK_TICKER], errors="coerce").dropna()
        if not bpx.empty:
            bidx = bpx / bpx.iloc[0] * 100
            figt.add_trace(go.Scatter(
                x=bidx.index, y=bidx.values, mode="lines", name=dl.BENCHMARK_TICKER,
                line=dict(color=ACCENT, width=2.4, dash="dash"),
                hovertemplate=f"<b>{dl.BENCHMARK_TICKER}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1f}}<extra></extra>"))
    figt.add_hline(y=100, line=dict(color=MUTED, dash="dot", width=1))
    figt = _theme(figt, height=440, legend=True)
    figt.update_yaxes(title="Indexed value (start = 100)")
    figt.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                                   yanchor="top", y=1, xanchor="left", x=1.01))
    st.plotly_chart(figt, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Not enough overlapping price history to build the timeline.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Leaderboard + Risk/return
# ---------------------------------------------------------------------------
cA, cB = st.columns([1, 1], gap="large")

with cA:
    st.markdown(f'<div class="sect">Leaderboard · {rank_by}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Leader highlighted; green/red by sign.</div>', unsafe_allow_html=True)
    vals = agg_sorted[rank_by]
    neutral = rank_by in ("Volatility (Ann %)", "Max Drawdown (%)", "P/E (median)",
                          "Mkt Cap ($B)", "Names", "Beta")
    colors = ([ACCENT] + [SEQ[0]] * (len(vals) - 1)) if neutral else \
             [GAIN if v >= 0 else LOSS for v in vals]
    if not neutral:
        colors[0] = ACCENT
    figl = go.Figure(go.Bar(
        x=vals, y=agg_sorted["Ecosystem"], orientation="h", marker=dict(color=colors),
        text=[("—" if pd.isna(v) else (f"{v:,.0f}" if is_count else
              (f"{v:.2f}" if rank_by in ("Sharpe", "Beta") else f"{v:+.1f}"))) for v in vals],
        textposition="outside", textfont=dict(color=INK, size=12),
        hovertemplate="<b>%{y}</b><br>" + rank_by + ": %{x:.2f}<extra></extra>"))
    if not np.isnan(bench_ret) and rank_by in ("Return CW (%)", "Return EW (%)", "Median Return (%)"):
        figl.add_vline(x=bench_ret, line=dict(color=ACCENT, dash="dot", width=1.5))
    figl.update_layout(yaxis=dict(autorange="reversed"))
    _vmin = float(np.nanmin(vals.values)) if len(vals) else 0.0
    _vmax = float(np.nanmax(vals.values)) if len(vals) else 1.0
    _pad = max((_vmax - _vmin) * 0.18, abs(_vmax) * 0.12, 1.0)
    figl.update_xaxes(range=[min(_vmin, 0) - _pad * 0.4, _vmax + _pad])
    figl = _theme(figl, height=max(320, 44 * len(agg_sorted)))
    st.plotly_chart(figl, use_container_width=True, config={"displayModeBar": False})

with cB:
    st.markdown('<div class="sect">Risk vs return</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Bubble ≈ market cap. Up-and-left of the lines beats the benchmark '
                'with less risk.</div>', unsafe_allow_html=True)
    d = agg.dropna(subset=["Volatility (Ann %)", ret_col])
    caps = pd.to_numeric(d["Mkt Cap ($B)"], errors="coerce")
    max_cap = caps.max() if caps.notna().any() else None
    sizes = (18 + 46 * np.sqrt(caps / max_cap)).fillna(20) if max_cap else pd.Series(24, index=d.index)
    figr = go.Figure()
    figr.add_trace(go.Scatter(
        x=d["Volatility (Ann %)"], y=d[ret_col], mode="markers+text",
        text=d["Ecosystem"], textposition="top center", textfont=dict(color=MUTED, size=10),
        marker=dict(size=sizes, color=[SEQ[i % len(SEQ)] for i in range(len(d))],
                    line=dict(color=BASE, width=1.5), opacity=0.9),
        customdata=np.stack([d["Sharpe"], d["Names"]], axis=-1),
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>Vol %{x:.1f}%"
                      "<br>Sharpe %{customdata[0]:.2f}<br>%{customdata[1]} names<extra></extra>"))
    if not np.isnan(bench_ret) and bench_row is not None and not pd.isna(bench_row.get("Volatility (Ann %)")):
        bx = float(bench_row["Volatility (Ann %)"])
        figr.add_hline(y=bench_ret, line=dict(color=ACCENT, dash="dot", width=1))
        figr.add_vline(x=bx, line=dict(color=ACCENT, dash="dot", width=1))
        figr.add_annotation(x=bx, y=bench_ret, text=dl.BENCHMARK_TICKER, showarrow=False,
                            font=dict(color=ACCENT, size=11), yshift=10)
    figr.update_xaxes(title="Annualized volatility (%)")
    figr.update_yaxes(title=f"{'Cap' if weighting=='cap' else 'Equal'}-weighted return (%)")
    figr = _theme(figr, height=max(320, 44 * len(agg_sorted)))
    st.plotly_chart(figr, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Breadth + valuation/momentum quadrant
# ---------------------------------------------------------------------------
cC, cD = st.columns([1, 1], gap="large")

with cC:
    st.markdown('<div class="sect">Breadth</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">% of constituents positive. High return + low breadth = a few '
                'mega-caps carrying it.</div>', unsafe_allow_html=True)
    b = agg.sort_values("% Positive", ascending=True)
    figb = go.Figure(go.Bar(
        x=b["% Positive"], y=b["Ecosystem"], orientation="h",
        marker=dict(color=[GAIN if v >= 50 else LOSS for v in b["% Positive"]]),
        text=[f"{v:.0f}%" for v in b["% Positive"]], textposition="outside",
        textfont=dict(color=INK, size=12),
        hovertemplate="<b>%{y}</b><br>%{x:.0f}% positive<extra></extra>"))
    figb.add_vline(x=50, line=dict(color=MUTED, dash="dot", width=1))
    figb = _theme(figb, height=max(300, 42 * len(agg)))
    figb.update_xaxes(title="% positive", range=[0, 108])
    st.plotly_chart(figb, use_container_width=True, config={"displayModeBar": False})

with cD:
    st.markdown('<div class="sect">Valuation vs momentum</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Cheap + rising (bottom-right) vs expensive + fading (top-left).</div>',
                unsafe_allow_html=True)
    q = agg.dropna(subset=["P/E (median)", "Momentum (%)"])
    if not q.empty:
        figq = go.Figure(go.Scatter(
            x=q["Momentum (%)"], y=q["P/E (median)"], mode="markers+text",
            text=q["Ecosystem"], textposition="top center", textfont=dict(color=MUTED, size=10),
            marker=dict(size=20, color=[SEQ[i % len(SEQ)] for i in range(len(q))],
                        line=dict(color=BASE, width=1.5), opacity=0.9),
            hovertemplate="<b>%{text}</b><br>Median P/E %{y:.1f}<br>Momentum %{x:+.1f}%<extra></extra>"))
        figq.add_hline(y=float(q["P/E (median)"].median()), line=dict(color=MUTED, dash="dot", width=1))
        figq.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1))
        figq.update_xaxes(title="Momentum — % above 50-day SMA")
        figq.update_yaxes(title="Median P/E")
        figq = _theme(figq, height=max(300, 42 * len(agg)))
        st.plotly_chart(figq, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No P/E data resolved for these ecosystems yet — Yahoo may be rate-limiting.")

# ---------------------------------------------------------------------------
# Scorecard table (matplotlib-free coloring)
# ---------------------------------------------------------------------------
st.markdown('<div class="sect">Ecosystem scorecard</div>', unsafe_allow_html=True)
cols = ["Ecosystem", "Names", "Return CW (%)", "Return EW (%)", "CW − EW (pp)",
        "Median Return (%)", "Sharpe", "Volatility (Ann %)", "Max Drawdown (%)",
        "Beta", "Momentum (%)", "P/E (median)", "% Positive", "Mkt Cap ($B)"]
if has_bench:
    cols[3:3] = ["Excess CW (pp)", "% Beating Bmk"]
cols = [c for c in cols if c in agg_sorted.columns]
show = agg_sorted[cols].copy()

pct_fmt = {c: "{:+.1f}" for c in show.columns
           if c not in ("Ecosystem", "Names", "Sharpe", "Beta", "P/E (median)",
                        "Mkt Cap ($B)", "% Positive", "% Beating Bmk", "Volatility (Ann %)",
                        "Max Drawdown (%)")}
sty = show.style.format({**pct_fmt, "Sharpe": "{:.2f}", "Beta": "{:.2f}",
                         "P/E (median)": "{:.1f}", "Mkt Cap ($B)": "{:,.0f}",
                         "% Positive": "{:.0f}%", "Volatility (Ann %)": "{:.1f}",
                         "Max Drawdown (%)": "{:.1f}",
                         **({"% Beating Bmk": "{:.0f}%"} if has_bench else {})}, na_rep="—")
for c in ("Return CW (%)", "Return EW (%)", "Median Return (%)", "Sharpe", "Momentum (%)",
          "Excess CW (pp)"):
    if c in show.columns:
        sty = sty.apply(diverging_bg, subset=[c], center=0.0)
if "Max Drawdown (%)" in show.columns:
    sty = sty.apply(lambda s: diverging_bg(s, center=float(pd.to_numeric(s, errors="coerce").max())),
                    subset=["Max Drawdown (%)"])
st.dataframe(sty, use_container_width=True, hide_index=True)

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
    m = st.columns(5)
    m[0].metric("Cap-wtd return", _fmt_signed(row["Return CW (%)"]))
    m[1].metric("Equal-wtd return", _fmt_signed(row["Return EW (%)"]))
    m[2].metric("Sharpe", f"{row['Sharpe']:.2f}" if not pd.isna(row["Sharpe"]) else "—")
    m[3].metric("Median P/E", f"{row['P/E (median)']:.1f}" if not pd.isna(row["P/E (median)"]) else "—")
    m[4].metric("Names", f"{int(row['Names'])}")

    t1, t2, t3, t4 = st.tabs(["Constituents", "Return drivers", "By sector", "Leaders & laggards"])

    with t1:
        cc = ["Ticker", "Company", "Sector", "Country", "Market Cap ($B)",
              "Total Return (%)", "Excess vs Bench (pp)", "Volatility (Ann %)",
              "Sharpe Ratio", "Max Drawdown (%)", "Beta", "SMA Deviation (%)", "P/E Ratio"]
        cc = [c for c in cc if c in sub.columns]
        cdf = sub[cc].sort_values("Total Return (%)", ascending=False, na_position="last")
        cst = cdf.style.format({
            "Market Cap ($B)": "{:,.1f}", "Total Return (%)": "{:+.1f}",
            "Excess vs Bench (pp)": "{:+.1f}", "Volatility (Ann %)": "{:.1f}",
            "Sharpe Ratio": "{:.2f}", "Max Drawdown (%)": "{:.1f}", "Beta": "{:.2f}",
            "SMA Deviation (%)": "{:+.1f}", "P/E Ratio": "{:.1f}"}, na_rep="—")
        for c in ("Total Return (%)", "Sharpe Ratio", "SMA Deviation (%)", "Excess vs Bench (pp)"):
            if c in cdf.columns:
                cst = cst.apply(diverging_bg, subset=[c], center=0.0)
        st.dataframe(cst, use_container_width=True, hide_index=True)

    with t2:
        contrib = contribution_breakdown(sub, dd, top_n=10)
        if contrib.empty:
            st.info("Need market caps and returns to compute contributions.")
        else:
            st.markdown('<div class="sub">Cap-weighted contribution to the ecosystem return '
                        '(weight × return) — what actually moved the aggregate.</div>',
                        unsafe_allow_html=True)
            contrib = contrib.sort_values("Contribution (pp)")
            figc = go.Figure(go.Bar(
                x=contrib["Contribution (pp)"], y=contrib["Company"], orientation="h",
                marker=dict(color=[GAIN if v >= 0 else LOSS for v in contrib["Contribution (pp)"]]),
                text=[f"{v:+.1f}" for v in contrib["Contribution (pp)"]],
                textposition="outside", textfont=dict(color=INK, size=11),
                hovertemplate="<b>%{y}</b><br>%{x:+.2f} pp<extra></extra>"))
            figc = _theme(figc, height=max(260, 40 * len(contrib)))
            figc.update_xaxes(title="Contribution to ecosystem return (pp)")
            st.plotly_chart(figc, use_container_width=True, config={"displayModeBar": False})

    with t3:
        by_sec = sub.groupby("Sector").agg(
            Names=("Ticker", "count"), Return=("Total Return (%)", "mean"),
            Cap=("Market Cap ($B)", "sum")).reset_index().sort_values("Return", ascending=False)
        figs = go.Figure(go.Bar(
            x=by_sec["Return"], y=by_sec["Sector"], orientation="h",
            marker=dict(color=[GAIN if v >= 0 else LOSS for v in by_sec["Return"]]),
            text=[f"{v:+.1f}%" for v in by_sec["Return"]], textposition="outside",
            textfont=dict(color=INK, size=11), customdata=by_sec["Names"],
            hovertemplate="<b>%{y}</b><br>Avg %{x:.1f}%<br>%{customdata} names<extra></extra>"))
        figs.update_layout(yaxis=dict(autorange="reversed"))
        figs = _theme(figs, height=max(240, 42 * len(by_sec)))
        figs.update_xaxes(title="Average constituent return (%)")
        st.plotly_chart(figs, use_container_width=True, config={"displayModeBar": False})

    with t4:
        ranked = sub.dropna(subset=["Total Return (%)"]).sort_values("Total Return (%)", ascending=False)
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**Leaders**")
            for _, r in ranked.head(5).iterrows():
                st.markdown(f'<div class="row"><span class="pill">{r["Ticker"]}</span>{r["Company"]}'
                            f'<span style="float:right;color:{GAIN};font-weight:700">'
                            f'{r["Total Return (%)"]:+.1f}%</span></div>', unsafe_allow_html=True)
        with rc:
            st.markdown("**Laggards**")
            for _, r in ranked.tail(5).iloc[::-1].iterrows():
                col = LOSS if r["Total Return (%)"] < 0 else MUTED
                st.markdown(f'<div class="row"><span class="pill">{r["Ticker"]}</span>{r["Company"]}'
                            f'<span style="float:right;color:{col};font-weight:700">'
                            f'{r["Total Return (%)"]:+.1f}%</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Export + footnote
# ---------------------------------------------------------------------------
st.markdown("---")
st.download_button("Download ecosystem scorecard (CSV)",
                   agg_sorted.to_csv(index=False).encode(),
                   file_name=f"ecosystem_scorecard_{period}.csv", mime="text/csv")
st.caption("Cap-weighted aggregates use market cap converted to USD; equal-weighted treats every "
           "name alike. Returns are computed on each ticker's own trading sessions (no forward-fill "
           "across foreign-market holidays), and are measured in each stock's local currency — a "
           "unitless percentage, so directly comparable across listings; only market-cap weighting "
           "needs the FX conversion. P/E is the median of resolved trailing/forward figures. "
           "Not investment advice — a screening tool on free, best-effort market data.")
