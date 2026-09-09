"""
Data layer for the Ecosystem Comparison app.

Reuses the ECOSYSTEMS registry and the resilient yfinance fetching backbone
from the original Gradio analyzer, but reorients the outputs toward
ecosystem-level aggregation rather than portfolio allocation.
"""
import hashlib
import os
import pickle
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK_TICKER = "SPY"
BENCHMARK_NAME = "S&P 500 ETF"
BENCHMARK_SECTOR = "Benchmark"

TRADING_DAYS = 252

PERIOD_CHOICES = ["1mo", "3mo", "6mo", "ytd", "1y", "2y"]

CACHE_DIR = Path(os.environ.get("EQUITY_CACHE_DIR", Path.home() / ".cache" / "eco_compare"))
FUNDAMENTAL_TTL = 6 * 3600
PRICE_TTL = 10 * 60
FX_TTL = 6 * 3600

MAX_WORKERS = 4
MIN_REQUEST_INTERVAL = 0.12
INFO_RETRIES = 2

_MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01),
                "ILA": ("ILS", 0.01), "ZAc": ("ZAR", 0.01)}

try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _exc:
    print(f"[WARN] disk cache disabled: {_exc}")
    CACHE_DIR = None


# ---------------------------------------------------------------------------
# Ecosystem registry (verbatim from the original app)
# ---------------------------------------------------------------------------
ECOSYSTEMS = {
    "Semiconductors": {
        "SECTOR_COMPANIES": {
            "Chip Designers": ["NVDA", "AMD", "QCOM", "AVGO", "ARM", "AAPL"],
            "Foundries": ["TSM", "0981.HK"],
            "Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "ASM.AS", "BESI.AS", "ICHR", "UCTT"],
            "IDM": ["INTC"],
            "Analog/Power": ["ADI", "TXN", "ON", "STM", "WOLF"],
            "Automated Test Equipment": ["TER", "6857.T", "AEHR"],
            "Test & Assembly (OSAT)": ["AMKR", "ASX", "2449.TW", "6239.TW", "IMOS"],
            "Wafers & Materials": ["4063.T", "3436.T", "6488.TWO", "ENTG"],
        },
        "COMPANY_COUNTRIES": {
            "NVDA": "USA", "AMD": "USA", "QCOM": "USA", "AVGO": "USA", "ARM": "UK", "AAPL": "USA",
            "TSM": "Taiwan", "0981.HK": "China",
            "ASML": "Netherlands", "ASM.AS": "Netherlands", "BESI.AS": "Netherlands",
            "AMAT": "USA", "LRCX": "USA", "KLAC": "USA", "INTC": "USA",
            "ADI": "USA", "TXN": "USA", "ON": "USA", "STM": "Switzerland", "WOLF": "USA",
            "TER": "USA", "6857.T": "Japan", "AEHR": "USA",
            "AMKR": "USA", "ASX": "Taiwan", "2449.TW": "Taiwan", "6239.TW": "Taiwan", "IMOS": "Taiwan",
            "4063.T": "Japan", "3436.T": "Japan", "6488.TWO": "Taiwan", "ENTG": "USA",
            "ICHR": "USA", "UCTT": "USA",
        },
        "SHORT_NAMES": {
            "NVDA": "NVIDIA", "AMD": "AMD", "QCOM": "Qualcomm", "AVGO": "Broadcom", "ARM": "Arm",
            "AAPL": "Apple", "TSM": "TSMC", "0981.HK": "SMIC",
            "ASML": "ASML", "AMAT": "Applied Materials", "LRCX": "Lam Research",
            "KLAC": "KLA", "ASM.AS": "ASM Int.", "BESI.AS": "BE Semiconductor",
            "INTC": "Intel", "ADI": "Analog Devices", "TXN": "Texas Instr.", "ON": "Onsemi",
            "STM": "STMicroelectronics", "WOLF": "Wolfspeed",
            "TER": "Teradyne", "6857.T": "Advantest", "AEHR": "Aehr Test Systems",
            "AMKR": "Amkor", "ASX": "ASE Technology", "2449.TW": "King Yuan",
            "6239.TW": "Powertech", "IMOS": "ChipMOS",
            "4063.T": "Shin-Etsu Chemical", "3436.T": "SUMCO", "6488.TWO": "GlobalWafers",
            "ENTG": "Entegris", "ICHR": "Ichor Holdings", "UCTT": "Ultra Clean Holdings",
        },
    },
    "Optical & Photonics": {
        "SECTOR_COMPANIES": {
            "Optical Components & Transceivers": ["LITE", "AAOI", "POET", "LWLG"],
            "Optical Semiconductors & Connectivity": ["MTSI", "AXTI"],
            "Photonics & Laser Systems": ["COHR", "LASR"],
            "Optical Manufacturing": ["FN"],
            "Optical Networking": ["NOK"],
            "Fiber Infrastructure": ["GLW"],
        },
        "COMPANY_COUNTRIES": {
            "COHR": "USA", "LITE": "USA", "MTSI": "USA", "AAOI": "USA", "POET": "Canada",
            "FN": "Thailand", "LWLG": "USA", "NOK": "Finland", "AXTI": "USA",
            "LASR": "USA", "GLW": "USA",
        },
        "SHORT_NAMES": {
            "COHR": "Coherent", "LITE": "Lumentum", "MTSI": "MACOM", "AAOI": "AOI",
            "FN": "Fabrinet", "POET": "POET", "LWLG": "Lightwave", "AXTI": "AXT",
            "LASR": "nLIGHT", "NOK": "Nokia", "GLW": "Corning",
        },
    },
    "Quantum Computing": {
        "SECTOR_COMPANIES": {
            "Quantum Hardware (Trapped Ion)": ["IONQ", "INFQ"],
            "Quantum Hardware (Superconducting)": ["RGTI", "QBTS"],
            "Quantum Hardware (Photonic)": ["XNDU"],
            "Quantum Software & Applications": ["HQ", "QUBT"],
            "Quantum Networking & Security": ["ARQQ"],
            "Quantum Ecosystem Partners": ["IBM", "HON", "GFS"],
        },
        "COMPANY_COUNTRIES": {
            "IONQ": "USA", "INFQ": "USA", "RGTI": "USA", "QBTS": "Canada", "XNDU": "Canada",
            "HQ": "Singapore", "QUBT": "USA", "ARQQ": "UK", "IBM": "USA", "HON": "USA", "GFS": "USA",
        },
        "SHORT_NAMES": {
            "IONQ": "IonQ", "INFQ": "Infleqtion", "RGTI": "Rigetti", "QBTS": "D-Wave",
            "XNDU": "Xanadu", "HQ": "Horizon", "QUBT": "QCI", "ARQQ": "Arqit",
            "IBM": "IBM", "HON": "Honeywell", "GFS": "GlobalFoundries",
        },
    },
    "Memory & Storage": {
        "SECTOR_COMPANIES": {
            "DRAM Manufacturers": ["005930.KS", "000660.KS", "MU", "2408.TW", "2344.TW"],
            "NAND Flash Manufacturers": ["285A.T"],
            "Storage & SSD": ["SNDK"],
            "Specialty Memory": ["2344.TW", "2408.TW"],
            "Data Storage Infrastructure": ["STX", "WDC"],
        },
        "COMPANY_COUNTRIES": {
            "005930.KS": "South Korea", "000660.KS": "South Korea", "MU": "USA",
            "STX": "USA", "SNDK": "USA", "285A.T": "Japan", "WDC": "USA",
            "2408.TW": "Taiwan", "2344.TW": "Taiwan",
        },
        "SHORT_NAMES": {
            "005930.KS": "Samsung", "000660.KS": "SK Hynix", "MU": "Micron",
            "STX": "Seagate", "SNDK": "SanDisk", "285A.T": "Kioxia",
            "WDC": "WDC", "2408.TW": "Nanya", "2344.TW": "Winbond",
        },
    },
    "AI Data Center Infrastructure": {
        "SECTOR_COMPANIES": {
            "Data Center Power & Cooling": ["VRT", "ETN"],
            "AI Servers & Infrastructure": ["DELL", "HPE", "SMCI"],
            "AI Data Centers": ["APLD", "EQIX", "DLR"],
            "Server ODMs & Assembly": ["2317.TW", "2382.TW", "6669.TW"],
            "Liquid Cooling": ["3017.TW"],
        },
        "COMPANY_COUNTRIES": {
            "VRT": "USA", "ETN": "Ireland", "DELL": "USA", "HPE": "USA", "SMCI": "USA",
            "APLD": "USA", "EQIX": "USA", "DLR": "USA", "2317.TW": "Taiwan",
            "2382.TW": "Taiwan", "6669.TW": "Taiwan", "3017.TW": "Taiwan",
        },
        "SHORT_NAMES": {
            "VRT": "Vertiv", "ETN": "Eaton", "DELL": "Dell", "HPE": "HPE", "SMCI": "Supermicro",
            "APLD": "Applied Digital", "EQIX": "Equinix", "DLR": "Digital Realty",
            "2317.TW": "Foxconn (Hon Hai)", "2382.TW": "Quanta Computer",
            "6669.TW": "Wiwynn", "3017.TW": "Asia Vital Components",
        },
    },
    "AI Networking & Connectivity": {
        "SECTOR_COMPANIES": {
            "Data Center Networking": ["ANET", "CSCO"],
            "High-Speed Interconnects": ["APH", "ALAB"],
            "AI Connectivity & Custom Silicon": ["MRVL", "CRDO"],
            "RF Semiconductors": ["QRVO"],
        },
        "COMPANY_COUNTRIES": {
            "ANET": "USA", "CSCO": "USA", "APH": "USA", "ALAB": "USA",
            "MRVL": "USA", "CRDO": "USA", "QRVO": "USA",
        },
        "SHORT_NAMES": {
            "ANET": "Arista", "CSCO": "Cisco", "APH": "Amphenol", "ALAB": "Astera Labs",
            "MRVL": "Marvell", "CRDO": "Credo", "QRVO": "Qorvo",
        },
    },
    "Semiconductor Design & Specialty Chips": {
        "SECTOR_COMPANIES": {
            "Electronic Design Automation": ["SNPS", "CDNS"],
            "Edge AI Semiconductors": ["AMBA"],
            "Automotive Semiconductors": ["INDI"],
            "Power Management Semiconductors": ["MPWR"],
            "Microcontrollers & Embedded": ["MCHP"],
        },
        "COMPANY_COUNTRIES": {
            "SNPS": "USA", "CDNS": "USA", "AMBA": "USA", "INDI": "USA", "MPWR": "USA", "MCHP": "USA",
        },
        "SHORT_NAMES": {
            "SNPS": "Synopsys", "CDNS": "Cadence", "AMBA": "Ambarella",
            "INDI": "indie Semiconductor", "MPWR": "Monolithic Power", "MCHP": "Microchip Technology",
        },
    },
    "Cloud Platforms & AI Compute": {
        "SECTOR_COMPANIES": {
            "Hyperscale Cloud": ["MSFT", "AMZN", "ORCL", "BABA", "GOOGL"],
            "AI Cloud": ["CRWV", "NBIS", "DOCN"],
        },
        "COMPANY_COUNTRIES": {
            "MSFT": "USA", "AMZN": "USA", "ORCL": "USA", "BABA": "China", "GOOGL": "USA",
            "CRWV": "USA", "NBIS": "Netherlands", "DOCN": "USA",
        },
        "SHORT_NAMES": {
            "MSFT": "Microsoft", "AMZN": "Amazon", "ORCL": "Oracle", "BABA": "Alibaba",
            "GOOGL": "Alphabet", "CRWV": "CoreWeave", "NBIS": "Nebius", "DOCN": "DigitalOcean",
        },
    },
    "Enterprise AI & Software": {
        "SECTOR_COMPANIES": {
            "Enterprise AI": ["PLTR", "AI", "NOW", "IOT"],
            "CRM & Business Applications": ["CRM", "ADBE", "WDAY", "INTU", "VEEV"],
            "Data Platforms": ["SNOW", "MDB"],
            "Developer & Observability Platforms": ["DDOG", "ESTC", "TEAM", "GTLB", "DT"],
            "Automation": ["PATH", "APPN"],
            "Consumer & Social AI": ["META"],
            "Voice & Conversational AI": ["SOUN", "TWLO"],
            "Design & Engineering Software": ["ADSK"],
            "Marketing & Sales Platforms": ["HUBS"],
            "Cloud Storage & Collaboration": ["BOX"],
            "3D & Interactive Development Platforms": ["U"],
        },
        "COMPANY_COUNTRIES": {
            "PLTR": "USA", "AI": "USA", "NOW": "USA", "IOT": "USA", "CRM": "USA", "ADBE": "USA",
            "SNOW": "USA", "MDB": "USA", "DDOG": "USA", "ESTC": "USA", "PATH": "USA",
            "META": "USA", "SOUN": "USA", "WDAY": "USA", "ADSK": "USA", "INTU": "USA",
            "VEEV": "USA", "HUBS": "USA", "TEAM": "Australia", "TWLO": "USA", "APPN": "USA",
            "BOX": "USA", "U": "USA", "DT": "USA", "GTLB": "USA",
        },
        "SHORT_NAMES": {
            "PLTR": "Palantir", "AI": "C3.ai", "NOW": "ServiceNow", "IOT": "Samsara",
            "CRM": "Salesforce", "ADBE": "Adobe", "SNOW": "Snowflake", "MDB": "MongoDB",
            "DDOG": "Datadog", "ESTC": "Elastic", "PATH": "UiPath", "META": "Meta",
            "SOUN": "SoundHound AI", "WDAY": "Workday", "ADSK": "Autodesk", "INTU": "Intuit",
            "VEEV": "Veeva Systems", "HUBS": "HubSpot", "TEAM": "Atlassian", "TWLO": "Twilio",
            "APPN": "Appian", "BOX": "Box", "U": "Unity", "DT": "Dynatrace", "GTLB": "GitLab",
        },
    },
    "Cybersecurity": {
        "SECTOR_COMPANIES": {
            "Endpoint & Cloud Security": ["CRWD"],
            "Network Security": ["PANW", "FTNT"],
            "Cloud & Edge Security": ["NET"],
        },
        "COMPANY_COUNTRIES": {"CRWD": "USA", "PANW": "USA", "FTNT": "USA", "NET": "USA"},
        "SHORT_NAMES": {
            "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks", "FTNT": "Fortinet", "NET": "Cloudflare",
        },
    },
    "Energy & Power Infrastructure": {
        "SECTOR_COMPANIES": {
            "Nuclear & Power Generation": ["CEG"],
            "Power Generation & Grid": ["GEV", "NEE"],
        },
        "COMPANY_COUNTRIES": {"CEG": "USA", "GEV": "USA", "NEE": "USA"},
        "SHORT_NAMES": {"CEG": "Constellation Energy", "GEV": "GE Vernova", "NEE": "NextEra Energy"},
    },
}
ECOSYSTEM_CHOICES = list(ECOSYSTEMS.keys())


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def all_tickers() -> List[str]:
    seen: Dict[str, None] = {}
    for cfg in ECOSYSTEMS.values():
        for lst in cfg.get("SECTOR_COMPANIES", {}).values():
            for t in lst:
                seen.setdefault(t, None)
    seen.setdefault(BENCHMARK_TICKER, None)
    return list(seen)


def ecosystem_tickers(ecosystem: str) -> List[str]:
    sc = ECOSYSTEMS.get(ecosystem, {}).get("SECTOR_COMPANIES", {})
    out: Dict[str, None] = {}
    for lst in sc.values():
        for t in lst:
            out.setdefault(t, None)
    return list(out)


def ticker_universe(ecosystems: Sequence[str], include_benchmark: bool = True) -> List[str]:
    out: Dict[str, None] = {}
    for eco in ecosystems:
        for t in ecosystem_tickers(eco):
            out.setdefault(t, None)
    if include_benchmark:
        out.setdefault(BENCHMARK_TICKER, None)
    return list(out)


def ticker_ecosystem_map() -> Dict[str, str]:
    """First ecosystem a ticker appears in (a few names, e.g. IBM, appear once)."""
    m: Dict[str, str] = {}
    for eco in ECOSYSTEM_CHOICES:
        for t in ecosystem_tickers(eco):
            m.setdefault(t, eco)
    return m


def sector_of(ecosystem: str, ticker: str) -> str:
    sc = ECOSYSTEMS.get(ecosystem, {}).get("SECTOR_COMPANIES", {})
    for sector, tickers in sc.items():
        if ticker in tickers:
            return sector
    return "—"


def name_of(ecosystem: str, ticker: str) -> str:
    return ECOSYSTEMS.get(ecosystem, {}).get("SHORT_NAMES", {}).get(ticker, ticker)


def country_of(ecosystem: str, ticker: str) -> str:
    return ECOSYSTEMS.get(ecosystem, {}).get("COMPANY_COUNTRIES", {}).get(ticker, "—")


def global_name(ticker: str) -> str:
    if ticker == BENCHMARK_TICKER:
        return BENCHMARK_NAME
    for cfg in ECOSYSTEMS.values():
        n = cfg.get("SHORT_NAMES", {}).get(ticker)
        if n:
            return n
    return ticker


def global_country(ticker: str) -> str:
    if ticker == BENCHMARK_TICKER:
        return "USA"
    for cfg in ECOSYSTEMS.values():
        c = cfg.get("COMPANY_COUNTRIES", {}).get(ticker)
        if c:
            return c
    return "—"


def all_countries() -> List[str]:
    cs: Set[str] = set()
    for cfg in ECOSYSTEMS.values():
        cs.update(cfg.get("COMPANY_COUNTRIES", {}).values())
    return sorted(cs)


# ---------------------------------------------------------------------------
# Rate limiting + caching (from original)
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_LIMITER = _RateLimiter(MIN_REQUEST_INTERVAL)
_FUNDAMENTAL_CACHE: Dict[str, Tuple[float, dict]] = {}


def _cache_file(kind: str, key: str) -> Optional[Path]:
    if CACHE_DIR is None:
        return None
    digest = hashlib.md5(f"{kind}:{key}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{kind}_{digest}.pkl"


def _disk_get(kind: str, key: str, ttl: float):
    path = _cache_file(kind, key)
    if path is None or not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _disk_put(kind: str, key: str, obj) -> None:
    path = _cache_file(kind, key)
    if path is None:
        return
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh)
        tmp.replace(path)
    except Exception as exc:
        print(f"[WARN] cache write failed: {exc}")


def clear_caches() -> int:
    _FUNDAMENTAL_CACHE.clear()
    removed = 0
    if CACHE_DIR is not None:
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def _build_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


_SESSION = _build_session()


def _mk_ticker(ticker: str):
    if _SESSION is not None:
        try:
            return yf.Ticker(ticker, session=_SESSION)
        except Exception:
            pass
    return yf.Ticker(ticker)


def _fi_get(fast_info, *keys):
    for k in keys:
        try:
            val = fast_info[k]
            if val is not None:
                return val
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Batch quote + fundamentals
# ---------------------------------------------------------------------------
_QUOTE_URL = "https://query2.finance.yahoo.com/v7/finance/quote"
_QUOTE_FIELDS = ("symbol,regularMarketPrice,marketCap,trailingPE,forwardPE,"
                 "epsTrailingTwelveMonths,epsForward,currency")


def _batch_quote(tickers: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    tickers = list(tickers)
    if not tickers:
        return out
    try:
        from yfinance.data import YfData
    except Exception as exc:
        print(f"[INFO] batch quote unavailable ({exc})")
        return out
    try:
        yfd = YfData(session=_SESSION)
    except Exception:
        try:
            yfd = YfData()
        except Exception as exc:
            print(f"[INFO] YfData init failed ({exc})")
            return out

    consecutive_failures = 0
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        params = {"symbols": ",".join(chunk), "fields": _QUOTE_FIELDS}
        try:
            try:
                params["crumb"] = yfd._get_crumb()
            except Exception:
                pass
            js = yfd.get_raw_json(_QUOTE_URL, params=params)
            results = (js or {}).get("quoteResponse", {}).get("result") or []
            for q in results:
                sym = q.get("symbol")
                if sym:
                    out[sym] = q
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[INFO] batch quote chunk failed ({exc})")
            if consecutive_failures >= 2:
                break
    return out


def _from_quote(q: dict) -> dict:
    price = q.get("regularMarketPrice")
    mc = q.get("marketCap")
    pe, source = np.nan, "unavailable"
    eps = q.get("epsTrailingTwelveMonths")
    if price and eps and eps > 0:
        pe, source = round(price / eps, 2), "price/trailingEps"
    elif (q.get("trailingPE") or 0) > 0:
        pe, source = round(float(q["trailingPE"]), 2), "trailingPE"
    elif price and (q.get("epsForward") or 0) > 0:
        pe, source = round(price / q["epsForward"], 2), "price/forwardEps (fwd)"
    elif (q.get("forwardPE") or 0) > 0:
        pe, source = round(float(q["forwardPE"]), 2), "forwardPE (fwd)"
    elif eps is not None and eps <= 0:
        source = "negative earnings"
    return {
        "P/E Ratio": pe, "P/E Source": source,
        "_market_cap": float(mc) if mc else np.nan,
        "Currency": q.get("currency") or "", "_price": price,
    }


def _fetch_single_fundamental(ticker: str, seed: Optional[dict] = None) -> Tuple[str, dict]:
    tk = _mk_ticker(ticker)
    seed = seed or {}
    price = seed.get("_price")
    market_cap_local = seed.get("_market_cap", np.nan)
    currency = seed.get("Currency") or None

    if not price:
        _LIMITER.acquire()
        try:
            fi = tk.fast_info
            p = _fi_get(fi, "last_price", "lastPrice", "regular_market_price")
            price = float(p) if p else None
            if pd.isna(market_cap_local):
                mc = _fi_get(fi, "market_cap", "marketCap")
                market_cap_local = float(mc) if mc else np.nan
            currency = currency or _fi_get(fi, "currency")
        except Exception as exc:
            print(f"[WARN] {ticker} fast_info failed: {exc}")

    info: dict = {}
    for attempt in range(INFO_RETRIES):
        _LIMITER.acquire()
        try:
            info = tk.info or {}
            if len(info) < 10:
                raise ValueError(f"incomplete ({len(info)} keys)")
            break
        except Exception as exc:
            info = {}
            if attempt == INFO_RETRIES - 1:
                print(f"[WARN] {ticker} .info unavailable: {exc}")
                break
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.5)

    if not price:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    if pd.isna(market_cap_local) and info.get("marketCap"):
        market_cap_local = float(info["marketCap"])
    currency = currency or info.get("currency")

    pe, source = seed.get("P/E Ratio", np.nan), seed.get("P/E Source", "unavailable")
    if pd.isna(pe):
        eps = info.get("trailingEps")
        if price and eps and eps > 0:
            pe, source = round(price / eps, 2), "price/trailingEps"
    if pd.isna(pe) and (info.get("trailingPE") or 0) > 0:
        pe, source = round(float(info["trailingPE"]), 2), "trailingPE"
    if pd.isna(pe):
        f_eps = info.get("forwardEps")
        if price and f_eps and f_eps > 0:
            pe, source = round(price / f_eps, 2), "price/forwardEps (fwd)"
    if pd.isna(pe) and (info.get("forwardPE") or 0) > 0:
        pe, source = round(float(info["forwardPE"]), 2), "forwardPE (fwd)"

    return ticker, {
        "P/E Ratio": pe, "P/E Source": source,
        "_market_cap": market_cap_local, "Currency": currency or "",
    }


def fx_to_usd(currencies: Set[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {"USD": 1.0}
    needed: Set[str] = set()
    for c in currencies:
        base, _ = _MINOR_UNITS.get(c, (c, 1.0))
        if base and base != "USD":
            needed.add(base)
    for base in list(needed):
        hit = _disk_get("fx", base, FX_TTL)
        if hit:
            rates[base] = float(hit)
            needed.discard(base)
    if needed:
        symbols = {f"{b}USD=X": b for b in needed}
        quotes = _batch_quote(list(symbols))
        for sym, base in symbols.items():
            rate = (quotes.get(sym) or {}).get("regularMarketPrice")
            if not rate:
                try:
                    _LIMITER.acquire()
                    rate = _fi_get(_mk_ticker(sym).fast_info, "last_price", "lastPrice")
                except Exception as exc:
                    print(f"[WARN] FX {sym} failed: {exc}")
                    rate = None
            if rate:
                rates[base] = float(rate)
                _disk_put("fx", base, float(rate))
    out: Dict[str, float] = {}
    for c in currencies:
        base, mult = _MINOR_UNITS.get(c, (c, 1.0))
        rate = rates.get(base)
        out[c] = rate * mult if rate else np.nan
    out["USD"] = 1.0
    out.setdefault("", np.nan)
    return out


def _cache_get_fundamental(ticker: str) -> Optional[dict]:
    hit = _FUNDAMENTAL_CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < FUNDAMENTAL_TTL:
        return hit[1]
    disk = _disk_get("fund2", ticker, FUNDAMENTAL_TTL)
    if disk is not None:
        _FUNDAMENTAL_CACHE[ticker] = (time.time(), disk)
        return disk
    return None


def _cache_put_fundamental(ticker: str, row: dict) -> None:
    _FUNDAMENTAL_CACHE[ticker] = (time.time(), row)
    _disk_put("fund2", ticker, row)


def fetch_fundamentals(tickers: List[str], force: bool = False) -> pd.DataFrame:
    out: Dict[str, dict] = {}
    todo: List[str] = []
    for t in tickers:
        hit = None if force else _cache_get_fundamental(t)
        if hit is not None:
            out[t] = hit
        else:
            todo.append(t)

    deep: Dict[str, dict] = {}
    if todo:
        quotes = _batch_quote(todo)
        for t in todo:
            q = quotes.get(t)
            if not q:
                deep[t] = {}
                continue
            row = _from_quote(q)
            unresolved = pd.isna(row["P/E Ratio"]) and row["P/E Source"] != "negative earnings"
            if unresolved or not row.get("_price"):
                deep[t] = row
            else:
                row.pop("_price", None)
                out[t] = row
                _cache_put_fundamental(t, row)

    if deep:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(deep))) as pool:
            futures = [pool.submit(_fetch_single_fundamental, t, seed) for t, seed in deep.items()]
            for fut in futures:
                try:
                    t, row = fut.result()
                except Exception as exc:
                    print(f"[WARN] deep fetch error: {exc}")
                    continue
                row.pop("_price", None)
                out[t] = row
                _cache_put_fundamental(t, row)

    for t in tickers:
        out.setdefault(t, {"P/E Ratio": np.nan, "P/E Source": "unavailable",
                           "_market_cap": np.nan, "Currency": ""})

    df = pd.DataFrame.from_dict(out, orient="index")
    for col, default in (("_market_cap", np.nan), ("Currency", "")):
        if col not in df.columns:
            df[col] = default

    ccys = {c for c in df["Currency"].fillna("").tolist() if c}
    fx = fx_to_usd(ccys) if ccys else {}
    mult = df["Currency"].map(lambda c: fx.get(c, 1.0 if c in ("", "USD") else np.nan))
    df["Market Cap ($B)"] = (pd.to_numeric(df["_market_cap"], errors="coerce")
                             * pd.to_numeric(mult, errors="coerce") / 1e9)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
_YF_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}


def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col not in raw.columns:
            return pd.DataFrame()
        return raw[[col]].rename(columns={col: tickers[0]})
    l0 = raw.columns.get_level_values(0).unique().tolist()
    l1 = raw.columns.get_level_values(1).unique().tolist()
    price = pd.DataFrame()
    if l0[0] in _YF_FIELDS:
        col = "Adj Close" if "Adj Close" in l0 else "Close"
        price = raw[col] if col in raw.columns.get_level_values(0) else pd.DataFrame()
    elif l1[0] in _YF_FIELDS:
        col = "Adj Close" if "Adj Close" in l1 else "Close"
        price = raw.xs(col, axis=1, level=1, drop_level=True)
    else:
        for level in (0, 1):
            for col in ("Adj Close", "Close"):
                try:
                    candidate = raw.xs(col, axis=1, level=level, drop_level=True)
                    if any(t in candidate.columns for t in tickers):
                        price = candidate
                        break
                except Exception:
                    continue
            else:
                continue
            break
    if isinstance(price, pd.Series):
        price = price.to_frame(name=tickers[0] if len(tickers) == 1 else "price")
    keep = [t for t in tickers if t in price.columns]
    return price[keep] if keep else pd.DataFrame()


def _download_multi(tickers: List[str], period: str) -> pd.DataFrame:
    base = dict(period=period, actions=False, progress=False)
    for kwargs in [{**base, "auto_adjust": False}, {**base, "auto_adjust": True}, base]:
        try:
            raw = yf.download(tickers, **kwargs)
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            print(f"[WARN] download attempt failed ({exc})")
    return pd.DataFrame()


def _download_per_ticker(tickers: List[str], period: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        try:
            raw = yf.download(t, period=period, actions=False, progress=False, auto_adjust=False)
            if raw is None or raw.empty:
                raw = yf.download(t, period=period, actions=False, progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                if col in raw.columns:
                    frames[t] = raw[col]
        except Exception as exc:
            print(f"[WARN] per-ticker fetch failed {t}: {exc}")
    return pd.DataFrame(frames) if frames else pd.DataFrame()


def fetch_price_data(tickers: List[str], period: str = "1y", force: bool = False) -> pd.DataFrame:
    cache_key = f"{period}|{','.join(sorted(tickers))}"
    if not force:
        cached = _disk_get("price", cache_key, PRICE_TTL)
        if cached is not None:
            return cached
    raw = _download_multi(tickers, period)
    price = _extract_close(raw, tickers) if not raw.empty else pd.DataFrame()
    if price.empty:
        price = _download_per_ticker(tickers, period)
    if price.empty:
        return pd.DataFrame()
    price = price.dropna(how="all")
    price = price.loc[:, price.notna().any()]
    if not price.empty:
        _disk_put("price", cache_key, price)
    return price


# ---------------------------------------------------------------------------
# Metrics (per-ticker, computed on each ticker's own sessions)
# ---------------------------------------------------------------------------
def _clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def daily_returns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for col in df.columns:
        s = _clean_series(df, col)
        if len(s) < 3:
            out[col] = pd.Series(dtype=float)
            continue
        out[col] = (s.pct_change(fill_method=None)
                    .replace([np.inf, -np.inf], np.nan).dropna())
    return out


def _max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    running_max = prices.cummax()
    return float((prices / running_max - 1.0).min() * 100)


def calculate_metrics(df: pd.DataFrame, fundamental_df: pd.DataFrame,
                      risk_free_rate: float = 0.04) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    rets = daily_returns(df)
    bench_rets = rets.get(BENCHMARK_TICKER, pd.Series(dtype=float))
    bench_px = _clean_series(df, BENCHMARK_TICKER) if BENCHMARK_TICKER in df.columns else pd.Series(dtype=float)
    if not bench_px.empty:
        session_count = len(bench_px)
    else:
        session_count = max((len(_clean_series(df, c)) for c in df.columns), default=1)
    session_count = max(session_count, 1)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    window = min(50, len(df))

    metrics: Dict[str, dict] = {}
    for col in df.columns:
        s = _clean_series(df, col)
        if s.empty:
            continue
        r = rets.get(col, pd.Series(dtype=float))
        price = float(s.iloc[-1])
        total_return = (price / float(s.iloc[0]) - 1.0) * 100
        if len(r) > 2 and r.std() > 0:
            sd = float(r.std())
            volatility = sd * np.sqrt(TRADING_DAYS) * 100
            sharpe = float((r - rf_daily).mean() / sd * np.sqrt(TRADING_DAYS))
        else:
            volatility, sharpe = np.nan, np.nan
        sma = float(s.rolling(window=min(window, len(s)), min_periods=1).mean().iloc[-1])
        sma_diff = ((price - sma) / sma * 100) if sma else np.nan

        beta, excess = np.nan, np.nan
        if col == BENCHMARK_TICKER:
            beta, excess = 1.0, 0.0
        else:
            if not bench_rets.empty and len(r) > 2:
                joint = pd.concat([r.rename("a"), bench_rets.rename("b")], axis=1).dropna()
                if len(joint) >= 30:
                    var_b = float(joint["b"].var())
                    if var_b > 1e-12:
                        beta = float(joint["a"].cov(joint["b"]) / var_b)
            if not bench_px.empty:
                common = s.index.intersection(bench_px.index)
                if len(common) >= 2:
                    a0, a1 = float(s.loc[common[0]]), float(s.loc[common[-1]])
                    b0, b1 = float(bench_px.loc[common[0]]), float(bench_px.loc[common[-1]])
                    if a0 > 0 and b0 > 0:
                        excess = ((a1 / a0) - (b1 / b0)) * 100

        metrics[col] = {
            "Current Price": price,
            "Total Return (%)": total_return,
            "Excess vs Bench (pp)": excess,
            "Volatility (Ann %)": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": _max_drawdown(s),
            "Beta": beta,
            "SMA Deviation (%)": sma_diff,
            "P/E Ratio": (
                fundamental_df.loc[col, "P/E Ratio"]
                if (not fundamental_df.empty and col in fundamental_df.index
                    and "P/E Ratio" in fundamental_df.columns) else np.nan
            ),
            "Coverage (%)": min(100.0, len(s) / session_count * 100),
        }
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(metrics, orient="index").round(2)
