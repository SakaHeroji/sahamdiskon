"""
Stock Valuation Analyzer v3
============================
4 valuation methods, probability of upside, single clean table.
"""

import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import sys
import time
import math

# ════════════════════════════════════════════════════════════════
# Data Model
# ════════════════════════════════════════════════════════════════

@dataclass
class StockResult:
    ticker: str
    price: Optional[float] = None
    # Valuasi per metode
    graham: Optional[float] = None
    epv: Optional[float] = None
    dcf: Optional[float] = None
    rim: Optional[float] = None
    # Blended
    blended_iv: Optional[float] = None
    intrinsic: Optional[float] = None
    mos_pct: Optional[float] = None       # Margin of Safety %
    score: float = 0.0                     # Value Score (0-100)
    prob_up_pct: float = 0.0              # Probabilitas naik % (sigmoid model)
    confidence: str = "D"                  # Data confidence: A/B/C/D
    data_quality: str = "D"               # Completeness quality tag
    status: str = "N/A"
    # Raw metrics (used for scoring)
    pe: Optional[float] = None
    pbv: Optional[float] = None
    roe: Optional[float] = None
    de: Optional[float] = None
    growth: Optional[float] = None
    div_yield: Optional[float] = None
    analyst_target: Optional[float] = None
    methods_agree: int = 0                 # Jumlah metode yg bilang undervalued
    sector_template: Optional[str] = None   # Template sektor untuk DCF
    wacc_used: Optional[float] = None       # WACC yang dipakai (% display)
    terminal_g_used: Optional[float] = None # Terminal growth yang dipakai (% display)
    error: Optional[str] = None


# ════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════

class ValueAnalyzer:
    RETRY = 2
    WORKERS = 8
    # Indonesia market parameters
    RISK_FREE    = 0.065   # BI rate / SBN yield ~6.5%
    EQUITY_PREM  = 0.055   # Indonesia equity risk premium
    DISCOUNT     = 0.12    # RISK_FREE + EQUITY_PREM = 12% (fallback)
    TERM_GROWTH  = 0.05    # Indonesia long-term GDP growth ~5% (fallback)
    AAA_YIELD    = 7.0     # Indonesia AAA corporate bond yield %
    TAX_RATE     = 0.22    # Indonesia corporate tax rate
    # WACC bounds
    WACC_MIN     = 0.085   # 8.5% floor (Rf + minimal premium)
    WACC_MAX     = 0.18    # 18% ceiling
    # Terminal growth bounds
    TG_MIN       = 0.02    # 2% floor (below GDP)
    TG_MAX       = 0.06    # 6% ceiling (above long-run GDP)
    # Active valuation regime
    VALUATION_REGIME = "conservative"

    # Conservative defaults (active params)
    CAL_GRAHAM   = 0.92
    CAL_EPV      = 0.90
    CAL_DCF      = 0.82
    CAL_RIM      = 0.88
    EPV_BUFFER   = 0.85
    DCF_G1_CAP   = 0.18
    DCF_TG_CAP   = 0.04
    DCF_TERMINAL_HAIRCUT = 0.85
    DCF_MIN_TERMINAL_SPREAD = 0.03
    RIM_TERMINAL_DECAY = 0.35
    DCF_TO_EPV_CAP = 6.0
    DCF_TO_GRAHAM_CAP = 4.5
    BLEND_WEIGHTS = {
        "graham": 0.25,
        "dcf": 0.20,
        "rim": 0.25,
        "epv": 0.30,
    }

    REGIME_PRESETS = {
        "aggressive": {
            "CAL_GRAHAM": 0.98,
            "CAL_EPV": 0.98,
            "CAL_DCF": 0.95,
            "CAL_RIM": 0.96,
            "EPV_BUFFER": 0.92,
            "DCF_G1_CAP": 0.25,
            "DCF_TG_CAP": 0.05,
            "DCF_TERMINAL_HAIRCUT": 0.95,
            "DCF_MIN_TERMINAL_SPREAD": 0.02,
            "RIM_TERMINAL_DECAY": 0.50,
            "DCF_TO_EPV_CAP": 8.0,
            "DCF_TO_GRAHAM_CAP": 6.0,
            "BLEND_WEIGHTS": {"graham": 0.30, "dcf": 0.30, "rim": 0.25, "epv": 0.15},
        },
        "balanced": {
            "CAL_GRAHAM": 0.95,
            "CAL_EPV": 0.94,
            "CAL_DCF": 0.90,
            "CAL_RIM": 0.93,
            "EPV_BUFFER": 0.88,
            "DCF_G1_CAP": 0.21,
            "DCF_TG_CAP": 0.045,
            "DCF_TERMINAL_HAIRCUT": 0.90,
            "DCF_MIN_TERMINAL_SPREAD": 0.025,
            "RIM_TERMINAL_DECAY": 0.42,
            "DCF_TO_EPV_CAP": 7.0,
            "DCF_TO_GRAHAM_CAP": 5.0,
            "BLEND_WEIGHTS": {"graham": 0.27, "dcf": 0.24, "rim": 0.25, "epv": 0.24},
        },
        "conservative": {
            "CAL_GRAHAM": 0.92,
            "CAL_EPV": 0.90,
            "CAL_DCF": 0.82,
            "CAL_RIM": 0.88,
            "EPV_BUFFER": 0.85,
            "DCF_G1_CAP": 0.18,
            "DCF_TG_CAP": 0.04,
            "DCF_TERMINAL_HAIRCUT": 0.85,
            "DCF_MIN_TERMINAL_SPREAD": 0.03,
            "RIM_TERMINAL_DECAY": 0.35,
            "DCF_TO_EPV_CAP": 6.0,
            "DCF_TO_GRAHAM_CAP": 4.5,
            "BLEND_WEIGHTS": {"graham": 0.25, "dcf": 0.20, "rim": 0.25, "epv": 0.30},
        },
    }

    # DCF base-case per sektor (sumber: DCF_Template_Sektor_Indonesia.xlsx)
    # Semua angka dalam desimal (contoh 10.5% -> 0.105)
    SECTOR_DCF_BASE: Dict[str, Dict[str, object]] = {
        "Consumer Staples": {
            "wacc": 0.105,
            "growth_curve": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05, 0.045],
            "terminal_g": 0.045,
        },
        "Consumer Disc.": {
            "wacc": 0.11,
            "growth_curve": [0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05],
            "terminal_g": 0.05,
        },
        "Telco & Tower": {
            "wacc": 0.10,
            "growth_curve": [0.09, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.045, 0.04],
            "terminal_g": 0.04,
        },
        "Perbankan & Fintech": {
            "wacc": 0.12,
            "growth_curve": [0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
            "terminal_g": 0.045,
        },
        "Properti & Infra": {
            "wacc": 0.11,
            "growth_curve": [0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.055, 0.05, 0.045],
            "terminal_g": 0.035,
        },
        "Komoditas – Coal": {
            "wacc": 0.115,
            "growth_curve": [0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "terminal_g": 0.02,
        },
        "Komoditas – CPO": {
            "wacc": 0.115,
            "growth_curve": [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.035, 0.035, 0.03],
            "terminal_g": 0.025,
        },
        "Healthcare & Farmasi": {
            "wacc": 0.105,
            "growth_curve": [0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.065, 0.06, 0.055],
            "terminal_g": 0.04,
        },
        "Energi & Utilitas": {
            "wacc": 0.11,
            "growth_curve": [0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.035],
            "terminal_g": 0.03,
        },
        "Teknologi & Digital": {
            "wacc": 0.13,
            "growth_curve": [0.25, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05],
            "terminal_g": 0.04,
        },
    }

    # Mapping keyword dari sector/industry Yahoo ke template sektor DCF
    SECTOR_KEYWORDS: List[Tuple[str, str]] = [
        ("Consumer Staples", "consumer defensive"),
        ("Consumer Staples", "staples"),
        ("Consumer Staples", "food"),
        ("Consumer Staples", "beverage"),
        ("Consumer Disc.", "consumer cyclical"),
        ("Consumer Disc.", "retail"),
        ("Consumer Disc.", "apparel"),
        ("Telco & Tower", "communication services"),
        ("Telco & Tower", "telecom"),
        ("Telco & Tower", "tower"),
        ("Perbankan & Fintech", "financial services"),
        ("Perbankan & Fintech", "banks"),
        ("Perbankan & Fintech", "bank"),
        ("Perbankan & Fintech", "insurance"),
        ("Perbankan & Fintech", "fintech"),
        ("Properti & Infra", "real estate"),
        ("Properti & Infra", "property"),
        ("Properti & Infra", "infrastructure"),
        ("Properti & Infra", "engineering"),
        ("Properti & Infra", "construction"),
        ("Komoditas – Coal", "coal"),
        ("Komoditas – Coal", "mining"),
        ("Komoditas – Coal", "metals"),
        ("Komoditas – Coal", "nickel"),
        ("Komoditas – Coal", "gold"),
        ("Komoditas – Coal", "copper"),
        ("Komoditas – CPO", "palm"),
        ("Komoditas – CPO", "plantation"),
        ("Komoditas – CPO", "agricultural"),
        ("Healthcare & Farmasi", "healthcare"),
        ("Healthcare & Farmasi", "pharmaceutical"),
        ("Energi & Utilitas", "energy"),
        ("Energi & Utilitas", "oil"),
        ("Energi & Utilitas", "gas"),
        ("Energi & Utilitas", "utility"),
        ("Teknologi & Digital", "technology"),
        ("Teknologi & Digital", "software"),
        ("Teknologi & Digital", "internet"),
        ("Teknologi & Digital", "digital"),
        ("Teknologi & Digital", "ecommerce"),
    ]

    @classmethod
    def set_regime(cls, regime: str) -> None:
        key = (regime or "").strip().lower()
        if key not in cls.REGIME_PRESETS:
            supported = ", ".join(cls.REGIME_PRESETS.keys())
            raise ValueError(f"Invalid valuation regime '{regime}'. Supported: {supported}")

        preset = cls.REGIME_PRESETS[key]
        cls.VALUATION_REGIME = key
        cls.CAL_GRAHAM = preset["CAL_GRAHAM"]
        cls.CAL_EPV = preset["CAL_EPV"]
        cls.CAL_DCF = preset["CAL_DCF"]
        cls.CAL_RIM = preset["CAL_RIM"]
        cls.EPV_BUFFER = preset["EPV_BUFFER"]
        cls.DCF_G1_CAP = preset["DCF_G1_CAP"]
        cls.DCF_TG_CAP = preset["DCF_TG_CAP"]
        cls.DCF_TERMINAL_HAIRCUT = preset["DCF_TERMINAL_HAIRCUT"]
        cls.DCF_MIN_TERMINAL_SPREAD = preset["DCF_MIN_TERMINAL_SPREAD"]
        cls.RIM_TERMINAL_DECAY = preset["RIM_TERMINAL_DECAY"]
        cls.DCF_TO_EPV_CAP = preset["DCF_TO_EPV_CAP"]
        cls.DCF_TO_GRAHAM_CAP = preset["DCF_TO_GRAHAM_CAP"]
        cls.BLEND_WEIGHTS = dict(preset["BLEND_WEIGHTS"])

    @staticmethod
    def _g(info, key):
        """Safely extract numeric value, returns None for any bad data."""
        v = info.get(key)
        if v is None or v == "" or v == "None":
            return None
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_series(df: Optional[pd.DataFrame], candidates: List[str]) -> dict:
        if df is None or df.empty:
            return {}
        row_name = next((name for name in candidates if name in df.index), None)
        if row_name is None:
            return {}

        result = {}
        row = df.loc[row_name]
        for col, val in row.items():
            try:
                year = pd.Timestamp(col).year
            except Exception:
                text = str(col)
                if len(text) < 4 or not text[:4].isdigit():
                    continue
                year = int(text[:4])
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                result[year] = num
        return result

    @staticmethod
    def _latest_value(series_map: dict) -> Optional[float]:
        if not series_map:
            return None
        year = max(series_map.keys())
        return series_map.get(year)

    @staticmethod
    def _cagr(series_map: dict, periods: int = 3) -> Optional[float]:
        if not series_map:
            return None
        years = sorted(series_map.keys(), reverse=True)
        if len(years) < periods + 1:
            return None

        newest_year = years[0]
        oldest_year = years[periods]
        newest = series_map.get(newest_year)
        oldest = series_map.get(oldest_year)

        if None in (newest, oldest) or newest <= 0 or oldest <= 0:
            return None

        year_gap = newest_year - oldest_year
        if year_gap <= 0:
            return None
        return (newest / oldest) ** (1 / year_gap) - 1

    @staticmethod
    def _c(val):
        """Clean any value: convert nan/inf to None. Keep full precision."""
        if val is None:
            return None
        try:
            f = float(val)
            if not math.isfinite(f):
                return None
            # Round appropriately: integers for large values, 2dp for small
            if abs(f) >= 100:
                return round(f, 0)
            elif abs(f) >= 1:
                return round(f, 2)
            else:
                return round(f, 4)
        except (ValueError, TypeError):
            return None

    # ══════════════════════════════════════════════════════
    # 4 VALUATION METHODS (IDX-optimized)
    # ══════════════════════════════════════════════════════

    @classmethod
    def v_graham(cls, eps, bv, growth_pct=None, aaa_yield=7.0):
        """
        Graham Growth Formula (Benjamin Graham, revised):
        V = EPS × (8.5 + 2g) × 4.4 / Y

        - 8.5  = base PE for zero-growth company
        - g    = expected 5yr earnings growth rate (%)
        - 4.4  = Graham's assumed AAA bond yield at his time
        - Y    = current AAA bond yield (%)

        If growth data unavailable, fallback to classic:
        V = √(22.5 × EPS × BV)
        """
        if eps is None or eps <= 0 or aaa_yield is None or aaa_yield <= 0:
            return None
        if growth_pct is not None and growth_pct > -5:
            g = max(0, min(growth_pct, cls.DCF_G1_CAP * 100))
            return eps * (8.5 + 2 * g) * (4.4 / aaa_yield) * cls.CAL_GRAHAM
        # Fallback: classic Graham Number
        if bv is not None and bv > 0:
            return math.sqrt(22.5 * eps * bv) * cls.CAL_GRAHAM
        return None

    @classmethod
    def v_epv(cls, eps, r=0.12):
        """
        Earnings Power Value = Normalized Owner Earnings / Cost of Equity

        Conservative floor valuation (assumes zero growth).
        Applied 0.9× margin of safety buffer since zero-growth
        assumption may still be optimistic for some companies.
        """
        if eps is None or eps <= 0:
            return None
        return (eps / r) * cls.EPV_BUFFER * cls.CAL_EPV

    @classmethod
    def v_dcf(cls, base_cf_ps, roe=None, growth_rate=None, payout=0.3,
              r=0.12, tg=0.05, yrs_high=5, yrs_stable=5,
              growth_curve: Optional[List[float]] = None):
        """
        Two-Stage DCF:
          Stage 1 (5yr): High growth at sustainable rate
            Stage 2 (5yr): Growth decays gradually to terminal rate
          Terminal: Gordon Growth Model

        Sustainable growth = ROE × retention ratio, capped by actual earnings growth.
        Using min(g_roe, g_actual) prevents overestimate when ROE is structurally
        high but actual growth momentum is lower (e.g., mature banks, cyclicals).
        """
        if base_cf_ps is None or base_cf_ps <= 0:
            return None

        # Estimate growth rate (priority order)
        if roe is not None and roe > 0:
            retention = 1.0 - payout
            g_roe = (roe / 100) * retention        # ROE × (1 - payout)
            if growth_rate is not None and growth_rate > 0:
                # Conservative: use min of ROE-implied and actual growth
                g_sustainable = min(g_roe, growth_rate)
            else:
                g_sustainable = g_roe
        elif growth_rate is not None:
            g_sustainable = growth_rate
        else:
            g_sustainable = 0.05  # default 5%

        tg_eff = min(tg, cls.DCF_TG_CAP)

        pv = 0.0
        e = base_cf_ps

        # Jika growth_curve sektor tersedia (Y1..Y10), gunakan langsung.
        if growth_curve:
            for t, g_t in enumerate(growth_curve, 1):
                g_eff = max(-0.10, min(float(g_t), 0.35))
                e *= (1 + g_eff)
                pv += e / ((1 + r) ** t)
            total_yrs = len(growth_curve)
            # Template sektor bisa punya terminal g > cap konservatif global
            tg_eff = max(0.0, min(tg, 0.06))
        else:
            # Clamp growth to reasonable range
            g1 = max(-0.03, min(g_sustainable, cls.DCF_G1_CAP))

            # Stage 1: high growth
            for t in range(1, yrs_high + 1):
                e *= (1 + g1)
                pv += e / ((1 + r) ** t)

            # Stage 2: declining growth (linear glide path from g1 -> tg)
            for step in range(1, yrs_stable + 1):
                t = yrs_high + step
                blend_ratio = step / yrs_stable
                g_t = g1 + (tg_eff - g1) * blend_ratio
                e *= (1 + g_t)
                pv += e / ((1 + r) ** t)

            total_yrs = yrs_high + yrs_stable

        # Terminal value
        terminal_spread = r - tg_eff
        if terminal_spread >= cls.DCF_MIN_TERMINAL_SPREAD:
            tv = e * (1 + tg_eff) / terminal_spread
            tv *= cls.DCF_TERMINAL_HAIRCUT
            pv += tv / ((1 + r) ** total_yrs)

        return pv * cls.CAL_DCF

    @classmethod
    def v_rim(cls, bv, roe, payout=0.3, r=0.12, tg=0.05, yrs=10):
        """
        Residual Income Model (RIM):
        V = Book Value + Σ (Excess Earnings / (1+r)^t)

        Excess Earnings = (ROE - r) × Book Value per share

        Book value rolls forward each year:
        BV(t+1) = BV(t) + NI(t) - DIV(t)

        More accurate than PEG because:
        - Anchored in tangible book value
        - Measures VALUE CREATION (ROE vs cost of equity)
        - Works for both growth and value stocks
        - A company is worth more than book only if ROE > r
        """
        if None in (bv, roe) or bv <= 0:
            return None
        if payout is None or payout < 0 or payout >= 1:
            payout = 0.3
        roe_dec = roe / 100  # convert % to decimal

        if roe_dec <= 0:
            return bv * 0.7  # negative ROE → discount to book

        # Roll-forward residual income with gradual fade of excess ROE
        bv_t = bv
        excess_roe0 = roe_dec - r

        # PV of residual income stream (finite horizon)
        pv_excess = 0.0
        for t in range(1, yrs + 1):
            fade = max(0.0, 1.0 - (t - 1) / max(yrs, 1))
            roe_t = r + excess_roe0 * fade

            ni_t = roe_t * bv_t
            div_t = ni_t * payout
            excess_t = ni_t - (r * bv_t)

            pv_excess += excess_t / ((1 + r) ** t)
            bv_t = bv_t + ni_t - div_t

        # Terminal residual income on ending BV, conservatively decayed
        terminal_spread = r - tg
        if terminal_spread >= 0.02:
            final_fade = 1.0 / max(yrs, 1)
            roe_terminal = r + excess_roe0 * final_fade * cls.RIM_TERMINAL_DECAY
            terminal_excess = (roe_terminal - r) * bv_t / terminal_spread
            pv_excess += terminal_excess / ((1 + r) ** yrs)

        return (bv + pv_excess) * cls.CAL_RIM

    # ── Blended Intrinsic ─────────────────────────────────

    @classmethod
    def blend(cls, graham, epv, dcf, rim, method_quality: Optional[dict] = None):
        """
                Dynamic weighted blend based on reliability:
                    Graham Growth  25%
                    DCF (2-stage)  20%
                    RIM            25%
                    EPV            30% (conservative floor gets higher weight)

        Weights re-normalize if some methods return None.
        Optional `method_quality` applies multipliers per method:
        keys: graham/epv/dcf/rim with range ~0.5 - 1.2
        """
        quality = method_quality or {}
        weights = cls.BLEND_WEIGHTS
        m = []
        if graham is not None and graham > 0:
            w = weights["graham"] * max(0.5, min(1.2, quality.get("graham", 1.0)))
            m.append((graham, w))
        if dcf is not None and dcf > 0:
            w = weights["dcf"] * max(0.5, min(1.2, quality.get("dcf", 1.0)))
            m.append((dcf, w))
        if rim is not None and rim > 0:
            w = weights["rim"] * max(0.5, min(1.2, quality.get("rim", 1.0)))
            m.append((rim, w))
        if epv is not None and epv > 0:
            w = weights["epv"] * max(0.5, min(1.2, quality.get("epv", 1.0)))
            m.append((epv, w))
        if not m:
            return None
        tw = sum(w for _, w in m)
        return sum(v * w for v, w in m) / tw

    @staticmethod
    def mos(price, iv):
        if None in (price, iv) or iv <= 0:
            return None
        return round(((iv - price) / iv) * 100, 2)

    @staticmethod
    def calc_roe(ni, bv, shares):
        if None in (ni, bv, shares):
            return None
        eq = bv * shares
        return round((ni / eq) * 100, 2) if eq > 0 else None

    # ── Value Score (0-100) ───────────────────────────────

    @staticmethod
    def calc_score(pe, pbv, roe, de, mos_val, dy):
        s = 0.0
        # PE (0-20)
        if pe is not None and pe > 0:
            s += max(0, min(20, 20 - (pe - 5) * 0.8))
        # PBV (0-20)
        if pbv is not None and pbv > 0:
            s += max(0, min(20, 20 - (pbv - 0.5) * 5))
        # ROE (0-15)
        if roe is not None and roe > 0:
            s += min(15, roe * 0.75)
        # D/E (0-15)
        if de is not None and de >= 0:
            s += max(0, min(15, 15 - de * 0.075))
        # MoS (0-20)
        if mos_val is not None and mos_val > 0:
            s += min(20, mos_val * 0.5)
        # Dividend (0-10)
        if dy is not None and dy > 0:
            dp = dy if dy < 1 else dy / 100
            s += min(10, dp * 200)
        return round(min(s, 100), 2)

    # ══════════════════════════════════════════════════════
    # PROBABILITY MODEL (Sigmoid / Logistic)
    # ══════════════════════════════════════════════════════

    @staticmethod
    def calc_probability(pe, pbv, roe, de, mos_val, methods_agree,
                         growth, analyst_up_pct,
                         confidence: str = "D", data_quality: str = "D"):
        """
        Sigmoid-based probability model for 12-month upside.

        Theoretical foundation:
        - Fama-French: Value factor (low PE, low PBV) ~4-5% annual premium
        - Novy-Marx: Profitability factor (high ROE) ~3-4% premium
        - Mean-reversion: Stocks with large MoS tend to revert to intrinsic
        - Multi-model consensus increases prediction reliability

        Model:
          1. Compute 5 normalized factor scores (0 to 1)
          2. Weighted composite score S
          3. P(up) = P_min + (P_max - P_min) × sigmoid(k × (S - 0.5))

        Range:
          - P_min = 22% (worst case — even bad stocks sometimes recover)
          - P_max = 78% (best case — even great stocks can fall)
          - This is REALISTIC. No stock has 90%+ probability.

        Why sigmoid:
          - Natural asymptotic bounds (can't exceed P_min/P_max)
          - Diminishing returns at extremes
          - Calibrated to academic factor premium data
        """
        # ── 1) Normalize each factor to [0, 1] ──

        # Value Factor: low PE & low PBV = higher score
        f_pe = 0.5  # neutral if no data
        if pe is not None and pe > 0:
            f_pe = max(0, min(1, (20 - pe) / 15))  # PE 5→1.0, PE 20→0.0

        f_pbv = 0.5
        if pbv is not None and pbv > 0:
            f_pbv = max(0, min(1, (3 - pbv) / 2.5))  # PBV 0.5→1.0, PBV 3→0.0

        f_mos = 0.3  # slightly below neutral if no data
        if mos_val is not None:
            f_mos = max(0, min(1, (mos_val + 20) / 70))  # MoS -20%→0, +50%→1

        VALUE = 0.35 * f_pe + 0.30 * f_pbv + 0.35 * f_mos

        # Quality Factor: high ROE, low Debt = sustainable value
        f_roe = 0.3
        if roe is not None and roe > 0:
            f_roe = max(0, min(1, roe / 25))  # ROE 0→0, 25+→1

        f_de = 0.5
        if de is not None and de >= 0:
            f_de = max(0, min(1, (150 - de) / 150))  # D/E 0→1, 150+→0

        QUALITY = 0.60 * f_roe + 0.40 * f_de

        # Growth Momentum
        f_growth = 0.3
        if growth is not None:
            g_pct = growth * 100 if abs(growth) < 1 else growth
            f_growth = max(0, min(1, (g_pct + 5) / 35))  # -5%→0, 30%→1

        GROWTH = f_growth

        # Consensus: how many independent methods agree stock is undervalued
        CONSENSUS = methods_agree / 4  # 0/4→0 ... 4/4→1

        # Analyst confirmation (weak signal)
        f_analyst = 0.4
        if analyst_up_pct is not None:
            f_analyst = max(0, min(1, (analyst_up_pct + 10) / 40))  # -10%→0, 30%→1

        ANALYST = f_analyst

        # ── 2) Weighted composite score ──
        S = (0.30 * VALUE +
             0.25 * QUALITY +
             0.20 * GROWTH +
             0.15 * CONSENSUS +
             0.10 * ANALYST)

        # Reliability calibration from confidence and data quality
        quality_scale = {
            "A": 1.00,
            "B": 0.85,
            "C": 0.70,
            "D": 0.55,
        }
        conf_scale = quality_scale.get((confidence or "D").upper(), 0.55)
        dq_scale = quality_scale.get((data_quality or "D").upper(), 0.55)
        reliability = max(0.50, min(1.00, (conf_scale + dq_scale) / 2))

        # Pull toward neutral score when reliability is low
        S = 0.5 + (S - 0.5) * reliability

        # ── 3) Sigmoid mapping to probability ──
        # k scales with reliability: low quality = flatter curve
        k = 5.0
        k_eff = k * reliability
        P_MIN = 0.22   # even terrible stocks sometimes recover
        P_MAX = 0.78   # even great stocks can fall

        logit = k_eff * (S - 0.5)
        sigmoid = 1.0 / (1.0 + math.exp(-logit))
        prob = P_MIN + (P_MAX - P_MIN) * sigmoid

        return round(prob * 100, 1)

    # ── Data Confidence Rating ────────────────────────────

    @staticmethod
    def calc_confidence(eps, bv, pe, pbv, roe, de, growth, analyst):
        """
        Data completeness → confidence rating.
        More data = more reliable probability estimate.

        A = 7-8 metrics (high confidence)
        B = 5-6 metrics (moderate)
        C = 3-4 metrics (low)
        D = 0-2 metrics (unreliable — probability is mostly base rate)
        """
        available = sum(1 for v in [eps, bv, pe, pbv, roe, de, growth, analyst]
                        if v is not None)
        if available >= 7:
            return "A"
        elif available >= 5:
            return "B"
        elif available >= 3:
            return "C"
        else:
            return "D"

    # ── Status (4 Levels) ─────────────────────────────────

    @staticmethod
    def get_status(mos_val, score):
        m = mos_val if mos_val is not None else -999
        if m > 25 and score >= 55:
            return "🟢 SANGAT MURAH"
        elif m > 5 and score >= 35:
            return "🟡 MURAH"
        elif m > -10:
            return "🟠 WAJAR"
        else:
            return "🔴 MAHAL"

    # ── Analyze Single Stock ──────────────────────────────

    # Sektor yang dianggap sebagai perusahaan keuangan/bank
    FINANCIAL_SECTORS = {
        "financial services", "banks", "bank", "diversified financials",
        "insurance", "financialservices", "financial",
    }

    @classmethod
    def _is_bank(cls, info: dict) -> bool:
        """Deteksi apakah saham adalah bank/lembaga keuangan.
        Bank memiliki OCF/FCF yang tidak representatif untuk DCF standar.
        """
        sector = (info.get("sector") or "").lower().strip()
        industry = (info.get("industry") or "").lower().strip()
        for kw in cls.FINANCIAL_SECTORS:
            if kw in sector or kw in industry:
                return True
        return False

    @classmethod
    def _resolve_sector_dcf(cls, info: dict) -> Tuple[Optional[str], Optional[dict]]:
        """Map sector/industry Yahoo Finance ke template DCF sektor Indonesia."""
        sector = (info.get("sector") or "").lower().strip()
        industry = (info.get("industry") or "").lower().strip()
        text = f"{sector} | {industry}"

        # Prioritas agar "Software - Infrastructure" tidak salah masuk Properti & Infra
        if any(k in text for k in ["technology", "software", "internet", "digital", "ecommerce"]):
            template = "Teknologi & Digital"
            return template, cls.SECTOR_DCF_BASE.get(template)

        for template, keyword in cls.SECTOR_KEYWORDS:
            if keyword in text:
                return template, cls.SECTOR_DCF_BASE.get(template)

        # Fallback berdasarkan broad sector Yahoo
        if "utilities" in sector:
            template = "Energi & Utilitas"
        elif "industrials" in sector:
            template = "Properti & Infra"
        elif "basic materials" in sector:
            template = "Komoditas – Coal"
        else:
            return None, None

        return template, cls.SECTOR_DCF_BASE.get(template)

    @classmethod
    def _calc_stock_wacc(
        cls,
        info: dict,
        beta: Optional[float],
        de_ratio: Optional[float],          # D/E dari Yahoo (total debt / equity)
        income_stmt: Optional[pd.DataFrame] = None,
        balance_sheet: Optional[pd.DataFrame] = None,
        sector_wacc: Optional[float] = None,  # batas referensi dari sektor
    ) -> float:
        """
        Hitung WACC per saham menggunakan CAPM + struktur modal aktual.

        WACC = Ke × We + Kd × (1 - t) × Wd

        Ke (cost of equity via CAPM):
            Ke = Rf + β × ERP
            - β  = beta dari Yahoo Finance (default 1.0 jika tidak ada)
            - Rf = BI rate / SBN 10Y ≈ 6.5%
            - ERP= Indonesia equity risk premium ≈ 5.5%

        Kd (cost of debt):
            Prioritas: interest expense / total debt dari laporan keuangan
            Fallback  : AAA yield Indonesia ~7% + spread berdasarkan D/E

        We / Wd dari D/E:
            D/E = total debt / total equity  →  Wd = D/(D+E),  We = E/(D+E)
        """
        # ── Ke via CAPM ──────────────────────────────────────────
        beta_eff = beta if (beta is not None and 0.3 <= beta <= 3.5) else 1.0
        ke = cls.RISK_FREE + beta_eff * cls.EQUITY_PREM
        ke = max(0.08, min(ke, 0.25))  # clamp Ke 8%-25%

        # ── Kd dari laporan keuangan ────────────────────────────
        kd_market = None
        try:
            # Coba ambil interest expense dari income statement
            ie_series = cls._extract_series(
                income_stmt,
                ["Interest Expense", "Interest Expense Non Operating",
                 "Net Interest Income", "Interest And Debt Expense"]
            )
            ie_val = cls._latest_value(ie_series)
            if ie_val is not None:
                ie_abs = abs(ie_val)
                # Total debt dari balance sheet
                td_series = cls._extract_series(
                    balance_sheet,
                    ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
                )
                td_val = cls._latest_value(td_series)
                if td_val is not None and td_val > 0:
                    kd_market = ie_abs / td_val
        except Exception:
            kd_market = None

        # Fallback Kd: AAA yield + spread dari D/E
        if kd_market is None or not (0.03 <= kd_market <= 0.25):
            de_safe = de_ratio if (de_ratio is not None and de_ratio >= 0) else 0.0
            # Spread: 0bp (de=0) s/d ~250bp (de=200+)
            spread = min(0.025, de_safe / 200 * 0.025)
            kd_market = cls.RISK_FREE + spread + 0.005  # Rf + spread + 50bp default
            kd_market = max(0.055, min(kd_market, 0.15))

        kd_after_tax = kd_market * (1 - cls.TAX_RATE)

        # ── Bobot ekuitas / hutang ──────────────────────────────
        de_safe = de_ratio if (de_ratio is not None and de_ratio >= 0) else 0.0
        # de_ratio Yahoo = total debt / total equity  (%)
        # Konversi ke desimal
        de_dec = de_safe / 100.0
        if de_dec < 0:
            de_dec = 0.0
        wd = de_dec / (1.0 + de_dec)   # D/(D+E)
        we = 1.0 - wd                   # E/(D+E)

        wacc = ke * we + kd_after_tax * wd

        # Clamp absolut
        wacc = max(cls.WACC_MIN, min(wacc, cls.WACC_MAX))

        # Soft anchor ke sektor: jika ada referensi sektor,
        # blend 60% stock / 40% sektor agar tidak terlalu jauh
        if sector_wacc is not None:
            wacc = 0.60 * wacc + 0.40 * sector_wacc
            wacc = max(cls.WACC_MIN, min(wacc, cls.WACC_MAX))

        return round(wacc, 4)

    @classmethod
    def _calc_stock_terminal_g(
        cls,
        roe: Optional[float],              # ROE % (misal 15.0)
        best_growth: Optional[float],      # pertumbuhan historis (desimal)
        de_ratio: Optional[float],         # D/E
        payout: float = 0.3,
        sector_tg: Optional[float] = None, # batas referensi terminal g sektor
    ) -> float:
        """
        Hitung terminal growth per saham.

        Pendekatan fundamentalis:
            g_sustainable = ROE × (1 - payout)  → batas atas pertumbuhan organik
            g_historical  = CAGR laba historis    → momentum aktual
            g_base        = min(g_sustainable, g_historical) — pilih yg konservatif

        Kemudian:
            - Haircut 50% karena pertumbuhan jangka panjang jauh lebih rendah
            - Tambah penalty jika D/E tinggi (leveraged company tumbuh lebih lamban)
            - Clamp ke [TG_MIN, TG_MAX]
            - Jika sektor punya referensi tg, blend 60% stock / 40% sektor
        """
        # Kandidat 1: sustainable growth dari ROE
        g_roe = None
        if roe is not None and roe > 0:
            retention = 1.0 - max(0.0, min(payout, 0.99))
            g_roe = (roe / 100.0) * retention  # desimal

        # Kandidat 2: pertumbuhan historis (sudah desimal)
        g_hist = None
        if best_growth is not None:
            raw = best_growth if abs(best_growth) < 1 else best_growth / 100
            if raw > 0:
                g_hist = raw

        # Pilih estimasi: min(g_roe, g_hist) jika keduanya ada
        if g_roe is not None and g_hist is not None:
            g_base = min(g_roe, g_hist)
        elif g_roe is not None:
            g_base = g_roe
        elif g_hist is not None:
            g_base = g_hist
        else:
            g_base = cls.TERM_GROWTH  # fallback GDP Indonesia

        # Haircut terminal: pertumbuhan jangka panjang selalu lebih rendah
        tg = g_base * 0.50

        # Penalty leverage: D/E > 100% → kurangi tg
        de_safe = de_ratio if (de_ratio is not None and de_ratio >= 0) else 0.0
        if de_safe > 100:
            penalty = min(0.01, (de_safe - 100) / 100 * 0.005)
            tg -= penalty

        # Clamp
        tg = max(cls.TG_MIN, min(tg, cls.TG_MAX))

        # Blend dengan sektor jika tersedia
        if sector_tg is not None:
            tg = 0.60 * tg + 0.40 * sector_tg
            tg = max(cls.TG_MIN, min(tg, cls.TG_MAX))

        return round(tg, 4)

    def analyze_one(self, ticker: str) -> StockResult:
        raw = ticker.upper().strip()
        jk = raw if raw.endswith(".JK") else f"{raw}.JK"

        for attempt in range(self.RETRY + 1):
            try:
                tk = yf.Ticker(jk)
                info = (tk.info) or {}

                income_stmt = tk.income_stmt
                if income_stmt is None or income_stmt.empty:
                    income_stmt = tk.financials
                balance_sheet = tk.balance_sheet
                cashflow = tk.cashflow

                net_income_map = self._extract_series(
                    income_stmt,
                    [
                        "Net Income Common Stockholders",
                        "Net Income",
                        "NetIncome",
                        "Net Income Applicable To Common Shares",
                    ],
                )
                revenue_map = self._extract_series(
                    income_stmt,
                    ["Total Revenue", "Revenue", "Operating Revenue"],
                )
                equity_map = self._extract_series(
                    balance_sheet,
                    [
                        "Stockholders Equity",
                        "Total Equity Gross Minority Interest",
                        "Common Stock Equity",
                        "Total Stockholder Equity",
                    ],
                )
                ocf_map = self._extract_series(
                    cashflow,
                    [
                        "Operating Cash Flow",
                        "Total Cash From Operating Activities",
                    ],
                )
                capex_map = self._extract_series(
                    cashflow,
                    [
                        "Capital Expenditure",
                        "Capital Expenditures",
                    ],
                )
                dep_map = self._extract_series(
                    cashflow,
                    [
                        "Depreciation And Amortization",
                        "Depreciation",
                        "Depreciation & Amortization",
                    ],
                )

                cp = self._g(info, "currentPrice") or self._g(info, "regularMarketPrice")
                if cp is None:
                    if attempt < self.RETRY:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    return StockResult(ticker=jk, error="No data")

                eps = self._g(info, "trailingEps")
                bv = self._g(info, "bookValue")
                rg  = self._g(info, "revenueGrowth")
                eg  = self._g(info, "earningsGrowth")
                ni  = self._g(info, "netIncomeToCommon")
                so  = self._g(info, "sharesOutstanding")
                pe  = self._g(info, "trailingPE")
                pbv = self._g(info, "priceToBook")
                de  = self._g(info, "debtToEquity")
                dy  = self._g(info, "dividendYield")
                tm  = self._g(info, "targetMeanPrice")
                pr  = self._g(info, "payoutRatio")
                roe_info = self._g(info, "returnOnEquity")

                stmt_ni = self._latest_value(net_income_map)
                stmt_rev = self._latest_value(revenue_map)
                stmt_eq = self._latest_value(equity_map)
                stmt_ocf = self._latest_value(ocf_map)
                stmt_capex = self._latest_value(capex_map)
                stmt_dep = self._latest_value(dep_map)

                # --- CURRENCY FIX FOR ADRO, ITMG, INCO, MEDC ---
                # If financials are in USD but stock price is IDR
                currency = info.get("currency", "IDR")
                fin_currency = info.get("financialCurrency", "IDR")
                fx_rate = 1.0

                if currency == "IDR" and fin_currency == "USD":
                    try:
                        fx_tk = yf.Ticker("USDIDR=X")
                        fx_rate = self._g(fx_tk.info, "regularMarketPrice") or self._g(fx_tk.info, "previousClose") or 15500.0
                    except:
                        fx_rate = 15500.0

                    # Convert all raw financial statement values to IDR
                    if stmt_ni is not None: stmt_ni *= fx_rate
                    if stmt_rev is not None: stmt_rev *= fx_rate
                    if stmt_eq is not None: stmt_eq *= fx_rate
                    if stmt_ocf is not None: stmt_ocf *= fx_rate
                    if stmt_capex is not None: stmt_capex *= fx_rate
                    if stmt_dep is not None: stmt_dep *= fx_rate

                    # Also fix info['bookValue'] which is often returned in USD when price is IDR
                    if bv is not None and bv < 10:
                        bv *= fx_rate

                so_eff = so
                implied_shares = None
                if stmt_ni is not None and eps is not None and eps > 0:
                    implied_shares = stmt_ni / eps

                if implied_shares is not None and implied_shares > 0:
                    if so_eff is None or so_eff <= 0:
                        so_eff = implied_shares
                    else:
                        ratio = implied_shares / so_eff if so_eff > 0 else None
                        if ratio is not None and (ratio > 3.0 or ratio < 0.33):
                            so_eff = implied_shares

                if ni is None:
                    ni = stmt_ni

                capex_abs = abs(stmt_capex) if stmt_capex is not None else None

                if stmt_ocf is not None and capex_abs is not None:
                    fcf_total = stmt_ocf - capex_abs
                else:
                    fcf_total = None

                if stmt_ni is not None and stmt_dep is not None and capex_abs is not None:
                    owner_earnings_total = stmt_ni + stmt_dep - capex_abs
                else:
                    owner_earnings_total = None

                owner_earnings_ps = None
                fcf_ps = None

                if so_eff is not None and so_eff > 0:
                    if eps is None and stmt_ni is not None:
                        eps = stmt_ni / so_eff
                    if bv is None and stmt_eq is not None:
                        bv = stmt_eq / so_eff
                    if owner_earnings_total is not None:
                        owner_earnings_ps = owner_earnings_total / so_eff
                    if fcf_total is not None:
                        fcf_ps = fcf_total / so_eff

                if rg is None and revenue_map:
                    rg = self._cagr(revenue_map, periods=1)
                if eg is None and net_income_map:
                    eg = self._cagr(net_income_map, periods=3)

                if roe_info is not None:
                    roe_from_info = roe_info * 100 if abs(roe_info) < 1 else roe_info
                else:
                    roe_from_info = None

                if stmt_ni is not None and stmt_eq is not None and stmt_eq > 0:
                    roe_stmt = (stmt_ni / stmt_eq) * 100
                else:
                    roe_stmt = None

                roe = self._c(roe_stmt if roe_stmt is not None else self.calc_roe(ni, bv, so))
                if roe is None:
                    roe = self._c(roe_from_info)

                # Best growth estimate: earningsGrowth > revenueGrowth
                best_growth = eg if eg is not None else rg
                growth_pct = None
                if best_growth is not None:
                    growth_pct = best_growth * 100 if abs(best_growth) < 1 else best_growth
                # payout=0 valid (perusahaan reinvest semua laba, tidak bayar dividen)
                # payout>=1 invalid (dividen > laba) → default 0.3
                payout = pr if pr is not None and 0 <= pr < 1 else 0.3

                # Deteksi bank/lembaga keuangan
                is_bank = self._is_bank(info)
                sector_template, sector_dcf = self._resolve_sector_dcf(info)

                # Referensi sektor sebagai anchor (bukan nilai final)
                sector_wacc_ref = float(sector_dcf.get("wacc", self.DISCOUNT)) if sector_dcf else None
                sector_tg_ref   = float(sector_dcf.get("terminal_g", self.TERM_GROWTH)) if sector_dcf else None
                dcf_curve = list(sector_dcf.get("growth_curve", [])) if sector_dcf else None

                # ── WACC per saham ──────────────────────────────────
                beta_val = self._g(info, "beta")
                dcf_r = self._calc_stock_wacc(
                    info=info,
                    beta=beta_val,
                    de_ratio=de,
                    income_stmt=income_stmt,
                    balance_sheet=balance_sheet,
                    sector_wacc=sector_wacc_ref,
                )

                # ── Terminal growth per saham ────────────────────────
                dcf_tg = self._calc_stock_terminal_g(
                    roe=roe,
                    best_growth=best_growth,
                    de_ratio=de,
                    payout=payout,
                    sector_tg=sector_tg_ref,
                )

                # Pastikan terminal spread cukup (WACC - tg >= 3%)
                if dcf_r - dcf_tg < 0.03:
                    dcf_tg = max(self.TG_MIN, dcf_r - 0.03)

                epv_base_ps = owner_earnings_ps if owner_earnings_ps is not None and owner_earnings_ps > 0 else eps

                if is_bank:
                    # Bank: OCF/FCF tidak representatif (simpanan masuk OCF).
                    # Gunakan EPS sebagai basis DCF — lebih andal untuk bank.
                    # EPV juga lebih tepat pakai EPS untuk bank.
                    dcf_base_ps = eps if eps is not None and eps > 0 else None
                    epv_base_ps = eps if eps is not None and eps > 0 else None
                else:
                    dcf_base_ps = fcf_ps if fcf_ps is not None and fcf_ps > 0 else epv_base_ps

                quality_graham = 1.0 if growth_pct is not None and bv is not None else 0.85
                # Bank: turunkan quality DCF karena pakai EPS bukan FCF
                if is_bank:
                    quality_epv = 1.0 if eps is not None and eps > 0 else 0.8
                    quality_dcf = 0.9 if eps is not None and eps > 0 else 0.7
                else:
                    quality_epv = 1.15 if owner_earnings_ps is not None and owner_earnings_ps > 0 else 0.9
                    quality_dcf = 1.2 if fcf_ps is not None and fcf_ps > 0 else (1.0 if epv_base_ps is not None else 0.75)
                quality_rim = 1.1 if stmt_eq is not None else 0.9

                # Growth-quality tilt:
                # For high-quality compounders, EPV (zero-growth floor) can be too punitive.
                # Reduce EPV influence and slightly favor forward-looking methods.
                is_growth_quality = (
                    (best_growth is not None and best_growth >= 0.10) and
                    (roe is not None and roe >= 10) and
                    (de is None or de <= 80)
                )
                if is_growth_quality:
                    quality_epv *= 0.65
                    quality_dcf *= 1.15
                    quality_graham *= 1.05
                    quality_rim *= 1.05

                # Asset-light premium (common in healthcare/consumer growth):
                # very high PBV can make EPV and BV-anchored RIM too conservative.
                is_asset_light_growth = is_growth_quality and (pbv is not None and pbv >= 2.5)
                if is_asset_light_growth:
                    quality_epv *= 0.55
                    quality_rim *= 0.60
                    quality_dcf *= 1.20
                    quality_graham *= 1.15

                method_quality = {
                    "graham": quality_graham,
                    "epv": quality_epv,
                    "dcf": quality_dcf,
                    "rim": quality_rim,
                }

                # 4 valuations — new formulas
                vg = self._c(self.v_graham(eps, bv, growth_pct, self.AAA_YIELD))
                ve = self._c(self.v_epv(epv_base_ps, self.DISCOUNT))
                vd = self._c(self.v_dcf(dcf_base_ps, roe, best_growth, payout,
                                        dcf_r, dcf_tg, growth_curve=dcf_curve))
                vr = self._c(self.v_rim(bv, roe, payout, self.DISCOUNT,
                                        self.TERM_GROWTH))

                # Soft anchor for asset-light quality growth stocks:
                # floor-based methods (EPV/RIM) can materially understate intangible compounding.
                # We add a forward PE-based anchor and (if available) analyst anchor.
                growth_anchor = None
                analyst_anchor = None
                if is_asset_light_growth and eps is not None and eps > 0:
                    g_dec = best_growth if best_growth is not None else 0.08
                    g_dec = max(0.0, min(g_dec, 0.25))

                    growth_pct_eff = g_dec * 100
                    roe_eff = roe if roe is not None else 10.0
                    fair_pe = 16.0 + (0.8 * growth_pct_eff) + (0.5 * max(0.0, roe_eff - 10.0))
                    fair_pe = max(18.0, min(34.0, fair_pe))

                    forward_eps = eps * (1.0 + g_dec)
                    growth_anchor = self._c(forward_eps * fair_pe * 0.95)

                    if tm is not None and cp is not None and cp > 0 and tm > cp * 1.05:
                        analyst_anchor = self._c(tm * 0.90)

                soft_anchor = None
                if growth_anchor is not None and analyst_anchor is not None:
                    soft_anchor = self._c(0.60 * growth_anchor + 0.40 * analyst_anchor)
                elif growth_anchor is not None:
                    soft_anchor = growth_anchor
                elif analyst_anchor is not None:
                    soft_anchor = analyst_anchor

                # Guardrail: cap DCF outliers relative to other anchor methods
                if vd is not None and vd > 0:
                    caps = []
                    if ve is not None and ve > 0:
                        caps.append(ve * self.DCF_TO_EPV_CAP)
                    if vg is not None and vg > 0:
                        caps.append(vg * self.DCF_TO_GRAHAM_CAP)
                    # Bank guardrail tambahan: DCF bank max 2.5× book value
                    if is_bank and bv is not None and bv > 0:
                        caps.append(bv * 2.5)
                    if caps:
                        vd = self._c(min(vd, min(caps)))

                # Sanitasi: pastikan semua valuasi negatif dibuang
                if vg is not None and vg <= 0:
                    vg = None
                if ve is not None and ve <= 0:
                    ve = None
                if vd is not None and vd <= 0:
                    vd = None
                if vr is not None and vr <= 0:
                    vr = None

                # Blended intrinsic (RIM replaces PEG)
                iv = self._c(self.blend(vg, ve, vd, vr, method_quality))
                if is_asset_light_growth and soft_anchor is not None:
                    if iv is None:
                        iv = soft_anchor
                    else:
                        iv = self._c(max(iv, soft_anchor))
                iv_use = iv if iv is not None else (vg if vg is not None else ve)
                mos_val = self._c(self.mos(cp, iv_use))

                # Count methods that say undervalued
                agree = sum(1 for v in [vg, ve, vd, vr]
                            if v is not None and v > cp)

                # Score
                sc = self.calc_score(
                    self._c(pe), self._c(pbv), roe,
                    self._c(de), mos_val, self._c(dy)
                )

                # Analyst upside %
                a_up = self._c(((tm - cp) / cp * 100)) if tm is not None and cp is not None and cp != 0 else None

                # Data confidence
                conf = self.calc_confidence(
                    eps, bv, pe, pbv, roe, de, best_growth, tm
                )

                data_points = [
                    eps, bv, pe, pbv, roe, de, best_growth, tm, ni, so,
                    stmt_ni, stmt_eq, stmt_rev, stmt_ocf, stmt_capex, stmt_dep,
                    owner_earnings_ps, fcf_ps, so_eff, implied_shares,
                ]
                available_points = sum(1 for value in data_points if value is not None)
                if available_points >= 14:
                    dq = "A"
                elif available_points >= 11:
                    dq = "B"
                elif available_points >= 8:
                    dq = "C"
                else:
                    dq = "D"

                # Probability (quality-calibrated sigmoid)
                prob = self.calc_probability(
                    self._c(pe), self._c(pbv), roe, self._c(de),
                    mos_val, agree, best_growth, a_up,
                    confidence=conf, data_quality=dq,
                )

                return StockResult(
                    ticker=jk, price=self._c(cp),
                    graham=vg, epv=ve, dcf=vd, rim=vr,
                    blended_iv=iv,
                    intrinsic=iv_use,
                    mos_pct=mos_val, score=sc,
                    prob_up_pct=prob,
                    confidence=conf,
                    data_quality=dq,
                    status=self.get_status(mos_val, sc),
                    pe=self._c(pe), pbv=self._c(pbv),
                    roe=roe, de=self._c(de),
                    growth=self._c(best_growth * 100) if best_growth is not None else None,
                    div_yield=self._c(dy * 100) if dy is not None else None,
                    analyst_target=self._c(tm),
                    methods_agree=agree,
                    sector_template=sector_template,
                    wacc_used=self._c(dcf_r * 100) if dcf_r is not None else None,
                    terminal_g_used=self._c(dcf_tg * 100) if dcf_tg is not None else None,
                )

            except Exception as e:
                if attempt < self.RETRY:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return StockResult(ticker=jk, error=str(e))

        return StockResult(ticker=jk, error="Max retries")

    # ── Sequential Batch (reliable, no rate limit) ──────

    def analyze_batch(self, tickers: List[str]) -> List[StockResult]:
        results = []
        total = len(tickers)

        print(f"\n{'='*60}")
        print(f"  📊 Analyzing {total} stocks (sequential)")
        print(f"{'='*60}\n")

        for i, t in enumerate(tickers, 1):
            r = self.analyze_one(t)
            results.append(r)
            st = "✓" if r.error is None else f"✗ {r.error}"

            pct = i / total
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            sys.stdout.write(f"\r  [{bar}] {i}/{total} ({pct:.0%}) — {t}: {st}    ")
            sys.stdout.flush()

            # Delay to avoid rate limiting
            if i < total:
                time.sleep(0.5)

        ok = sum(1 for r in results if r.error is None)
        print(f"\n\n  ✅ {ok}/{total} berhasil\n")

        return sorted(results,
                      key=lambda x: x.prob_up_pct if x.error is None else -1,
                      reverse=True)


# ════════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════════

def show(results: List[StockResult]):
    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not valid:
        print("  ⚠ Tidak ada data valid.")
        return

    # Build simple table
    rows = []
    for r in valid:
        rows.append({
            "Ticker": r.ticker.replace(".JK", ""),
            "SektorDCF": r.sector_template,
            "WACC%": r.wacc_used,
            "Tg%": r.terminal_g_used,
            "Harga": r.price,
            "Graham": r.graham,
            "EPV": r.epv,
            "DCF": r.dcf,
            "RIM": r.rim,

            "Intrinsic": r.intrinsic,
            "MoS%": r.mos_pct,
            "Score": r.score,
            "Prob⬆%": r.prob_up_pct,
            "Conf": r.confidence,
            "DataQ": r.data_quality,
            "Status": r.status,
        })

    df = pd.DataFrame(rows)

    # Replace all NaN/None with "—" for display
    df = df.fillna("—")

    print(f"\n{'='*130}")
    print(f"  📈 STOCK VALUATION — {len(valid)} Saham IDX (sorted by Probabilitas Naik)")
    print(f"{'='*130}")
    print(f"  Graham = EPS×(8.5+2g)×4.4/Y | EPV = OwnerEarnings/12%×0.9 | DCF = growth curve per sektor (template Indonesia) | RIM = BV+ExcessEarnings")
    print(f"  Prob⬆ = sigmoid model (22%-78% range) | Conf = A(best)/B/C/D(least data)")
    print(f"  DataQ = kualitas data input per ticker (A terbaik)")
    print(f"{'='*130}\n")

    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="rounded_grid",
                       showindex=False, floatfmt=".1f", missingval="—"))
    except ImportError:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.float_format", "{:.1f}".format)
        print(df.to_string(index=False))

    # Summary
    mos_data = [r.mos_pct for r in valid if r.mos_pct is not None]
    sangat_murah = [r for r in valid if "SANGAT MURAH" in r.status]
    murah = [r for r in valid if "MURAH" in r.status and "SANGAT" not in r.status]
    wajar = [r for r in valid if "WAJAR" in r.status]
    mahal = [r for r in valid if "MAHAL" in r.status]

    print(f"\n  {'─'*55}")
    print(f"  📊 RINGKASAN")
    print(f"  {'─'*55}")
    print(f"     Total          : {len(valid)} saham")
    print(f"     🟢 Sangat Murah: {len(sangat_murah)}")
    print(f"     🟡 Murah       : {len(murah)}")
    print(f"     🟠 Wajar       : {len(wajar)}")
    print(f"     🔴 Mahal       : {len(mahal)}")
    if mos_data:
        print(f"     Avg MoS        : {sum(mos_data)/len(mos_data):.1f}%")
    avg_prob = sum(r.prob_up_pct for r in valid) / len(valid)
    print(f"     Avg Prob Naik  : {avg_prob:.1f}%")

    if sangat_murah:
        print(f"\n  🏆 TOP PICKS (Sangat Murah + Prob tertinggi):")
        for i, r in enumerate(sangat_murah[:5], 1):
            print(f"     {i}. {r.ticker.replace('.JK','')} — "
                  f"Harga: {r.price:,.0f} → Intrinsic: {r.intrinsic:,.0f} | "
                  f"MoS: {r.mos_pct}% | Score: {r.score} | Prob⬆: {r.prob_up_pct}%")
    print(f"  {'─'*55}\n")

    if errors:
        print(f"  ⚠ {len(errors)} gagal: {', '.join(e.ticker.replace('.JK','') for e in errors)}\n")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    regime = "conservative"
    try:
        ValueAnalyzer.set_regime(regime)
    except ValueError:
        print(f"  ⚠ Regime '{regime}' tidak valid, fallback ke conservative")
        ValueAnalyzer.set_regime("conservative")

    a = ValueAnalyzer()

    print(f"\n  ⚙️  Valuation regime: {ValueAnalyzer.VALUATION_REGIME}")


    tickers = [
    "DKFT", "AGRO", "BANK", "MDIA", "GJTL",
    "JTPE", "KRAS", "MBSS", "MSTI", "MPMX",
    "EURO", "ELSA", "SMRA", "LPKR", "INET",
    "SMDR", "SINI", "ACES", "ERAA", "NICL",
    "OMED", "BULL", "GMFI", "TMAS", "WBSA",
    "SSMS", "BTPS", "FORE", "PNLF", "DMAS",
    "SSIA", "BJTM", "CNMA", "CYBR", "BJBR",
    "LSIP", "SIMP", "KPIG", "MIDI", "PACK",
    "BBKP", "WIFI", "BFIN", "STAA", "HRTA",
    "HRUM", "NSSS", "POWR", "BIPI", "RATU",
    "SMGR", "AUTO", "SIDO", "ESSA", "CTRA",
    "PSAB", "AALI", "COIN", "INDY", "DSNG",
    "BKSL", "RMKE", "BUKA", "ALII", "APIC",
    "RAJA", "CMNT", "PWON", "DEWA", "BSDE",
    "HEAL", "ULTJ", "BUVA", "SCMA", "MAPA",
    "ARKO", "INTP", "TKIM", "BBTN", "AVIA",
    "BBHI", "JSMR", "SRTG", "BNBR", "PNBN",
    "TOWR", "MIKA", "GIAA", "MAPI", "TINS",
    "AKRA", "MSIN", "FILM", "SUPA", "TAPG",
    "JPFA", "ARCI", "VKTR", "TBIG", "MEDC",
    "CMRY", "KLBF", "ENRG", "PGEO", "PTRO",
    "EMTK", "BNGA", "MYOR", "MTEL", "INKP",
    "BELI", "PGAS", "BDMN", "AMRT", "EXCL",
    "MBMA", "NCKL", "TCPI", "INCO", "CUAN",
    "GOTO", "INDF", "ADMR", "BUMI", "AADI",
    "UNVR", "ADRO", "MDKA", "ISAT", "CPIN",
    "ANTM", "ICBP", "HMSP", "BRMS", "DSSA",
    "BRIS", "UNTR", "CDIA", "IMPC", "EMAS",
    "BBNI", "PANI", "BRPT", "TPIA", "ASII",
    "AMMN", "TLKM", "BREN", "BBRI", "BBCA",
]

    t0 = time.time()
    results = a.analyze_batch(tickers)
    el = time.time() - t0

    show(results)

    print(f"  ⏱  {el:.1f}s ({el/len(tickers):.2f}s/saham) | "
            f"proses sequential\n")

    # ── Save to CSV ──
    valid = [r for r in results if r.error is None]
    if valid:
        rows = []
        for r in valid:
            rows.append({
                "Ticker": r.ticker.replace(".JK", ""),
                "Sektor_DCF": r.sector_template,
                "WACC_%": r.wacc_used,
                "Terminal_g_%": r.terminal_g_used,
                "Harga": r.price,
                "Graham": r.graham,
                "EPV": r.epv,
                "DCF": r.dcf,
                "RIM": r.rim,

                "Intrinsic": r.intrinsic,
                "MoS%": r.mos_pct,
                "Score": r.score,
                "Prob_Naik%": r.prob_up_pct,
                "Confidence": r.confidence,
                "Data_Quality": r.data_quality,
                "Status": r.status,
            })
        csv_path = "stock_valuation_results.csv"
        pd.DataFrame(rows).fillna("—").to_csv(csv_path, index=False)
        print(f"  💾 Disimpan ke: {csv_path}\n")