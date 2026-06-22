"""
Portfolio Optimizer — Ensemble (DE + PSO + SA)
===============================================
Runs 3 metaheuristics and auto-picks the best result.

Algorithms:
  1. Differential Evolution (DE)  — best for continuous weight optimization
  2. Particle Swarm Optimization (PSO) — best exploration
  3. Simulated Annealing (SA) — good for escaping local optima

Why this ensemble beats single-GA:
  - DE converges faster with fewer params (only F=0.8, CR=0.9)
  - PSO explores solution space differently (velocity-based)
  - SA adds stochastic escape from local optima
  - Ensemble picks the BEST result = lower variance

Source: stock_valuation_results.csv (run stock_valuation_optimizer.py first)
Filter: Only 🟢 SANGAT MURAH stocks as candidates.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import math
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable
from copy import deepcopy

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════

RISK_FREE_RATE = 0.065
TRADING_DAYS = 252
PORTFOLIO_SIZE = 5
HIST_PERIOD = "1y"
HIST_PERIOD_DOWNLOAD_PERIODS = {
    "3y": "5y",
    "4y": "5y",
}

# Optimizer params
POP_SIZE = 200
MAX_ITER = 500
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.50

CSV_FILE = "stock_valuation_results.csv"


def _history_download_period(hist_period: str) -> str:
    period = str(hist_period or HIST_PERIOD).strip().lower()
    return HIST_PERIOD_DOWNLOAD_PERIODS.get(period, period)


def _trim_history_to_period(hist: pd.DataFrame, hist_period: str) -> pd.DataFrame:
    period = str(hist_period or HIST_PERIOD).strip().lower()
    if hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
        return hist

    try:
        if period.endswith("mo"):
            start = hist.index.max() - pd.DateOffset(months=int(period[:-2]))
        elif period.endswith("y"):
            start = hist.index.max() - pd.DateOffset(years=int(period[:-1]))
        else:
            return hist
    except (TypeError, ValueError):
        return hist

    return hist[hist.index >= start]


def _download_price_history(ticker_obj, hist_period: str) -> pd.DataFrame:
    hist = ticker_obj.history(period=_history_download_period(hist_period))
    return _trim_history_to_period(hist, hist_period)


def load_price_history(ticker: str, hist_period: str = HIST_PERIOD) -> pd.DataFrame:
    jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
    return _download_price_history(yf.Ticker(jk), hist_period)


# ════════════════════════════════════════════════════════════════
# CSV Reader
# ════════════════════════════════════════════════════════════════

def load_valuation_csv(filepath: str = CSV_FILE) -> Tuple[List[str], Dict[str, dict]]:
    """
    Read from stock_valuation_results.csv.
    CSV columns: Ticker, Harga, Graham, EPV, DCF, PEG, Intrinsic, MoS%, Score, Prob_Naik%, Status
    Filter: only SANGAT MURAH.
    """
    if not os.path.exists(filepath):
        print(f"  ❌ File tidak ditemukan: {filepath}")
        print(f"  💡 Jalankan dulu: python stock_valuation_optimizer.py")
        sys.exit(1)

    df = pd.read_csv(filepath)
    df = df.replace("—", pd.NA)
    total = len(df)

    # Filter SANGAT MURAH only
    df["_sc"] = df["Status"].astype(str).str.upper().str.replace(" ", "_")
    df = df[df["_sc"].str.contains("SANGAT_MURAH", na=False)]

    print(f"  📄 CSV: {total} total → {len(df)} kandidat SANGAT MURAH")

    if df.empty:
        print(f"  ❌ Tidak ada saham SANGAT MURAH!")
        sys.exit(1)

    def sf(val, default=0.0):
        try:
            v = float(val)
            return v if math.isfinite(v) else default
        except (ValueError, TypeError):
            return default

    tickers, val_data = [], {}
    for _, row in df.iterrows():
        t = str(row.get("Ticker", "")).strip()
        if not t:
            continue

        dc = sum(1 for c in ["Graham", "EPV", "DCF", "PEG", "Intrinsic", "MoS%", "Score"]
                 if pd.notna(row.get(c)) and str(row.get(c)).strip() != "")
        conf = "A" if dc >= 6 else ("B" if dc >= 4 else ("C" if dc >= 2 else "D"))

        tickers.append(t)
        val_data[t] = {
            "mos": sf(row.get("MoS%"), 0),
            "score": sf(row.get("Score"), 0),
            "prob": sf(row.get("Prob_Naik%"), 50),
            "conf": conf,
        }

    print(f"  ✅ {len(tickers)} kandidat siap\n")
    return tickers, val_data


def _to_float(value, default=0.0):
    """Convert display/csv values to finite floats."""
    try:
        if pd.isna(value):
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_valuation_dataframe(
    df: pd.DataFrame,
    status_filter: str = "SANGAT MURAH",
    min_score: float = 0.0,
    min_prob: float = 0.0,
    top_n: Optional[int] = None,
) -> Tuple[List[str], Dict[str, dict], pd.DataFrame]:
    """
    Convert valuation output to optimizer input.

    This is the Streamlit-friendly pair of load_valuation_csv(): it accepts the
    live valuation table directly instead of forcing a CSV round-trip.
    """
    if df is None or df.empty:
        return [], {}, pd.DataFrame()

    work = df.copy().replace("—", pd.NA)

    ticker_col = _first_existing_column(work, ["Ticker", "ticker"])
    mos_col = _first_existing_column(work, ["MoS%", "MOS%", "mos_pct", "MoS"])
    score_col = _first_existing_column(work, ["Score", "score"])
    prob_col = _first_existing_column(work, ["Prob_Naik%", "Prob⬆%", "Prob Naik%", "prob_up_pct"])
    conf_col = _first_existing_column(work, ["Confidence", "Conf", "confidence"])
    status_col = _first_existing_column(work, ["Status", "status"])

    if ticker_col is None:
        return [], {}, pd.DataFrame()

    if status_col is not None and status_filter and status_filter.upper() != "SEMUA":
        status_key = status_filter.upper().replace(" ", "_")
        normalized = work[status_col].astype(str).str.upper().str.replace(" ", "_", regex=False)
        work = work[normalized.str.contains(status_key, na=False)]

    if score_col is not None and min_score > 0:
        work = work[work[score_col].apply(lambda x: _to_float(x, -999)) >= min_score]

    if prob_col is not None and min_prob > 0:
        work = work[work[prob_col].apply(lambda x: _to_float(x, -999)) >= min_prob]

    if prob_col is not None or score_col is not None:
        sort_cols = [c for c in [prob_col, score_col, mos_col] if c is not None]
        for col in sort_cols:
            work[f"__sort_{col}"] = work[col].apply(_to_float)
        work = work.sort_values([f"__sort_{c}" for c in sort_cols], ascending=False)
        work = work.drop(columns=[f"__sort_{c}" for c in sort_cols])

    if top_n is not None and top_n > 0:
        work = work.head(top_n)

    tickers: List[str] = []
    val_data: Dict[str, dict] = {}
    for _, row in work.iterrows():
        ticker = str(row.get(ticker_col, "")).strip().upper().replace(".JK", "")
        if not ticker:
            continue
        tickers.append(ticker)
        val_data[ticker] = {
            "mos": _to_float(row.get(mos_col), 0.0) if mos_col else 0.0,
            "score": _to_float(row.get(score_col), 0.0) if score_col else 0.0,
            "prob": _to_float(row.get(prob_col), 50.0) if prob_col else 50.0,
            "conf": str(row.get(conf_col, "D") or "D")[:1].upper() if conf_col else "D",
        }

    return tickers, val_data, work


# ════════════════════════════════════════════════════════════════
# Data Models & Collection
# ════════════════════════════════════════════════════════════════

@dataclass
class StockData:
    ticker: str
    price: float
    returns: np.ndarray
    annual_return: float
    annual_vol: float
    max_drawdown: float
    market_cap: Optional[float]
    dividend_yield: float
    mos_pct: float
    val_score: float
    prob_up: float
    confidence: str


@dataclass
class PortfolioResult:
    stocks: List[str]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    fitness: float
    profile: str
    algorithm: str
    generations: int
    stock_data: List[StockData] = field(default_factory=list)


def fetch_stock(
    ticker: str,
    val_info: dict,
    hist_period: str = HIST_PERIOD,
) -> Optional[StockData]:
    jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
    try:
        tk = yf.Ticker(jk)
        hist = _download_price_history(tk, hist_period)
        if hist.empty or len(hist) < 60:
            return None

        closes = hist["Close"].values
        rets = np.diff(np.log(closes))
        if len(rets) < 30:
            return None

        ann_ret = np.mean(rets) * TRADING_DAYS
        ann_vol = np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)

        cum = np.cumprod(1 + np.diff(closes) / closes[:-1])
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        max_dd = np.max(dd) if len(dd) > 0 else 0

        info = tk.info or {}
        dy = info.get("dividendYield", 0) or 0

        return StockData(
            ticker=jk.replace(".JK", ""),
            price=closes[-1],
            returns=rets,
            annual_return=ann_ret,
            annual_vol=ann_vol,
            max_drawdown=max_dd,
            market_cap=info.get("marketCap"),
            dividend_yield=dy * 100 if dy < 1 else dy,
            mos_pct=val_info.get("mos", 0),
            val_score=val_info.get("score", 0),
            prob_up=val_info.get("prob", 50),
            confidence=val_info.get("conf", "D"),
        )
    except Exception:
        return None


def fetch_all(
    tickers: List[str],
    val_data: Dict,
    hist_period: str = HIST_PERIOD,
) -> List[StockData]:
    stocks = []
    total = len(tickers)
    print(f"  📥 Downloading {total} stocks...\n")

    for i, t in enumerate(tickers, 1):
        sd = fetch_stock(t, val_data.get(t, {}), hist_period=hist_period)
        st = "✓" if sd else "✗"
        if sd:
            stocks.append(sd)
        pct = i / total
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        sys.stdout.write(f"\r  [{bar}] {i}/{total} ({pct:.0%}) — {t}: {st}    ")
        sys.stdout.flush()
        time.sleep(0.3)

    print(f"\n\n  ✅ {len(stocks)}/{total} berhasil\n")
    return stocks


# ════════════════════════════════════════════════════════════════
# Portfolio Metrics Engine
# ════════════════════════════════════════════════════════════════

class PortfolioEngine:
    """Shared fitness evaluator for all algorithms."""

    def __init__(self, stocks: List[StockData], profile: str):
        self.stocks = stocks
        self.n = len(stocks)
        self.profile = profile.upper()

        # Pre-build annualized covariance matrix
        min_len = min(len(s.returns) for s in stocks)
        aligned = np.array([s.returns[-min_len:] for s in stocks])
        self.cov_matrix = np.cov(aligned) * TRADING_DAYS
        self.min_return_len = min_len
        self.eval_count = 0

    def decode(self, x: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Decode continuous vector to (stock_indices, weights).
        x[0:5] → stock indices (continuous → integer via ranking)
        x[5:10] → raw weights (normalized + constrained)
        """
        # Stock selection: use argsort ranking of first 5 dims
        # mapped to [0, n_stocks-1]
        idx_raw = x[:PORTFOLIO_SIZE] * self.n
        idx_raw = np.clip(idx_raw, 0, self.n - 0.01)

        # Get unique indices by ranking
        ranked = np.argsort(np.argsort(-idx_raw))  # rank descending
        all_indices = np.argsort(-idx_raw)
        selected = []
        for i in all_indices:
            idx = int(idx_raw[i]) % self.n
            if idx not in selected:
                selected.append(idx)
            if len(selected) == PORTFOLIO_SIZE:
                break

        # Fallback if not enough unique
        while len(selected) < PORTFOLIO_SIZE:
            for j in range(self.n):
                if j not in selected:
                    selected.append(j)
                    break

        # Weights
        w_raw = np.abs(x[PORTFOLIO_SIZE:2 * PORTFOLIO_SIZE]) + 0.01
        w = np.clip(w_raw / w_raw.sum(), MIN_WEIGHT, MAX_WEIGHT)
        w /= w.sum()

        return selected, w

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate portfolio fitness from continuous vector."""
        self.eval_count += 1
        indices, weights = self.decode(x)

        # Portfolio return
        ret = sum(weights[k] * self.stocks[indices[k]].annual_return
                  for k in range(PORTFOLIO_SIZE))

        # Portfolio volatility (covariance)
        sub_cov = self.cov_matrix[np.ix_(indices, indices)]
        vol = np.sqrt(max(weights @ sub_cov @ weights, 1e-10))

        # Sharpe
        sharpe = (ret - RISK_FREE_RATE) / max(vol, 1e-6)

        # Sortino
        port_rets = np.zeros(self.min_return_len)
        for k in range(PORTFOLIO_SIZE):
            port_rets += weights[k] * self.stocks[indices[k]].returns[-self.min_return_len:]
        downside = port_rets[port_rets < 0]
        if len(downside) >= 2:
            down_vol = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)
            sortino = (ret - RISK_FREE_RATE) / max(down_vol, 1e-6)
        else:
            sortino = 0.0

        # Max drawdown
        cum = np.cumprod(1 + port_rets)
        peak = np.maximum.accumulate(cum)
        mdd = np.max((peak - cum) / peak) if len(cum) > 0 else 0

        # Valuation bonus
        val = 0
        for k in range(PORTFOLIO_SIZE):
            s = self.stocks[indices[k]]
            mos_n = max(0, min(1, s.mos_pct / 50))
            sco_n = s.val_score / 100
            pro_n = (s.prob_up - 22) / 56
            cm = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4}.get(s.confidence, 0.4)
            val += weights[k] * (0.35 * mos_n + 0.30 * sco_n + 0.35 * pro_n) * cm

        # Avg dividend
        avg_div = sum(weights[k] * self.stocks[indices[k]].dividend_yield
                      for k in range(PORTFOLIO_SIZE))

        # Concentration penalty (HHI)
        hhi = sum(w ** 2 for w in weights)
        div_pen = max(0, hhi - 0.25) * 2

        # ── Profile-specific fitness ──
        if self.profile == "SAFE":
            f = (2.0 * sortino + 1.0 * sharpe - 3.0 * vol - 4.0 * mdd
                 + 1.5 * val + 0.5 * (avg_div / 5) - div_pen)
            if vol > 0.20: f -= (vol - 0.20) * 10
            if mdd > 0.15: f -= (mdd - 0.15) * 8
        elif self.profile == "AGGRESSIVE":
            f = (3.0 * ret + 1.0 * sharpe + 0.5 * sortino
                 - 0.5 * mdd + 2.0 * val - 0.5 * div_pen)
        else:  # MODERATE
            f = (2.0 * sharpe + 1.0 * sortino + 0.5 * ret
                 - 2.0 * vol - 2.0 * mdd + 1.5 * val - 0.8 * div_pen)

        return f

    def build_result(self, x: np.ndarray, algo: str,
                     gen: int, profile: str) -> PortfolioResult:
        """Build PortfolioResult from best solution vector."""
        indices, weights = self.decode(x)

        ret = sum(weights[k] * self.stocks[indices[k]].annual_return
                  for k in range(PORTFOLIO_SIZE))
        sub_cov = self.cov_matrix[np.ix_(indices, indices)]
        vol = np.sqrt(max(weights @ sub_cov @ weights, 1e-10))
        sharpe = (ret - RISK_FREE_RATE) / max(vol, 1e-6)

        port_rets = np.zeros(self.min_return_len)
        for k in range(PORTFOLIO_SIZE):
            port_rets += weights[k] * self.stocks[indices[k]].returns[-self.min_return_len:]
        downside = port_rets[port_rets < 0]
        sortino = 0.0
        if len(downside) >= 2:
            dv = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)
            sortino = (ret - RISK_FREE_RATE) / max(dv, 1e-6)
        cum = np.cumprod(1 + port_rets)
        peak = np.maximum.accumulate(cum)
        mdd = np.max((peak - cum) / peak) if len(cum) > 0 else 0

        return PortfolioResult(
            stocks=[self.stocks[i].ticker for i in indices],
            weights=weights.tolist(),
            expected_return=ret,
            volatility=vol,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=mdd,
            fitness=self.fitness(x),
            profile=profile,
            algorithm=algo,
            generations=gen,
            stock_data=[self.stocks[i] for i in indices],
        )


# ════════════════════════════════════════════════════════════════
# Algorithm 1: Differential Evolution (DE)
# ════════════════════════════════════════════════════════════════

class DifferentialEvolution:
    """
    DE/current-to-best/1/bin strategy.

    Why DE > GA for portfolios:
    - Vector difference mutation is naturally suited for continuous weights
    - Only 2 params: F (mutation scale) and CR (crossover rate)
    - Faster convergence on continuous optimization
    - Less prone to premature convergence than GA

    Reference: Storn & Price (1997), Das & Suganthan (2011)
    """

    def __init__(self, engine: PortfolioEngine, pop_size=POP_SIZE,
                 max_iter=MAX_ITER, F=0.8, CR=0.9):
        self.engine = engine
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F       # Mutation scale factor
        self.CR = CR     # Crossover rate
        self.dim = 2 * PORTFOLIO_SIZE  # 5 indices + 5 weights

    def optimize(self, verbose=True) -> Tuple[np.ndarray, float, int]:
        if verbose:
            print(f"\n  🔬 Differential Evolution (F={self.F}, CR={self.CR})")

        # Initialize population
        pop = np.random.rand(self.pop_size, self.dim)
        fit = np.array([self.engine.fitness(p) for p in pop])

        best_idx = np.argmax(fit)
        best = pop[best_idx].copy()
        best_fit = fit[best_idx]
        stagnation = 0

        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                # Select 3 distinct random vectors (not i)
                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2 = random.sample(candidates, 2)

                # Mutation: current-to-best/1
                mutant = pop[i] + self.F * (best - pop[i]) + \
                         self.F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, 0, 1)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, self.dim - 1)
                for j in range(self.dim):
                    if random.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection (greedy)
                trial_fit = self.engine.fitness(trial)
                if trial_fit >= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit

            gen_best = np.max(fit)
            if gen_best > best_fit:
                best_fit = gen_best
                best = pop[np.argmax(fit)].copy()
                stagnation = 0
            else:
                stagnation += 1

            if verbose and (gen % 50 == 0 or gen == self.max_iter - 1):
                pct = (gen + 1) / self.max_iter
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                sys.stdout.write(
                    f"\r    [{bar}] {gen+1}/{self.max_iter} "
                    f"| Best: {best_fit:.4f} | Stag: {stagnation}   "
                )
                sys.stdout.flush()

            if stagnation > 120:
                if verbose:
                    print(f"\n    ⏹ Early stop gen {gen+1}")
                break

        if verbose:
            print()
        return best, best_fit, gen + 1


# ════════════════════════════════════════════════════════════════
# Algorithm 2: Particle Swarm Optimization (PSO)
# ════════════════════════════════════════════════════════════════

class ParticleSwarmOptimization:
    """
    PSO with constriction coefficient.

    Why PSO complements DE:
    - Velocity-based search explores different regions
    - Social learning (swarm intelligence) finds global optima
    - Good at escaping local optima through momentum
    - Constriction factor prevents velocity explosion

    Reference: Clerc & Kennedy (2002)
    """

    def __init__(self, engine: PortfolioEngine, pop_size=POP_SIZE,
                 max_iter=MAX_ITER, c1=2.05, c2=2.05):
        self.engine = engine
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = 2 * PORTFOLIO_SIZE

        # Constriction coefficient (Clerc & Kennedy)
        phi = c1 + c2  # 4.1
        self.chi = 2.0 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))  # ≈0.7298
        self.c1 = c1
        self.c2 = c2

    def optimize(self, verbose=True) -> Tuple[np.ndarray, float, int]:
        if verbose:
            print(f"\n  🐦 Particle Swarm Optimization (χ={self.chi:.4f})")

        # Initialize swarm
        pos = np.random.rand(self.pop_size, self.dim)
        vel = (np.random.rand(self.pop_size, self.dim) - 0.5) * 0.1
        fit = np.array([self.engine.fitness(p) for p in pos])

        # Personal best
        pbest = pos.copy()
        pbest_fit = fit.copy()

        # Global best
        gbest_idx = np.argmax(fit)
        gbest = pos[gbest_idx].copy()
        gbest_fit = fit[gbest_idx]
        stagnation = 0

        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = random.random(), random.random()

                # PSO velocity update with constriction
                vel[i] = self.chi * (
                    vel[i] +
                    self.c1 * r1 * (pbest[i] - pos[i]) +
                    self.c2 * r2 * (gbest - pos[i])
                )

                # Velocity clamping
                vel[i] = np.clip(vel[i], -0.3, 0.3)

                # Position update
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], 0, 1)

                # Evaluate
                new_fit = self.engine.fitness(pos[i])
                fit[i] = new_fit

                # Update personal best
                if new_fit > pbest_fit[i]:
                    pbest[i] = pos[i].copy()
                    pbest_fit[i] = new_fit

            # Update global best
            gen_best_idx = np.argmax(fit)
            if fit[gen_best_idx] > gbest_fit:
                gbest = pos[gen_best_idx].copy()
                gbest_fit = fit[gen_best_idx]
                stagnation = 0
            else:
                stagnation += 1

            if verbose and (gen % 50 == 0 or gen == self.max_iter - 1):
                pct = (gen + 1) / self.max_iter
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                sys.stdout.write(
                    f"\r    [{bar}] {gen+1}/{self.max_iter} "
                    f"| Best: {gbest_fit:.4f} | Stag: {stagnation}   "
                )
                sys.stdout.flush()

            if stagnation > 120:
                if verbose:
                    print(f"\n    ⏹ Early stop gen {gen+1}")
                break

        if verbose:
            print()
        return gbest, gbest_fit, gen + 1


# ════════════════════════════════════════════════════════════════
# Algorithm 3: Simulated Annealing (SA)
# ════════════════════════════════════════════════════════════════

class SimulatedAnnealing:
    """
    SA with adaptive cooling and reheating.

    Why SA adds value:
    - Can escape deep local optima (accepts worse solutions probabilistically)
    - Simple, robust, no population overhead
    - Reheat mechanism prevents getting stuck
    - Good for final refinement after DE/PSO

    Reference: Kirkpatrick et al. (1983)
    """

    def __init__(self, engine: PortfolioEngine, max_iter=MAX_ITER * POP_SIZE,
                 T_init=1.0, T_min=0.001, alpha=0.9995):
        self.engine = engine
        self.max_iter = max_iter
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.dim = 2 * PORTFOLIO_SIZE

    def optimize(self, verbose=True) -> Tuple[np.ndarray, float, int]:
        if verbose:
            print(f"\n  🌡️  Simulated Annealing (T₀={self.T_init}, α={self.alpha})")

        # Start from random solution
        current = np.random.rand(self.dim)
        current_fit = self.engine.fitness(current)
        best = current.copy()
        best_fit = current_fit

        T = self.T_init
        stagnation = 0
        total_evals = 0
        report_interval = self.max_iter // 10

        for i in range(self.max_iter):
            # Generate neighbor (Gaussian perturbation)
            sigma = 0.05 * (T / self.T_init)  # adaptive step size
            neighbor = current + np.random.normal(0, sigma, self.dim)
            neighbor = np.clip(neighbor, 0, 1)

            neighbor_fit = self.engine.fitness(neighbor)
            total_evals += 1

            delta = neighbor_fit - current_fit

            # Accept better solutions always, worse with probability
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                current = neighbor
                current_fit = neighbor_fit

            if current_fit > best_fit:
                best = current.copy()
                best_fit = current_fit
                stagnation = 0
            else:
                stagnation += 1

            # Cool down
            T *= self.alpha
            if T < self.T_min:
                T = self.T_min

            # Reheat if stagnant
            if stagnation > 5000:
                T = self.T_init * 0.3
                stagnation = 0

            if verbose and (i % report_interval == 0 or i == self.max_iter - 1):
                pct = (i + 1) / self.max_iter
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                sys.stdout.write(
                    f"\r    [{bar}] {i+1}/{self.max_iter} "
                    f"| Best: {best_fit:.4f} | T: {T:.6f}   "
                )
                sys.stdout.flush()

        if verbose:
            print()
        return best, best_fit, total_evals


# ════════════════════════════════════════════════════════════════
# Ensemble Runner
# ════════════════════════════════════════════════════════════════

def run_ensemble(
    engine: PortfolioEngine,
    profile: str,
    verbose=True,
    algorithms: Optional[List[str]] = None,
    pop_size: int = POP_SIZE,
    max_iter: int = MAX_ITER,
    sa_iter: Optional[int] = None,
) -> Dict[str, PortfolioResult]:
    """Run selected algorithms and return comparable portfolio results."""
    selected_algorithms = [a.upper() for a in (algorithms or ["DE", "PSO", "SA"])]

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  🏆 ENSEMBLE OPTIMIZATION — {profile}")
        print(f"{'═'*65}")
        print(f"  Running {len(selected_algorithms)} algorithms to find optimal portfolio...\n")

    results = {}
    t0 = time.time()

    # 1. Differential Evolution
    if "DE" in selected_algorithms:
        de = DifferentialEvolution(engine, pop_size=pop_size, max_iter=max_iter)
        de_x, de_fit, de_gen = de.optimize(verbose)
        results["DE"] = engine.build_result(de_x, "DE", de_gen, profile)

    # 2. Particle Swarm Optimization
    if "PSO" in selected_algorithms:
        pso = ParticleSwarmOptimization(engine, pop_size=pop_size, max_iter=max_iter)
        pso_x, pso_fit, pso_gen = pso.optimize(verbose)
        results["PSO"] = engine.build_result(pso_x, "PSO", pso_gen, profile)

    # 3. Simulated Annealing (fewer iterations for speed)
    if "SA" in selected_algorithms:
        sa = SimulatedAnnealing(engine, max_iter=sa_iter or max_iter * pop_size)
        sa_x, sa_fit, sa_gen = sa.optimize(verbose)
        results["SA"] = engine.build_result(sa_x, "SA", sa_gen, profile)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  ⏱  Total: {elapsed:.1f}s\n")

    return results


def best_portfolio_result(results: Dict[str, PortfolioResult]) -> Optional[PortfolioResult]:
    if not results:
        return None
    return max(results.values(), key=lambda r: r.fitness)


def portfolio_allocation_frame(
    result: PortfolioResult,
    capital: float = 100_000_000,
) -> pd.DataFrame:
    rows = []
    for stock, weight, stock_data in zip(result.stocks, result.weights, result.stock_data):
        allocation = capital * weight
        lot = int(allocation / (stock_data.price * 100)) if stock_data.price > 0 else 0
        rows.append({
            "Ticker": stock,
            "Bobot%": round(weight * 100, 2),
            "Harga": round(stock_data.price, 0),
            "Alokasi_Rp": round(allocation, 0),
            "Lot": lot,
            "Return_Annual%": round(stock_data.annual_return * 100, 2),
            "Vol_Annual%": round(stock_data.annual_vol * 100, 2),
            "MaxDD%": round(stock_data.max_drawdown * 100, 2),
            "MoS%": round(stock_data.mos_pct, 2),
            "Score": round(stock_data.val_score, 2),
            "Prob_Naik%": round(stock_data.prob_up, 2),
        })
    return pd.DataFrame(rows)


def algorithm_comparison_frame(results: Dict[str, PortfolioResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append({
            "Algorithm": name,
            "Saham": ", ".join(result.stocks),
            "Return%": round(result.expected_return * 100, 2),
            "Volatility%": round(result.volatility * 100, 2),
            "Sharpe": round(result.sharpe, 3),
            "Sortino": round(result.sortino, 3),
            "MaxDD%": round(result.max_drawdown * 100, 2),
            "Fitness": round(result.fitness, 4),
        })
    return pd.DataFrame(rows).sort_values("Fitness", ascending=False) if rows else pd.DataFrame()


# ════════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════════

def display_comparison(all_results: Dict[str, PortfolioResult]):
    """Show comparison table of all algorithms."""
    rows = []
    for name, r in all_results.items():
        rows.append({
            "Algorithm": f"{'🏆 ' if r.fitness == max(x.fitness for x in all_results.values()) else '   '}{name}",
            "Stocks": ", ".join(r.stocks),
            "Return%": round(r.expected_return * 100, 1),
            "Vol%": round(r.volatility * 100, 1),
            "Sharpe": round(r.sharpe, 3),
            "Sortino": round(r.sortino, 3),
            "MaxDD%": round(r.max_drawdown * 100, 1),
            "Fitness": round(r.fitness, 4),
        })

    df = pd.DataFrame(rows)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="rounded_grid",
                       showindex=False, floatfmt=".1f"))
    except ImportError:
        print(df.to_string(index=False))


def display_best(result: PortfolioResult):
    """Display best portfolio in detail."""
    p = result
    emoji = {"SAFE": "🛡️", "MODERATE": "⚖️", "AGGRESSIVE": "🚀"}.get(p.profile, "📊")

    print(f"\n{'═'*65}")
    print(f"  {emoji} BEST PORTFOLIO — {p.profile} (by {p.algorithm})")
    print(f"{'═'*65}\n")

    rows = []
    for i in range(len(p.stocks)):
        s = p.stock_data[i]
        rows.append({
            "Ticker": s.ticker,
            "Weight%": round(p.weights[i] * 100, 1),
            "Harga": round(s.price, 0),
            "Return%": round(s.annual_return * 100, 1),
            "Vol%": round(s.annual_vol * 100, 1),
            "MaxDD%": round(s.max_drawdown * 100, 1),
            "MoS%": round(s.mos_pct, 1),
            "Score": round(s.val_score, 1),
            "Prob⬆%": round(s.prob_up, 1),
        })

    df = pd.DataFrame(rows)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="rounded_grid",
                       showindex=False, floatfmt=".1f"))
    except ImportError:
        print(df.to_string(index=False))

    print(f"\n  {'─'*50}")
    print(f"  📊 PORTFOLIO METRICS")
    print(f"  {'─'*50}")
    print(f"     Algorithm        : {p.algorithm} ({p.generations} iterations)")
    print(f"     Expected Return  : {p.expected_return*100:+.2f}%/thn")
    print(f"     Volatility       : {p.volatility*100:.2f}%/thn")
    print(f"     Sharpe Ratio     : {p.sharpe:.3f}")
    print(f"     Sortino Ratio    : {p.sortino:.3f}")
    print(f"     Max Drawdown     : {p.max_drawdown*100:.2f}%")
    print(f"     Fitness Score    : {p.fitness:.4f}")

    # Weight bar chart
    print(f"\n  {'─'*50}")
    print(f"  📊 ALOKASI BOBOT")
    print(f"  {'─'*50}")
    for i in range(len(p.stocks)):
        w = p.weights[i] * 100
        bl = int(w / 2)
        bar = "█" * bl + "░" * (25 - bl)
        print(f"     {p.stocks[i]:>6} [{bar}] {w:.1f}%")

    # Risk assessment
    print(f"\n  {'─'*50}")
    print(f"  🎯 RISK ASSESSMENT")
    print(f"  {'─'*50}")
    vr = "🟢 Rendah" if p.volatility < 0.15 else ("🟡 Sedang" if p.volatility < 0.25 else "🔴 Tinggi")
    dr = "🟢 Aman" if p.max_drawdown < 0.10 else ("🟡 Wajar" if p.max_drawdown < 0.20 else "🔴 Beresiko")
    sr = "🟢 Sangat Baik" if p.sharpe > 1.5 else ("🟡 Cukup" if p.sharpe > 0.5 else "🔴 Kurang")
    print(f"     Volatilitas : {vr}")
    print(f"     Drawdown    : {dr}")
    print(f"     Sharpe      : {sr}")

    # Investment simulation
    print(f"\n  {'─'*50}")
    print(f"  💰 SIMULASI INVESTASI (Modal Rp 100.000.000)")
    print(f"  {'─'*50}")
    modal = 100_000_000
    for i in range(len(p.stocks)):
        alokasi = modal * p.weights[i]
        lot = int(alokasi / (p.stock_data[i].price * 100))
        print(f"     {p.stocks[i]:>6}: Rp {alokasi:>14,.0f} ({p.weights[i]*100:.1f}%) = {lot} lot")
    print(f"     {'─'*40}")
    exp = modal * p.expected_return
    risk = modal * p.max_drawdown
    print(f"     Expected Profit: Rp {exp:>12,.0f} ({p.expected_return*100:+.1f}%/thn)")
    print(f"     Max Risk Loss  : Rp {risk:>12,.0f} (-{p.max_drawdown*100:.1f}%)")
    print(f"  {'─'*50}\n")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("""
╔═══════════════════════════════════════════════════════╗
║   🏆 Portfolio Optimizer — Ensemble (DE + PSO + SA)   ║
║   3 algorithms compete → best result wins             ║
╚═══════════════════════════════════════════════════════╝
    """)

    # Load data
    tickers, val_data = load_valuation_csv(CSV_FILE)

    # Fetch history
    stocks = fetch_all(tickers, val_data)
    if len(stocks) < PORTFOLIO_SIZE:
        print(f"  ❌ Butuh min {PORTFOLIO_SIZE} stocks, dapat {len(stocks)}")
        return

    # Run for each profile
    profiles = ["SAFE", "MODERATE", "AGGRESSIVE"]
    all_best = {}

    for profile in profiles:
        engine = PortfolioEngine(stocks, profile)
        algo_results = run_ensemble(engine, profile)

        # Show algorithm comparison
        print(f"\n  📊 Perbandingan Algoritma ({profile}):")
        display_comparison(algo_results)

        # Pick best
        best_name = max(algo_results, key=lambda k: algo_results[k].fitness)
        best = algo_results[best_name]
        all_best[profile] = best

        print(f"\n  🏆 Winner: {best_name} (fitness={best.fitness:.4f})")
        display_best(best)

    # Final comparison across profiles
    print(f"\n{'═'*70}")
    print(f"  🏆 RINGKASAN FINAL — Best Portfolio per Profile")
    print(f"{'═'*70}\n")

    final_rows = []
    for profile, r in all_best.items():
        em = {"SAFE": "🛡️", "MODERATE": "⚖️", "AGGRESSIVE": "🚀"}[profile]
        final_rows.append({
            "Profile": f"{em} {profile}",
            "Algorithm": r.algorithm,
            "Stocks": ", ".join(r.stocks),
            "Return%": round(r.expected_return * 100, 1),
            "Vol%": round(r.volatility * 100, 1),
            "Sharpe": round(r.sharpe, 3),
            "MaxDD%": round(r.max_drawdown * 100, 1),
        })

    fdf = pd.DataFrame(final_rows)
    try:
        from tabulate import tabulate
        print(tabulate(fdf, headers="keys", tablefmt="rounded_grid",
                       showindex=False, floatfmt=".1f"))
    except ImportError:
        print(fdf.to_string(index=False))

    print(f"\n  📝 Catatan:")
    print(f"  - Data historis: {HIST_PERIOD} | Kandidat: hanya SANGAT MURAH")
    print(f"  - Past performance ≠ future results")
    print(f"  - Rebalancing disarankan setiap 3-6 bulan")
    print(f"  - DE biasanya menang untuk weight optimization")
    print(f"  - PSO unggul di exploration, SA di escaping local optima\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
