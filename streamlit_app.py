from __future__ import annotations

import importlib.util
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from valuation import ValueAnalyzer


st.set_page_config(page_title="SahamDiskon", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_TICKERS = "BBCA, BBRI, BMRI, TLKM, ASII, ICBP, UNTR, INDF, KLBF, ADRO, ANTM, MDKA"


def load_optimizer_module():
    module_path = APP_DIR / "stock optimization.py"
    spec = importlib.util.spec_from_file_location("stock_optimization", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa memuat {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


optimizer = load_optimizer_module()


def parse_tickers(raw: str) -> List[str]:
    seen = set()
    tickers = []
    for item in re.split(r"[\s,;]+", raw.upper().strip()):
        ticker = item.replace(".JK", "").strip()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def clean_number(value):
    if value is None:
        return None
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
    except (TypeError, ValueError):
        pass
    return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def analyze_single_stock(ticker: str, regime: str) -> Dict:
    ValueAnalyzer.set_regime(regime)
    result = ValueAnalyzer().analyze_one(ticker)
    return asdict(result)


def build_valuation_frame(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append({
            "Ticker": str(record.get("ticker", "")).replace(".JK", ""),
            "Sektor_DCF": record.get("sector_template"),
            "WACC_%": clean_number(record.get("wacc_used")),
            "Terminal_g_%": clean_number(record.get("terminal_g_used")),
            "Harga": clean_number(record.get("price")),
            "Graham": clean_number(record.get("graham")),
            "EPV": clean_number(record.get("epv")),
            "DCF": clean_number(record.get("dcf")),
            "RIM": clean_number(record.get("rim")),
            "Intrinsic": clean_number(record.get("intrinsic")),
            "MoS%": clean_number(record.get("mos_pct")),
            "Score": clean_number(record.get("score")),
            "Prob_Naik%": clean_number(record.get("prob_up_pct")),
            "Confidence": record.get("confidence"),
            "Data_Quality": record.get("data_quality"),
            "Status": record.get("status"),
            "Error": record.get("error"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Error", "Prob_Naik%", "Score"], ascending=[True, False, False], na_position="last")
    return df


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_stock_history(ticker: str, mos: float, score: float, prob: float, conf: str, hist_period: str):
    val_info = {"mos": mos, "score": score, "prob": prob, "conf": conf}
    return optimizer.fetch_stock(ticker, val_info, hist_period=hist_period)


def collect_stock_history(tickers: List[str], val_data: Dict[str, dict], hist_period: str):
    stocks = []
    progress = st.progress(0)
    status = st.empty()

    for idx, ticker in enumerate(tickers, 1):
        info = val_data.get(ticker, {})
        status.write(f"Mengambil histori harga {ticker} ({idx}/{len(tickers)})")
        stock = fetch_stock_history(
            ticker,
            float(info.get("mos", 0.0)),
            float(info.get("score", 0.0)),
            float(info.get("prob", 50.0)),
            str(info.get("conf", "D")),
            hist_period,
        )
        if stock is not None:
            stocks.append(stock)
        progress.progress(idx / len(tickers))
        time.sleep(0.05)

    status.empty()
    progress.empty()
    return stocks


def idr(value: float) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")


def show_valuation(df: pd.DataFrame):
    if df.empty:
        st.warning("Belum ada hasil valuasi.")
        return

    valid = df[df["Error"].isna()]
    errors = df[df["Error"].notna()]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Saham valid", len(valid))
    metric_cols[1].metric("Rata-rata Prob Naik", f"{valid['Prob_Naik%'].mean():.1f}%" if not valid.empty else "-")
    metric_cols[2].metric("Rata-rata MoS", f"{valid['MoS%'].mean():.1f}%" if not valid.empty else "-")
    metric_cols[3].metric("Gagal", len(errors))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Top Probabilitas Naik")
        chart_df = valid.nlargest(min(12, len(valid)), "Prob_Naik%").set_index("Ticker")
        if not chart_df.empty:
            st.bar_chart(chart_df["Prob_Naik%"])
    with chart_cols[1]:
        st.caption("Top Margin of Safety")
        chart_df = valid.nlargest(min(12, len(valid)), "MoS%").set_index("Ticker")
        if not chart_df.empty:
            st.bar_chart(chart_df["MoS%"])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="%.0f"),
            "Intrinsic": st.column_config.NumberColumn(format="%.0f"),
            "MoS%": st.column_config.NumberColumn(format="%.2f%%"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Prob_Naik%": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def show_portfolio(result, all_results, capital: float):
    comparison = optimizer.algorithm_comparison_frame(all_results)
    allocation = optimizer.portfolio_allocation_frame(result, capital=capital)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Algorithm", result.algorithm)
    metric_cols[1].metric("Return/thn", f"{result.expected_return * 100:.2f}%")
    metric_cols[2].metric("Volatility", f"{result.volatility * 100:.2f}%")
    metric_cols[3].metric("Sharpe", f"{result.sharpe:.3f}")
    metric_cols[4].metric("Max Drawdown", f"{result.max_drawdown * 100:.2f}%")

    st.caption("Perbandingan Algoritma")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.caption("Alokasi Portofolio")
    st.dataframe(
        allocation,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Bobot%": st.column_config.NumberColumn(format="%.2f%%"),
            "Harga": st.column_config.NumberColumn(format="%.0f"),
            "Alokasi_Rp": st.column_config.NumberColumn(format="%.0f"),
            "Return_Annual%": st.column_config.NumberColumn(format="%.2f%%"),
            "Vol_Annual%": st.column_config.NumberColumn(format="%.2f%%"),
            "MaxDD%": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    weight_chart = allocation.set_index("Ticker")["Bobot%"]
    st.bar_chart(weight_chart)

    invested = allocation["Alokasi_Rp"].sum()
    expected_profit = capital * result.expected_return
    max_risk = capital * result.max_drawdown
    st.caption(f"Modal simulasi: {idr(capital)} | Terpakai: {idr(invested)} | Expected profit: {idr(expected_profit)} | Risiko drawdown: {idr(max_risk)}")


st.title("SahamDiskon")
st.caption("Valuasi fundamental dan optimasi portofolio IDX dalam satu alur.")

with st.sidebar:
    st.header("Input")
    ticker_text = st.text_area("Ticker IDX", value=DEFAULT_TICKERS, height=140)
    regime = st.selectbox("Regime valuasi", ["conservative", "balanced", "aggressive"], index=0)
    run_valuation = st.button("Jalankan Valuasi", type="primary", use_container_width=True)

    st.divider()
    st.header("Optimasi")
    candidate_filter = st.selectbox("Kandidat", ["SEMUA", "SANGAT MURAH", "MURAH", "WAJAR"], index=0)
    profile = st.selectbox("Profil risiko", ["MODERATE", "SAFE", "AGGRESSIVE"], index=0)
    hist_period = st.selectbox("Histori harga", ["6mo", "1y", "2y", "5y"], index=1)
    algorithms = st.multiselect("Algoritma", ["DE", "PSO", "SA"], default=["DE", "PSO"])
    pop_size = st.slider("Populasi", min_value=30, max_value=200, value=80, step=10)
    max_iter = st.slider("Iterasi", min_value=50, max_value=500, value=150, step=25)
    capital = st.number_input("Modal simulasi", min_value=1_000_000, value=100_000_000, step=1_000_000)
    seed = st.number_input("Seed optimizer", min_value=1, max_value=9999, value=42)
    run_optimizer = st.button("Jalankan Optimasi", use_container_width=True)

    if st.button("Clear cache data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("valuation_df", None)
        st.session_state.pop("portfolio_results", None)
        st.rerun()


tab_valuation, tab_optimizer = st.tabs(["Valuasi", "Optimasi Portofolio"])

if run_valuation:
    tickers = parse_tickers(ticker_text)
    if not tickers:
        st.warning("Masukkan minimal satu ticker.")
    else:
        records = []
        progress = st.progress(0)
        status = st.empty()
        for idx, ticker in enumerate(tickers, 1):
            status.write(f"Menganalisis {ticker} ({idx}/{len(tickers)})")
            records.append(analyze_single_stock(ticker, regime))
            progress.progress(idx / len(tickers))
        status.empty()
        progress.empty()
        st.session_state["valuation_df"] = build_valuation_frame(records)
        st.session_state.pop("portfolio_results", None)

valuation_df = st.session_state.get("valuation_df", pd.DataFrame())

with tab_valuation:
    show_valuation(valuation_df)

with tab_optimizer:
    if valuation_df.empty:
        st.info("Jalankan valuasi dulu untuk membuat kandidat portofolio.")
    else:
        valid_df = valuation_df[valuation_df["Error"].isna()].copy()
        tickers, val_data, candidate_df = optimizer.load_valuation_dataframe(
            valid_df,
            status_filter=candidate_filter,
            top_n=25,
        )

        st.caption(f"Kandidat optimizer: {len(tickers)} saham")
        st.dataframe(candidate_df, use_container_width=True, hide_index=True)

        if run_optimizer:
            if not algorithms:
                st.warning("Pilih minimal satu algoritma.")
            elif len(tickers) < optimizer.PORTFOLIO_SIZE:
                st.warning(f"Butuh minimal {optimizer.PORTFOLIO_SIZE} kandidat, saat ini hanya {len(tickers)}.")
            else:
                random.seed(int(seed))
                np.random.seed(int(seed))
                with st.spinner("Mengambil histori harga kandidat..."):
                    stocks = collect_stock_history(tickers, val_data, hist_period)

                if len(stocks) < optimizer.PORTFOLIO_SIZE:
                    st.warning(f"Histori harga valid hanya {len(stocks)} saham. Perlu minimal {optimizer.PORTFOLIO_SIZE}.")
                else:
                    with st.spinner("Menjalankan optimizer portofolio..."):
                        engine = optimizer.PortfolioEngine(stocks, profile)
                        results = optimizer.run_ensemble(
                            engine,
                            profile,
                            verbose=False,
                            algorithms=algorithms,
                            pop_size=int(pop_size),
                            max_iter=int(max_iter),
                        )
                    st.session_state["portfolio_results"] = results

        portfolio_results = st.session_state.get("portfolio_results")
        if portfolio_results:
            best = optimizer.best_portfolio_result(portfolio_results)
            if best is not None:
                show_portfolio(best, portfolio_results, float(capital))
