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
DEFAULT_SINGLE_TICKER = "BBCA"
ANALYSIS_TYPES = ["Individual", "Portfolio"]
REGIMES = ["conservative", "balanced", "aggressive"]
HIST_PERIODS = ["6mo", "1y", "2y", "3y", "4y", "5y"]
HIST_PERIOD_LABELS = {
    "6mo": "6 bulan",
    "1y": "1 tahun",
    "2y": "2 tahun",
    "3y": "3 tahun",
    "4y": "4 tahun",
    "5y": "5 tahun",
}
INDIVIDUAL_DATA_SCOPES = ["Saham tunggal", "Watchlist default", "Daftar ticker custom"]
PORTFOLIO_DATA_SCOPES = ["Watchlist default", "Daftar ticker custom"]


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


def format_hist_period(period: str) -> str:
    return HIST_PERIOD_LABELS.get(period, period)


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


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_close_history(tickers: List[str], hist_period: str) -> pd.DataFrame:
    frames = []

    for ticker in tickers:
        try:
            hist = optimizer.load_price_history(ticker, hist_period)
        except Exception:
            continue

        if hist.empty or "Close" not in hist.columns:
            continue

        clean_ticker = ticker.replace(".JK", "")
        frames.append(hist["Close"].dropna().rename(clean_ticker))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


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


def analysis_page_name(analysis_type=None) -> str:
    selected = analysis_type or st.session_state.get("analysis_type", "Individual")
    return "Analisis Portofolio" if selected == "Portfolio" else "Analisis Individual"


def extract_valuation_data(tickers: List[str], regime: str) -> pd.DataFrame:
    records = []
    progress = st.progress(0)
    status = st.empty()

    for idx, ticker in enumerate(tickers, 1):
        status.write(f"Mengekstrak dan menganalisis {ticker} ({idx}/{len(tickers)})")
        records.append(analyze_single_stock(ticker, regime))
        progress.progress(idx / len(tickers))

    status.empty()
    progress.empty()
    return build_valuation_frame(records)


def ticker_text_for_scope(data_scope: str, single_ticker: str, custom_tickers: str) -> str:
    if data_scope == "Saham tunggal":
        return single_ticker
    if data_scope == "Watchlist default":
        return DEFAULT_TICKERS
    return custom_tickers


def show_extraction_summary():
    config = st.session_state.get("extraction_config")
    if not config:
        return

    summary_cols = st.columns(5)
    summary_cols[0].metric("Tipe analisis", config["analysis_type"])
    summary_cols[1].metric("Cakupan data", config["data_scope"])
    summary_cols[2].metric("Jumlah ticker", config["ticker_count"])
    summary_cols[3].metric("Regime", config["regime"])
    summary_cols[4].metric("Histori harga", format_hist_period(config.get("hist_period", "1y")))


def show_extraction_page():
    st.subheader("Ekstraksi Data")

    current_type = st.session_state.get("analysis_type", "Individual")
    type_index = ANALYSIS_TYPES.index(current_type) if current_type in ANALYSIS_TYPES else 0
    analysis_type = st.radio(
        "Tipe analisis",
        ANALYSIS_TYPES,
        index=type_index,
        horizontal=True,
        key="analysis_type_selector",
    )

    scope_options = PORTFOLIO_DATA_SCOPES if analysis_type == "Portfolio" else INDIVIDUAL_DATA_SCOPES
    previous_scope = st.session_state.get("data_scope", scope_options[0])
    scope_index = scope_options.index(previous_scope) if previous_scope in scope_options else 0

    setup_cols = st.columns(3)
    with setup_cols[0]:
        data_scope = st.selectbox("Cakupan data", scope_options, index=scope_index, key="data_scope_selector")
    with setup_cols[1]:
        regime = st.selectbox("Regime valuasi", REGIMES, index=0, key="regime_selector")
    with setup_cols[2]:
        default_hist = st.session_state.get("hist_period", "1y")
        hist_index = HIST_PERIODS.index(default_hist) if default_hist in HIST_PERIODS else 1
        hist_period = st.selectbox(
            "Cakupan histori harga",
            HIST_PERIODS,
            index=hist_index,
            format_func=format_hist_period,
            key="hist_period_selector",
        )

    with st.form("extraction_form", clear_on_submit=False):
        if data_scope == "Saham tunggal":
            single_ticker = st.text_input("Ticker IDX", value=DEFAULT_SINGLE_TICKER, key="single_ticker_input")
            custom_tickers = st.session_state.get("custom_ticker_text", DEFAULT_TICKERS)
        elif data_scope == "Watchlist default":
            single_ticker = st.session_state.get("single_ticker_input", DEFAULT_SINGLE_TICKER)
            custom_tickers = st.text_area(
                "Ticker IDX",
                value=DEFAULT_TICKERS,
                height=140,
                disabled=True,
                help="Watchlist bawaan untuk cakupan awal.",
            )
        else:
            single_ticker = st.session_state.get("single_ticker_input", DEFAULT_SINGLE_TICKER)
            custom_tickers = st.text_area(
                "Ticker IDX",
                value=st.session_state.get("custom_ticker_text", DEFAULT_TICKERS),
                height=140,
                key="custom_ticker_text",
                help="Pisahkan ticker dengan koma, spasi, titik koma, atau baris baru.",
            )

        submitted = st.form_submit_button("Ekstrak Data", type="primary", use_container_width=True)

    if not submitted:
        return

    ticker_text = ticker_text_for_scope(data_scope, single_ticker, custom_tickers)
    tickers = parse_tickers(ticker_text)

    if not tickers:
        st.warning("Masukkan minimal satu ticker.")
        return
    if analysis_type == "Portfolio" and len(tickers) < optimizer.PORTFOLIO_SIZE:
        st.warning(f"Analisis portofolio butuh minimal {optimizer.PORTFOLIO_SIZE} ticker kandidat.")
        return

    valuation_df = extract_valuation_data(tickers, regime)
    st.session_state["valuation_df"] = valuation_df
    st.session_state.pop("portfolio_results", None)
    st.session_state["analysis_type"] = analysis_type
    st.session_state["data_scope"] = data_scope
    st.session_state["hist_period"] = hist_period
    st.session_state["extraction_config"] = {
        "analysis_type": analysis_type,
        "data_scope": data_scope,
        "ticker_count": len(tickers),
        "tickers": tickers,
        "regime": regime,
        "hist_period": hist_period,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state["active_page"] = analysis_page_name(analysis_type)
    st.rerun()


def show_individual_price_history(valuation_df: pd.DataFrame):
    config = st.session_state.get("extraction_config", {})
    hist_period = config.get("hist_period", st.session_state.get("hist_period", "1y"))
    tickers = config.get("tickers") or valuation_df["Ticker"].dropna().astype(str).tolist()

    if not tickers:
        return

    st.caption(f"Histori harga: {format_hist_period(hist_period)}")
    with st.spinner("Mengambil histori harga..."):
        price_df = load_close_history(tickers, hist_period)

    price_df = price_df.ffill().dropna(axis=1, how="all")
    if price_df.empty:
        st.warning("Histori harga tidak tersedia untuk cakupan yang dipilih.")
        return

    st.line_chart(price_df)


def show_individual_page():
    st.subheader("Analisis Individual")
    valuation_df = st.session_state.get("valuation_df", pd.DataFrame())

    if valuation_df.empty:
        st.info("Mulai dari page Ekstraksi Data untuk memilih cakupan saham.")
        return

    show_extraction_summary()
    st.divider()
    show_individual_price_history(valuation_df)
    st.divider()
    show_valuation(valuation_df)


def show_portfolio_page():
    st.subheader("Analisis Portofolio")
    valuation_df = st.session_state.get("valuation_df", pd.DataFrame())

    if valuation_df.empty:
        st.info("Mulai dari page Ekstraksi Data untuk membuat kandidat portofolio.")
        return

    show_extraction_summary()
    st.divider()

    with st.form("optimizer_form", clear_on_submit=False):
        action_cols = st.columns([1, 2])
        with action_cols[0]:
            run_optimizer = st.form_submit_button("Jalankan Optimasi", type="primary", use_container_width=True)
        with action_cols[1]:
            st.caption("Optimasi memakai hasil ekstraksi yang valid, mengambil histori harga, lalu membandingkan algoritma portofolio.")

        filter_cols = st.columns(4)
        with filter_cols[0]:
            candidate_filter = st.selectbox("Kandidat", ["SEMUA", "SANGAT MURAH", "MURAH", "WAJAR"], index=0)
        with filter_cols[1]:
            profile = st.selectbox("Profil risiko", ["MODERATE", "SAFE", "AGGRESSIVE"], index=0)
        with filter_cols[2]:
            default_hist = st.session_state.get("hist_period", "1y")
            hist_index = HIST_PERIODS.index(default_hist) if default_hist in HIST_PERIODS else 1
            hist_period = st.selectbox(
                "Histori harga",
                HIST_PERIODS,
                index=hist_index,
                format_func=format_hist_period,
            )
        with filter_cols[3]:
            capital = st.number_input("Modal simulasi", min_value=1_000_000, value=100_000_000, step=1_000_000)

        algo_cols = st.columns([2, 1, 1, 1])
        with algo_cols[0]:
            algorithms = st.multiselect("Algoritma", ["DE", "PSO", "SA"], default=["DE", "PSO"])
        with algo_cols[1]:
            pop_size = st.slider("Populasi", min_value=30, max_value=200, value=80, step=10)
        with algo_cols[2]:
            max_iter = st.slider("Iterasi", min_value=50, max_value=500, value=150, step=25)
        with algo_cols[3]:
            seed = st.number_input("Seed optimizer", min_value=1, max_value=9999, value=42)

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


def render_sidebar():
    valuation_df = st.session_state.get("valuation_df", pd.DataFrame())
    cached_portfolio = st.session_state.get("portfolio_results")
    has_extraction = not valuation_df.empty
    target_page = analysis_page_name()

    if not has_extraction and st.session_state.get("active_page") != "Ekstraksi Data":
        st.session_state["active_page"] = "Ekstraksi Data"

    with st.sidebar:
        st.header("Navigasi")

        extraction_label = "[Aktif] Ekstraksi Data" if st.session_state.get("active_page") == "Ekstraksi Data" else "Ekstraksi Data"
        if st.button(extraction_label, use_container_width=True, key="nav_extraction"):
            st.session_state["active_page"] = "Ekstraksi Data"
            st.rerun()

        analysis_label = f"[Aktif] {target_page}" if st.session_state.get("active_page") == target_page else target_page
        if st.button(analysis_label, use_container_width=True, disabled=not has_extraction, key="nav_analysis"):
            st.session_state["active_page"] = target_page
            st.rerun()

        st.divider()
        st.header("Status")
        st.metric("Hasil ekstraksi", 0 if valuation_df.empty else len(valuation_df))
        st.metric("Hasil optimasi", 0 if not cached_portfolio else len(cached_portfolio))

        config = st.session_state.get("extraction_config")
        if config:
            st.caption(f"Terakhir: {config['analysis_type']} | {config['ticker_count']} ticker | {config['extracted_at']}")

        st.divider()
        if st.button("Clear cache data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("valuation_df", None)
            st.session_state.pop("portfolio_results", None)
            st.session_state.pop("extraction_config", None)
            st.session_state["active_page"] = "Ekstraksi Data"
            st.rerun()


def main():
    st.session_state.setdefault("active_page", "Ekstraksi Data")
    st.session_state.setdefault("analysis_type", "Individual")
    st.session_state.setdefault("data_scope", "Saham tunggal")
    st.session_state.setdefault("hist_period", "1y")

    st.title("SahamDiskon")
    st.caption("Valuasi fundamental dan optimasi portofolio IDX dalam satu alur.")

    render_sidebar()

    active_page = st.session_state.get("active_page", "Ekstraksi Data")
    if active_page == "Analisis Individual":
        show_individual_page()
    elif active_page == "Analisis Portofolio":
        show_portfolio_page()
    else:
        show_extraction_page()


main()
