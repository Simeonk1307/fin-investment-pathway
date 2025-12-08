import sys
import os

# Ensure repo root is on sys.path so `from src...` imports work regardless
# Calculate repo root robustly whether this script is in repo root or in src/
this_dir = os.path.dirname(__file__)
if os.path.basename(this_dir) == 'src':
    repo_root = os.path.dirname(this_dir)
else:
    repo_root = this_dir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*missing ScriptRunContext.*"
)

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta, time as dtime

from src.main import orchestrate_signal, backtest_strategy, get_agent_insights, fetch_hist_data


SAMPLE_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","PG","MA","UNH","HD","BAC","XOM","CVX","CSCO",
    "PFE","INTC","KO","PEP","ADBE","CRM","ORCL","NKE","MCD","T",
    "SBUX","AMD","ZM","UBER","LYFT","SQ","PYPL","BA","GS","MMM",
]


def sidebar_inputs():
    st.sidebar.header("Portfolio Setup")
    tickers = st.sidebar.multiselect("Choose stocks (up to 40)", SAMPLE_TICKERS, default=["AAPL","MSFT"])
    capital = st.sidebar.number_input("Capital per stock (USD)", min_value=1, value=1000)
    st.sidebar.markdown("---")
    st.sidebar.header("Prediction Window")
    start_time = st.sidebar.time_input("Start time (UTC)", dtime(0, 0))
    end_time = st.sidebar.time_input("End time (UTC)", dtime(23, 59))
    if start_time >= end_time:
        st.sidebar.error("Start time must be before end time")
    return tickers, capital, start_time, end_time


def configuration_panel():
    st.header("Configuration")
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    if 'capital_per_stock' not in st.session_state:
        st.session_state.capital_per_stock = 1000
    with st.form('config_form'):
        tickers = st.multiselect('Select tickers (up to 40)', SAMPLE_TICKERS, default=st.session_state.selected_tickers)
        capital = st.number_input('Capital per stock (USD)', min_value=1, value=st.session_state.capital_per_stock)
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input('Start time (UTC)', dtime(0,0))
        with col2:
            end_time = st.time_input('End time (UTC)', dtime(23,59))
        submit = st.form_submit_button('Save & Continue')
        if submit:
            st.session_state.selected_tickers = tickers
            st.session_state.capital_per_stock = capital
            st.session_state.start_time = start_time
            st.session_state.end_time = end_time
            st.success('Configuration saved — navigate to tabs below')


def live_market_view(selected):
    st.header("Live Market")
    # Manual refresh button (increments session key to trigger rerun)
    if 'refresh_key' not in st.session_state:
        st.session_state.refresh_key = 0
    if st.button("Refresh now"):
        st.session_state.refresh_key += 1

    # Auto-refresh toggle: uses small JS snippet to reload page every interval.
    interval_s = 5
    auto = st.checkbox("Auto-refresh every 5s", value=True)
    if auto:
        try:
            import streamlit.components.v1 as components
            components.html(f"<script>setTimeout(()=>window.location.reload(), {interval_s*1000});</script>", height=0)
        except Exception:
            # if components not available, fall back to no auto-refresh
            pass

    for t in selected:
        df = fetch_hist_data(t, period="7d", interval="1m")
        if df is None or df.empty:
            st.warning(f"No data for {t}")
            continue
        latest = df.iloc[-1]
        signal = orchestrate_signal(t, df)
        ind = signal.get('indicators', {}) or {}
        # Compute confluence: bullish components
        bullish = 0
        bearish = 0
        # SMA checks
        if ind.get('latest') and ind.get('sma20'):
            if ind['latest'] > ind['sma20']:
                bullish += 1
            else:
                bearish += 1
        if ind.get('sma20') and ind.get('sma50'):
            if ind['sma20'] > ind['sma50']:
                bullish += 1
            else:
                bearish += 1
        # MACD
        if 'macd' in ind and 'macd_signal' in ind:
            if ind['macd'] > ind['macd_signal']:
                bullish += 1
            else:
                bearish += 1
        # RSI
        if 'rsi' in ind:
            if ind['rsi'] < 30:
                bullish += 1
            elif ind['rsi'] > 70:
                bearish += 1
        # Bollinger
        if 'bb_upper' in ind and 'bb_lower' in ind and 'latest' in ind:
            if ind['latest'] < ind['bb_upper'] and ind['latest'] > ind['bb_lower']:
                bullish += 0.5
        # Determine color
        color = "yellow"
        if bullish - bearish >= 2:
            color = "#d4f4dd"  # greenish
        elif bearish - bullish >= 2:
            color = "#fde6e6"  # reddish

        with st.container():
            st.markdown(f"<div style='background:{color}; padding:10px; border-radius:6px'>", unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"{t} — {signal.get('action')}")
                st.write(f"Price: {ind.get('latest', latest['close']):.2f}")
                st.write(f"Reason summary: {signal.get('reason')}")
            with col2:
                conf = signal.get("confidence", 0)
                st.metric("Confidence", f"{conf*100:.1f}%")
                st.progress(min(max(conf, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

        # Indicators table
        if ind:
            df_ind = pd.DataFrame([{
                'SMA20': ind.get('sma20'),
                'SMA50': ind.get('sma50'),
                'MACD': ind.get('macd'),
                'MACD_signal': ind.get('macd_signal'),
                'RSI': ind.get('rsi'),
                'BB_upper': ind.get('bb_upper'),
                'BB_lower': ind.get('bb_lower'),
            }])
            st.table(df_ind)

        # Expandable why + provenance
        with st.expander("Why? (show reasoning & provenance)"):
            st.write("Full reason:", signal.get('reason'))
            gr = signal.get('guardrail', [])
            if gr:
                st.warning("Guardrail notes:")
                for g in gr:
                    st.write(f"- {g}")
            st.write("Sources:")
            for s in signal.get('sources', []):
                st.markdown(f"- {s}")


def lstm_lab_view(selected):
    st.header("LSTM Lab — Backtesting")
    tick = st.multiselect("Choose tickers to backtest", selected, default=selected[:2])
    start = st.date_input("Start date", datetime.today() - timedelta(days=365))
    end = st.date_input("End date", datetime.today())
    run = st.button("Run backtest")
    if run and tick:
        results = []
        for t in tick:
            df = fetch_hist_data(t, period="2y", interval="1d")
            if df is None or df.empty:
                st.warning(f"No historical data for {t}")
                continue
            res = backtest_strategy(t, df[(df.index.date >= start) & (df.index.date <= end)], capital=1000)
            results.append((t, res))
            st.subheader(t)
            st.line_chart(res["equity_curve"]["equity"]) 
            st.write("Metrics:")
            st.table(pd.DataFrame([res["metrics"]]))
            csv = res["equity_curve"].to_csv().encode()
            st.download_button("Download equity CSV", csv, file_name=f"{t}_equity.csv")


def agent_insights_view(selected):
    st.header("Agent Insights")
    insights = get_agent_insights(selected)
    # table summary
    summary_rows = []
    for item in insights:
        summary_rows.append({
            'ticker': item.get('ticker'),
            'action': item.get('action'),
            'confidence': float(item.get('confidence',0)),
            'rejected': bool(item.get('rejected', False)),
        })
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary_display = df_summary.copy()
        df_summary_display['confidence'] = (df_summary_display['confidence']*100).map(lambda x: f"{x:.1f}%")
        st.dataframe(df_summary_display)
        csv = df_summary.to_csv(index=False).encode()
        st.download_button('Download signals CSV', csv, file_name='signals.csv')

    for item in insights:
        badge = ''
        if item.get('rejected'):
            badge = ' — Rejected (low confidence)'
        with st.expander(f"{item['ticker']} — {item['action']} ({item['confidence']*100:.1f}%){badge}"):
            st.write("Timestamp:", item.get("timestamp"))
            st.write("Confidence:", f"{item.get('confidence',0):.2f}")
            st.write("Reason:", item.get("reason", ""))
            st.write("Guardrail actions:")
            for g in item.get('guardrail', []):
                st.warning(g)
            st.write("Sources:")
            for s in item.get("sources", []):
                st.markdown(f"- {s}")


def main():
    st.title("FIN Investment Pathway — Dashboard")
    selected, capital, start_time, end_time = sidebar_inputs()
    if not selected:
        st.info("Please select at least one stock from the sidebar to begin.")
        return

    tabs = st.tabs(["Live Market", "LSTM Lab", "Agent Insights"])
    with tabs[0]:
        live_market_view(selected)
    with tabs[1]:
        lstm_lab_view(selected)
    with tabs[2]:
        agent_insights_view(selected)


if __name__ == "__main__":
    main()
