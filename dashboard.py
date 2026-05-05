import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from var_calculator import load_data, compute_var_mc, run_historical_analysis, COMMODITY_MAPPING

# --- PAGE CONFIG ---
st.set_page_config(page_title="Commodity Risk Dashboard", layout="wide", page_icon="📉")

# Let Streamlit handle native dark/light mode instead of forcing background colors
st.markdown("""
    <style>
    /* Add any transparent or adaptive styling here if needed in the future */
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Commodity Risk & Portfolio VaR")
st.markdown("### Interactive Monte Carlo Risk Analytics")

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def get_dashboard_data():
    prices_df, volumes_df, avg_cost = load_data()
    history_df = run_historical_analysis(prices_df, volumes_df)
    
    # Calculate MTM evolution over time
    trades_df = pd.read_csv('tradeprices.csv')
    t_date_col = 'Date' if 'Date' in trades_df.columns else trades_df.columns[0]
    trades_df[t_date_col] = pd.to_datetime(trades_df[t_date_col], dayfirst=False)
    
    qty = trades_df.pivot_table(index=t_date_col, columns="Ticker", values="Quantity", aggfunc="sum").fillna(0)
    cost_df = trades_df.copy()
    cost_df["Cost"] = cost_df["Quantity"] * cost_df["Trade_Price"]
    cost = cost_df.pivot_table(index=t_date_col, columns="Ticker", values="Cost", aggfunc="sum").fillna(0)
    
    qty = qty.cumsum().reindex(prices_df.index).ffill().fillna(0)
    cost = cost.cumsum().reindex(prices_df.index).ffill().fillna(0)
    
    ticker_mtm = (prices_df * qty) - cost
    mtm_history = pd.DataFrame(index=ticker_mtm.index)
    for ticker in prices_df.columns:
        if ticker in ticker_mtm.columns:
            group = COMMODITY_MAPPING.get(ticker, 'Other')
            if group not in mtm_history.columns:
                mtm_history[group] = 0
            mtm_history[group] += ticker_mtm[ticker].fillna(0)
    mtm_history['Total MTM'] = mtm_history.sum(axis=1)
    mtm_history.index.name = 'Date'
    
    return prices_df, volumes_df, avg_cost, history_df, mtm_history

try:
    prices_df, volumes_df, avg_cost, history_df, mtm_history = get_dashboard_data()
    latest_date = history_df.index[-1]
    
    # --- METRICS BAR ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate current Var and Total Value
    latest_pos = volumes_df.iloc[-1]
    var_final, asset_vars, _ = compute_var_mc(prices_df.loc[:latest_date], latest_pos, n_sims=50000)
    current_prices = prices_df.iloc[-1]
    aligned_positions = latest_pos.reindex(prices_df.columns).fillna(0)
    aligned_avg_cost = avg_cost.reindex(prices_df.columns).fillna(0)
    
    total_value = (current_prices * aligned_positions).sum()
    mtm_series = (current_prices - aligned_avg_cost) * aligned_positions
    total_mtm = mtm_series.sum()
    
    with col1:
        st.metric("Net MTM PnL", f"${total_mtm:,.2f}", delta=f"{'Profit' if total_mtm >= 0 else 'Loss'}")
    with col2:
        st.metric("Total Portfolio VaR (95%)", f"${var_final:,.2f}", delta_color="inverse")
    with col3:
        var_pct = (var_final / total_value * 100) if total_value != 0 else 0
        st.metric("VaR % of Portfolio", f"{var_pct:.2f}%")
    with col4:
        st.metric("Analysis Date", str(latest_date.date()))

    st.divider()

    # --- HISTORICAL CHART ---
    st.subheader("📈 MTM Evolution over Time")
    chart_cols = st.columns([2, 1])
    with chart_cols[0]:
        options = list(mtm_history.columns)
        selected_assets = st.multiselect("Select Asset(s) to Display", options, default=options)
        
        if not selected_assets:
            st.warning("Please select at least one asset to display.")
        else:
            area_assets = [a for a in selected_assets if a != 'Total MTM']
            fig = go.Figure()
            
            if area_assets:
                plot_df = mtm_history.reset_index().melt(id_vars='Date', value_vars=area_assets, var_name='Commodity', value_name='MTM PnL')
                area_fig = px.area(plot_df, x='Date', y='MTM PnL', color='Commodity', color_discrete_sequence=px.colors.qualitative.Antique)
                for trace in area_fig.data:
                    fig.add_trace(trace)
                    
            if 'Total MTM' in selected_assets:
                fig.add_trace(go.Scatter(x=mtm_history.index, y=mtm_history['Total MTM'], name='Total MTM PnL', line=dict(color='white', width=4)))
                
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        st.subheader("📁 Risk Allocation")
        var_comm_cols = [c for c in history_df.columns if c != 'Portfolio VaR']
        latest_grouped = history_df.iloc[-1][var_comm_cols]
        fig_pie = px.pie(values=latest_grouped.values, names=latest_grouped.index, 
                         template='plotly_dark', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Antique)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- ASSET TABLE ---
    st.subheader("📄 Detailed Asset Attribution")
    
    results_df = pd.DataFrame({
        'Ticker': prices_df.columns,
        'Group': [COMMODITY_MAPPING.get(t, 'Other') for t in prices_df.columns],
        'Market Price': current_prices.values,
        'Avg Trade Price': aligned_avg_cost.values,
        'Volume': aligned_positions.values,
        'MTM PnL': mtm_series.values,
        'Position Value': (current_prices * aligned_positions).values,
        'Individual VaR': asset_vars.values
    })
    
    # Format the dataframe
    st.dataframe(results_df.style.format({
        'Market Price': '${:,.2f}',
        'Avg Trade Price': '${:,.2f}',
        'MTM PnL': '${:,.2f}',
        'Position Value': '${:,.2f}',
        'Individual VaR': '${:,.2f}'
    }).applymap(lambda x: 'color: green' if x > 0 else ('color: red' if x < 0 else ''), subset=['MTM PnL']), use_container_width=True)

    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/fine-print.png")
        st.header("Simulation Settings")
        confidence = st.slider("Confidence Level (%)", 90, 99, 95)
        n_sims = st.number_input("Monte Carlo Simulations", 1000, 100000, 50000, step=1000)
        
        st.divider()
        st.info("The dashboard uses an EWMA volatility model with a lambda of 0.94, simulating correlated price shocks via Cholesky decomposition.")

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Check that prices.csv and tradeprices.csv are present and correctly formatted.")
