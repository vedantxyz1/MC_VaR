import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION & MAPPING ---
# Define how tickers group into parent commodities
COMMODITY_MAPPING = {
    'SIV6 Comdty': 'Silver',
    'SIZ6 Comdty': 'Silver',
    'GCZ6 Comdty': 'Gold',
    'PLF7 Comdty': 'Platinum',
    'PAZ6 Comdty': 'Palladium'
}

def compute_ewma_vol(log_returns, lam=0.94):
    """
    Compute EWMA volatility for each column (asset).
    """
    n = len(log_returns)
    weights = np.array([(1 - lam) * (lam ** i) for i in range(n)][::-1])
    weights /= weights.sum()

    var = (log_returns ** 2).multiply(weights, axis=0).sum()
    vol = np.sqrt(var)
    return vol

def compute_var_mc(prices_df, positions_series, n_sims=10000, lam=0.94, confidence=0.95):
    """
    Monte Carlo VaR with EWMA vol + correlation.
    """
    # Align positions
    positions = positions_series.reindex(prices_df.columns).fillna(0).values
    
    # Step 1: log returns
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    if log_returns.empty:
        return 0, pd.Series(0, index=prices_df.columns), None

    # Step 2: EWMA volatility
    vol = compute_ewma_vol(log_returns, lam=lam)

    # Step 3: correlation matrix
    corr = log_returns.corr()
    cov = np.outer(vol, vol) * corr.values

    # Step 4: Cholesky
    try:
        chol = np.linalg.cholesky(cov)
    except:
        cov += np.eye(len(cov)) * 1e-12
        chol = np.linalg.cholesky(cov)

    # Step 5: Simulation
    n_assets = prices_df.shape[1]
    Z = np.random.normal(size=(n_sims, n_assets))
    correlated_shocks = Z @ chol.T
    
    # Step 6: Next-day prices & PnL
    current_prices = prices_df.iloc[-1].values
    simulated_prices = current_prices * np.exp(correlated_shocks)
    asset_pnls = (simulated_prices - current_prices) * positions
    portfolio_pnl = asset_pnls.sum(axis=1)

    # Step 7: VaR results
    alpha = (1 - confidence) * 100
    portfolio_var = -np.percentile(portfolio_pnl, alpha)
    asset_vars = pd.Series(
        [-np.percentile(asset_pnls[:, i], alpha) for i in range(n_assets)],
        index=prices_df.columns
    )

    return portfolio_var, asset_vars, portfolio_pnl

def load_data(prices_file='prices.csv', trades_file='tradeprices.csv'):
    """
    Loads prices and calculates volume history from trade file.
    """
    if not os.path.exists(prices_file) or not os.path.exists(trades_file):
        raise FileNotFoundError("Missing prices.csv or tradeprices.csv")
        
    # Read prices
    prices_df = pd.read_csv(prices_file)
    date_col = 'Dates' if 'Dates' in prices_df.columns else prices_df.columns[0]
    prices_df[date_col] = pd.to_datetime(prices_df[date_col], dayfirst=False)
    prices_df = prices_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    prices_df = prices_df.apply(pd.to_numeric, errors='coerce').ffill()
    
    # Read tradeprices
    trades_df = pd.read_csv(trades_file)
    t_date_col = 'Date' if 'Date' in trades_df.columns else trades_df.columns[0]
    trades_df[t_date_col] = pd.to_datetime(trades_df[t_date_col], dayfirst=False)
    trades_df = trades_df.sort_values(t_date_col)
    
    # Pivot to get daily sums, then cumsum and reindex to price dates
    qty = trades_df.pivot_table(index=t_date_col, columns="Ticker", values="Quantity", aggfunc="sum").fillna(0)
    volumes_df = qty.cumsum().reindex(prices_df.index).ffill().fillna(0)
    
    # Calculate average trade price per ticker
    avg_cost = trades_df.groupby("Ticker").apply(
        lambda x: np.average(x["Trade_Price"], weights=np.abs(x["Quantity"]))
        if (not x.empty and x["Quantity"].sum() != 0) else 0
    )
    
    return prices_df, volumes_df, avg_cost

def run_historical_analysis(prices_df, volumes_df):
    """
    Calculates VaR for every date where we have volume data.
    """
    history = []
    
    print("Calculating historical VaR metrics...")
    for date in volumes_df.index:
        # Get prices up to this date
        current_prices_history = prices_df.loc[:date]
        if len(current_prices_history) < 2:
            continue
            
        current_positions = volumes_df.loc[date]
        
        # Calculate MC VaR
        p_var, a_vars, _ = compute_var_mc(current_prices_history, current_positions, n_sims=1000)
        
        # Group asset VaRs by commodity
        # For simplicity in historical view, we'll sum individual VaRs by group 
        # (This is conservative, but common for stacked charts)
        grouped_vars = {}
        for ticker, v in a_vars.items():
            group = COMMODITY_MAPPING.get(ticker, ticker)
            grouped_vars[group] = grouped_vars.get(group, 0) + v
            
        record = {
            'Date': date,
            'Portfolio VaR': p_var,
            **grouped_vars
        }
        history.append(record)
        
    return pd.DataFrame(history).set_index('Date')

def plot_var_history(history_df):
    """
    Generates the trend chart.
    """
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    # Identify commodity columns (those that aren't 'Portfolio VaR')
    comm_cols = [c for c in history_df.columns if c != 'Portfolio VaR']
    
    # Stacked Area for Commodity Groups
    plt.stackplot(history_df.index, [history_df[c] for c in comm_cols], 
                  labels=comm_cols, alpha=0.6)
    
    # Line for Total Portfolio VaR
    plt.plot(history_df.index, history_df['Portfolio VaR'], 
             color='white', linewidth=3, label='Total Portfolio VaR', marker='o')
    
    plt.title('VaR Evolution by Commodity Group', fontsize=16, pad=20)
    plt.ylabel('Value at Risk ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    chart_path = 'var_trend.png'
    plt.savefig(chart_path)
    print(f"\nSuccess! Chart saved to: {chart_path}")
    return chart_path

if __name__ == "__main__":
    try:
        # 1. Load data
        prices_df, volumes_df, avg_cost = load_data()
        
        # 2. Run Historical Simulation
        history_df = run_historical_analysis(prices_df, volumes_df)
        
        # 3. Print Latest Metrics (same as before)
        latest_date = history_df.index[-1]
        latest_pos = volumes_df.iloc[-1]
        
        # Re-run full simulation for latest date with more sims for accuracy
        var_final, asset_vars, _ = compute_var_mc(prices_df.loc[:latest_date], latest_pos, n_sims=100000)
        
        current_prices = prices_df.iloc[-1]
        aligned_positions = latest_pos.reindex(prices_df.columns).fillna(0)
        pos_values = current_prices * aligned_positions
        aligned_avg_cost = avg_cost.reindex(prices_df.columns).fillna(0)
        mtm_series = (current_prices - aligned_avg_cost) * aligned_positions
        
        print("\n" + "="*95)
        print(f"{'LATEST RISK REPORT (' + str(latest_date.date()) + ')':^95}")
        print("="*95)
        
        # Table of results
        results_df = pd.DataFrame({
            'Ticker': prices_df.columns,
            'Group': [COMMODITY_MAPPING.get(t, 'Other') for t in prices_df.columns],
            'Market Price': current_prices.values,
            'Trade Price': aligned_avg_cost.values,
            'Volume': aligned_positions.values,
            'MTM PnL': mtm_series.values,
            'Position Value': pos_values.values,
            'Individual VaR': asset_vars.values
        })
        pd.options.display.float_format = '{:,.2f}'.format
        print(results_df.to_string(index=False))
        
        print("-" * 95)
        print(f"Total Portfolio VaR: {var_final:,.2f}")
        print("="*95)
        
        # 4. Generate Visual Chart
        plot_var_history(history_df)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
