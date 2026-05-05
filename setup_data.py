import pandas as pd
import numpy as np

# Create more realistic dummy data with some variance
np.random.seed(42)
dates = pd.date_range(start='2026-03-01', periods=60, freq='D')
tickers = ['SIV6 Comdty', 'SIZ6 Comdty', 'GCZ6 Comdty', 'PLF7 Comdty', 'PAZ6 Comdty']

# Generate random walk prices
price_data = {'Date': dates}
for t in tickers:
    # Start price around the user's values
    start_price = 85000 if 'SI' in t else (2000 if 'GC' in t else 1500)
    # Random returns ~ 1% daily vol
    returns = np.random.normal(0, 0.01, len(dates))
    prices = start_price * np.exp(np.cumsum(returns))
    price_data[t] = prices

prices_df = pd.DataFrame(price_data)

# Sample volumes
volume_data = {
    'SIV6 Comdty': [10],
    'SIZ6 Comdty': [5],
    'GCZ6 Comdty': [100],
    'PLF7 Comdty': [50],
    'PAZ6 Comdty': [20]
}
vols_df = pd.DataFrame(volume_data)

# Save to Excel
with pd.ExcelWriter('VaR_Data.xlsx') as writer:
    prices_df.to_excel(writer, sheet_name='prices', index=False)
    vols_df.to_excel(writer, sheet_name='volumes', index=False)

print("VaR_Data.xlsx has been refreshed with randomized price history!")
