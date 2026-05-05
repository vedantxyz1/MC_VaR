import traceback
try:
    from dashboard import get_dashboard_data
    prices_df, volumes_df, avg_cost, history_df, mtm_history = get_dashboard_data()
    print("Dashboard data loaded successfully!")
    print("MTM History Head:")
    print(mtm_history.head())
    print("\nMTM History Tail:")
    print(mtm_history.tail())
except Exception as e:
    print("ERROR CAUGHT:")
    traceback.print_exc()
