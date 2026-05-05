import pandas as pd
import os

filename = 'prices.csv'
print(f"Checking {filename} type...")

try:
    xl = pd.ExcelFile(filename)
    print(f"File is an EXCEL file with sheets: {xl.sheet_names}")
except Exception as e:
    print(f"File is NOT an Excel file. (Error: {e})")
    try:
        df = pd.read_csv(filename)
        print("File is a CSV file.")
        print("First few rows:")
        print(df.head())
    except Exception as e2:
        print(f"File is NOT a CSV file either. (Error: {e2})")
