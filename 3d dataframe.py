import os
import pandas as pd

# Define the directory where your CSV files are stored
dir_path = 'K:\\git\\TickerForecast\\intraday\\combine'

# Create an empty list to store dataframes
dfs = []

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        # Read the CSV file and append it to the list
        filepath = os.path.join(dir_path, filename)
        mx = pd.read_csv(filepath)
        ticker_name = filename.split("_")[0]
        print(f'Ticker: {ticker_name}')
        mx['ticker'] = hash(ticker_name)
        dfs.append(mx)

# Combine the list of dataframes into a 3D matrix and sort by date
matrix = pd.concat(dfs).sort_values('time')

# Reorder the columns
matrix = matrix[['time', 'ticker'] + [c for c in matrix.columns if c not in ['time', 'ticker']]]

# Print the resulting matrix
matrix.to_csv('K:\\git\\TickerForecast\\intraday\\TEST_COMBINED.csv', index=False)
matrix.to_parquet('K:\\git\\TickerForecast\\intraday\\TEST_COMBINED.parquet', index=False)
