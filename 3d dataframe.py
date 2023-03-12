import os
import pandas as pd

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Define the directory where your CSV files are stored
dir_path = get_file_path('intraday/combine')

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

# Set the time column as datetime and the rest as float64
matrix['time'] = pd.to_datetime(matrix['time'])
matrix['time'] = matrix['time'].apply(lambda x: x.timestamp())
matrix['time'] = matrix['time'].astype('float64')

# Set all columns besides time and ticker as float64
matrix[matrix.columns.difference(['time'])] = matrix[matrix.columns.difference(['time'])].astype('float64')

# Print the resulting matrix
matrix.to_csv(get_file_path('intraday', filename=f'VAL_COMBINED.csv'), index=False)
matrix.to_parquet(get_file_path('intraday', filename=f'VAL_COMBINED.parquet'), index=False)