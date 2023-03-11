import os
import pandas as pd

# Define the directory where your CSV files are stored
dir_path = 'K:\\git\\TickerForecast\\intraday\\3d test'

# Create an empty list to store dataframes
dfs = []

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        # Read the CSV file and append it to the list
        filepath = os.path.join(dir_path, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)

# Combine the list of dataframes into a 3D matrix
matrix = pd.concat(dfs, keys=[i[:-4] for i in os.listdir(dir_path) if i.endswith('.csv')])

# Print the resulting matrix
matrix.to_csv('K:\\git\\TickerForecast\\intraday\\3d test\\COMBINED.csv')
