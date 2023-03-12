import pandas as pd
import mmh3

# Load each ticker dataset into a separate DataFrame
aapl_df = pd.read_csv('AAPL_COMBINED.csv')
spy_df = pd.read_csv('SPY_COMBINED.csv')

# Add a hash value column to each DataFrame
aapl_df['ticker_hash'] = mmh3.hash('AAPL')
spy_df['ticker_hash'] = mmh3.hash('MSFT')

# Merge all the DataFrames into a single DataFrame using the hash value column as the key
merged_df = pd.concat([aapl_df, spy_df], axis=0, ignore_index=True)

# Pivot the merged DataFrame to create a 3D matrix with time index as the first dimension, ticker index as the second dimension, and feature values as the third dimension
pivoted_df = merged_df.pivot(index='time', columns='ticker_hash', values=['open', 'high', 'low', 'close', 'volume'])
pivoted_df.columns = [f'{ticker}_{col}' for col, ticker in pivoted_df.columns]

# Export the pivoted DataFrame to a CSV file
pivoted_df.to_csv('3d_matrix.csv')
