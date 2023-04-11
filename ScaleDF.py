import pandas as pd
import sklearn.preprocessing as sk
import numpy as np

df = pd.read_csv('K:\\Git\\TickerForecast\\intraday\\TICKERS2\\FINAL.csv')

scaler = sk.RobustScaler()

# Set the time column as datetime
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].apply(lambda x: x.timestamp())
df['time'] = df['time'].astype('float64')

# Set all columns besides time as float64
df[df.columns.difference(['time'])] = df[df.columns.difference(['time'])].astype('float64')

# Drop all rows with NaN values
df = df.dropna(axis=0, how='any')

# Scale everything except time column
df_scaled = scaler.fit_transform(df.drop(['time', 'ticker'], axis=1))
df_scaled = np.insert(df_scaled, 0, df['time'], axis=1)
df_scaled = np.insert(df_scaled, 1, df['ticker'], axis=1)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled.to_csv('K:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.csv', index=False)
df_scaled.to_parquet('K:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.parquet')

# Read the parquet file and print the column names
df = pd.read_parquet('K:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.parquet')
print(df.columns)