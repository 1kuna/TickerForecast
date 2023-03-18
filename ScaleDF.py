import pandas as pd
import sklearn.preprocessing as sk
import numpy as np

df = pd.read_csv('O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED.csv')

scaler = sk.RobustScaler()

# Scale everything except time column
df_scaled = scaler.fit_transform(df.drop(['time'], axis=1))
df_scaled = np.insert(df_scaled, 0, df['time'], axis=1)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled.to_csv('O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.csv', index=False)
df_scaled.to_parquet('O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.parquet')

# Read the parquet file and print the column names
df = pd.read_parquet('O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.parquet')
print(df.columns)