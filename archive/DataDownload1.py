import yfinance as yf
import ta
import pandas as pd
import os

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Set the ticker symbol and date range
symbol = 'AAPL'
start_date = '2021-06-01'
end_date = '2023-03-01'

# Get daily historical data for the ticker symbol for the maximum date range then append 60 minute intervals into the data
df = yf.download(symbol, start='2016-01-01', interval='1d', progress=True, auto_adjust=True, rounding=True)
# df = df.append(yf.download(symbol, start=start_date, end=end_date, interval='60m', progress=True, auto_adjust=True, rounding=True))

# Sort by date and time and drop duplicates
df.sort_index(inplace=True)
df.drop_duplicates(inplace=True)

# Add simple and exponential moving averages to the data
# df['sma50'] = ta.trend.sma_indicator(df['Close'], 50)
# df['sma200'] = ta.trend.sma_indicator(df['Close'], 200)
# df['sma300'] = ta.trend.sma_indicator(df['Close'], 300)
# df['ema8'] = ta.trend.ema_indicator(df['Close'], 8)
# df['ema20'] = ta.trend.ema_indicator(df['Close'], 20)
# df['ema50'] = ta.trend.ema_indicator(df['Close'], 50)

# Add StochOSC, RSI, StochRSI, and MACD to the data
df['rsi'] = ta.momentum.rsi(df['Close'], 14)
df['macd'] = ta.trend.macd(df['Close'], 12, 26, 9)
df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], 14, 3, 3)
# df['stochrsi'] = ta.momentum.stochrsi(df['Close'], 14, 3, 3)

# Add Volume Weighted Average Price (VWAP) to the data
# df['vwap'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])

# # Add all Bollinger Bands to the data
# df['bb_hband'] = ta.volatility.bollinger_hband(df['Close'], 20, 2)
# df['bb_mavg'] = ta.volatility.bollinger_mavg(df['Close'], 20)
# df['bb_lband'] = ta.volatility.bollinger_lband(df['Close'], 20, 2)

# Add Aroon Oscillator to the data
df['aroon_up'] = ta.trend.aroon_up(df['Close'], 25)
df['aroon_down'] = ta.trend.aroon_down(df['Close'], 25)

# # Add Average True Range (ATR) to the data
# df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)

# # Add Chaikin Money Flow (CMF) to the data
# df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], 20)

# Add rate of change (ROC) to the data
df['roc'] = ta.momentum.roc(df['Close'], 12)

# # Add Ultimate Oscillator (UO) to the data
# df['uo'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'], 7, 14, 28)

# # Add True Strength Index (TSI) to the data
# df['tsi'] = ta.momentum.tsi(df['Close'], 25, 13)

# Add Average Directional Movement Index (ADX) to the data
# df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], 14)

# Add On Balance Volume (OBV) to the data
df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

# Add Accumulation/Distribution Index (ADI) to the data
df['adi'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

# # Add all Donchian Channel bands to the data
# df['dcu'] = ta.volatility.donchian_channel_hband(df['High'], df['Low'], df['Close'], 20)
# df['dcm'] = ta.volatility.donchian_channel_mband(df['High'], df['Low'], df['Close'], 20)
# df['dcl'] = ta.volatility.donchian_channel_lband(df['High'], df['Low'], df['Close'], 20)

# # Add all Keltner Channel bands to the data
# df['kcu'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'], 20, 2)
# df['kcm'] = ta.volatility.keltner_channel_mband(df['High'], df['Low'], df['Close'], 20, 2)
# df['kcl'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'], 20, 2)

# Drop all rows with NaN values
df.dropna(inplace=True, axis=0, how='any')

train = df.iloc[:int(len(df) * 0.8)]
test = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
val = df.iloc[int(len(df) * 0.9):]

# Set "datetime" header to first column
df = df.reset_index()
df = df.rename(columns={'index': 'Date'})
df = df.set_index('Date')

# Save the data to a csv file
train.to_csv(get_file_path('ticker data', filename='train.csv'))
val.to_csv(get_file_path('ticker data', filename='val.csv'))
test.to_csv(get_file_path('ticker data', filename='test.csv'))

# # Save the data to a parquet file
train.to_parquet(get_file_path('ticker data', filename='train.parquet'))
val.to_parquet(get_file_path('ticker data', filename='val.parquet'))
test.to_parquet(get_file_path('ticker data', filename='test.parquet'))
