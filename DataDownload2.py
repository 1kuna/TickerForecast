import yfinance as yf
import ta
import pandas as pd
import os
import glob

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Define the list of tickers to download
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'SPY',
           'QQQ', 'TSLA', 'GOOG', 'META', 'AMD']

# Loop through list of tickers and append data to a dataframe
for ticker in tickers:
    # Get daily historical data for the ticker symbol for the maximum date range then append 60 minute intervals into the data
    df1 = yf.download(ticker, start='2014-01-01', interval='1d', progress=True, auto_adjust=True, rounding=True)
    df2 = yf.download(ticker, start='2021-06-01', interval='60m', progress=True, auto_adjust=True, rounding=True)

    # Sort by date and time and drop duplicates
    df1.sort_index(inplace=True)
    df1.drop_duplicates(inplace=True)
    df2.sort_index(inplace=True)
    df2.drop_duplicates(inplace=True)

    # Add simple and exponential moving averages to df1
    df1['sma50'] = ta.trend.sma_indicator(df1['Close'], 50)
    df1['sma200'] = ta.trend.sma_indicator(df1['Close'], 200)
    df1['sma300'] = ta.trend.sma_indicator(df1['Close'], 300)
    df1['ema8'] = ta.trend.ema_indicator(df1['Close'], 8)
    df1['ema20'] = ta.trend.ema_indicator(df1['Close'], 20)
    df1['ema50'] = ta.trend.ema_indicator(df1['Close'], 50)

    # Add simple and exponential moving averages to df2
    df2['sma50'] = ta.trend.sma_indicator(df2['Close'], 50)
    df2['sma200'] = ta.trend.sma_indicator(df2['Close'], 200)
    df2['sma300'] = ta.trend.sma_indicator(df2['Close'], 300)
    df2['ema8'] = ta.trend.ema_indicator(df2['Close'], 8)
    df2['ema20'] = ta.trend.ema_indicator(df2['Close'], 20)
    df2['ema50'] = ta.trend.ema_indicator(df2['Close'], 50)   

    # Add StochOSC, RSI, StochRSI, and MACD to the data
    df1['rsi'] = ta.momentum.rsi(df1['Close'], 14)
    df1['macd'] = ta.trend.macd(df1['Close'], 12, 26, 9)
    df1['stoch'] = ta.momentum.stoch(df1['High'], df1['Low'], df1['Close'], 14, 3, 3)
    df1['stochrsi'] = ta.momentum.stochrsi(df1['Close'], 14, 3, 3)

    df2['rsi'] = ta.momentum.rsi(df2['Close'], 14)
    df2['macd'] = ta.trend.macd(df2['Close'], 12, 26, 9)
    df2['stoch'] = ta.momentum.stoch(df2['High'], df2['Low'], df2['Close'], 14, 3, 3)
    df2['stochrsi'] = ta.momentum.stochrsi(df2['Close'], 14, 3, 3)

    # Add Volume Weighted Average Price (VWAP) to the data
    df1['vwap'] = ta.volume.volume_weighted_average_price(df1['High'], df1['Low'], df1['Close'], df1['Volume'])
    df2['vwap'] = ta.volume.volume_weighted_average_price(df2['High'], df2['Low'], df2['Close'], df2['Volume'])

    # Add all Bollinger Bands to the data
    df1['bb_hband'] = ta.volatility.bollinger_hband(df1['Close'], 20, 2)
    df1['bb_mavg'] = ta.volatility.bollinger_mavg(df1['Close'], 20)
    df1['bb_lband'] = ta.volatility.bollinger_lband(df1['Close'], 20, 2)

    df2['bb_hband'] = ta.volatility.bollinger_hband(df2['Close'], 20, 2)
    df2['bb_mavg'] = ta.volatility.bollinger_mavg(df2['Close'], 20)
    df2['bb_lband'] = ta.volatility.bollinger_lband(df2['Close'], 20, 2)

    # # Add Aroon Oscillator to the data
    # df1['aroon_up'] = ta.trend.aroon_up(df1['Close'], 25)
    # df1['aroon_down'] = ta.trend.aroon_down(df1['Close'], 25)

    # df2['aroon_up'] = ta.trend.aroon_up(df2['Close'], 25)
    # df2['aroon_down'] = ta.trend.aroon_down(df2['Close'], 25)

    # Add Average True Range (ATR) to the data
    df1['atr'] = ta.volatility.average_true_range(df1['High'], df1['Low'], df1['Close'], 14)
    df2['atr'] = ta.volatility.average_true_range(df2['High'], df2['Low'], df2['Close'], 14)

    # # Add Chaikin Money Flow (CMF) to the data
    # df1['cmf'] = ta.volume.chaikin_money_flow(df1['High'], df1['Low'], df1['Close'], df1['Volume'], 20)
    # df2['cmf'] = ta.volume.chaikin_money_flow(df2['High'], df2['Low'], df2['Close'], df2['Volume'], 20)

    # Add Rate Of Change (ROC) to the data
    df1['roc'] = ta.momentum.roc(df1['Close'], 12)
    df2['roc'] = ta.momentum.roc(df2['Close'], 12)

    # # Add Ultimate Oscillator (UO) to the data
    # df1['uo'] = ta.momentum.ultimate_oscillator(df1['High'], df1['Low'], df1['Close'], 7, 14, 28)
    # df2['uo'] = ta.momentum.ultimate_oscillator(df2['High'], df2['Low'], df2['Close'], 7, 14, 28)

    # # Add True Strength Index (TSI) to the data
    # df1['tsi'] = ta.momentum.tsi(df1['Close'], 25, 13)
    # df2['tsi'] = ta.momentum.tsi(df2['Close'], 25, 13)

    # Add On Balance Volume (OBV) to the data
    df1['obv'] = ta.volume.on_balance_volume(df1['Close'], df1['Volume'])
    df2['obv'] = ta.volume.on_balance_volume(df2['Close'], df2['Volume'])

    # Add Accumulation/Distribution Index (ADI) to the data
    df1['adi'] = ta.volume.acc_dist_index(df1['High'], df1['Low'], df1['Close'], df1['Volume'])
    df2['adi'] = ta.volume.acc_dist_index(df2['High'], df2['Low'], df2['Close'], df2['Volume'])

    # # Add all Donchian Channel bands to the data
    # df1['dcu'] = ta.volatility.donchian_channel_hband(df1['High'], df1['Low'], df1['Close'], 20)
    # df1['dcm'] = ta.volatility.donchian_channel_mband(df1['High'], df1['Low'], df1['Close'], 20)
    # df1['dcl'] = ta.volatility.donchian_channel_lband(df1['High'], df1['Low'], df1['Close'], 20)

    # df2['dcu'] = ta.volatility.donchian_channel_hband(df2['High'], df2['Low'], df2['Close'], 20)
    # df2['dcm'] = ta.volatility.donchian_channel_mband(df2['High'], df2['Low'], df2['Close'], 20)
    # df2['dcl'] = ta.volatility.donchian_channel_lband(df2['High'], df2['Low'], df2['Close'], 20)

    # # Add all Keltner Channel bands to the data
    # df1['kcu'] = ta.volatility.keltner_channel_hband(df1['High'], df1['Low'], df1['Close'], 20, 2)
    # df1['kcm'] = ta.volatility.keltner_channel_mband(df1['High'], df1['Low'], df1['Close'], 20, 2)
    # df1['kcl'] = ta.volatility.keltner_channel_lband(df1['High'], df1['Low'], df1['Close'], 20, 2)

    # df2['kcu'] = ta.volatility.keltner_channel_hband(df2['High'], df2['Low'], df2['Close'], 20, 2)
    # df2['kcm'] = ta.volatility.keltner_channel_mband(df2['High'], df2['Low'], df2['Close'], 20, 2)
    # df2['kcl'] = ta.volatility.keltner_channel_lband(df2['High'], df2['Low'], df2['Close'], 20, 2)

    # Drop all rows with NaN values
    df1.dropna(inplace=True, axis=0, how='any')
    df2.dropna(inplace=True, axis=0, how='any')

    df1 = df1.add_prefix(ticker + '_')
    df2 = df2.add_prefix(ticker + '_')

    # Split the data into train and test sets; 80% train, 20% test
    train1 = df1[:int(len(df1) * 0.8)]
    train2 = df2[:int(len(df2) * 0.8)]
    test1 = df1[int(len(df1) * 0.8):]
    test2 = df2[int(len(df2) * 0.8):]

    # Append the train and test sets together
    train = train1.append(train2)
    test = test1.append(test2)

    # Sort by date and time and drop duplicates
    test.sort_index(inplace=True)
    test.drop_duplicates(inplace=True)

    # Set "datetime" header to first column
    train = train.reset_index()
    train = train.rename(columns={'index': 'datetime'})
    train = train.set_index('datetime')
    test = test.reset_index()
    test = test.rename(columns={'index': 'datetime'})
    test = test.set_index('datetime')

    # Save the data to a csv file
    train.to_csv(get_file_path('ticker data', filename=f'{ticker}_TRAIN.csv'))
    test.to_csv(get_file_path('ticker data', filename=f'{ticker}_TEST.csv'))

    # Save the data to a parquet file
    train.to_parquet(get_file_path('ticker data', filename=f'{ticker}_TRAIN.parquet'))
    test.to_parquet(get_file_path('ticker data', filename=f'{ticker}_TEST.parquet'))

# Append all the csv files into one dataframe keeping the date and time as the index
train = pd.concat([pd.read_csv(f, index_col='datetime') for f in glob.glob(get_file_path('ticker data', filename='*TRAIN.csv'))], axis=1)
test = pd.concat([pd.read_csv(f, index_col='datetime') for f in glob.glob(get_file_path('ticker data', filename='*TEST.csv'))], axis=1)

# Drop all rows with NaN values
train.dropna(inplace=True, axis=0, how='any')
test.dropna(inplace=True, axis=0, how='any')

# Save the data to a csv file
train.to_csv(get_file_path('ticker data', filename='train.csv'))
test.to_csv(get_file_path('ticker data', filename='test.csv'))

# Save the data to a parquet file
train.to_parquet(get_file_path('ticker data', filename='train.parquet'))
test.to_parquet(get_file_path('ticker data', filename='test.parquet'))