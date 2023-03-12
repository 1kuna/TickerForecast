import csv
import requests
import time
import os
import pandas as pd
import glob
import ta

tickers = ['SPY', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'XOM', 'QQQ']

# Function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

for ticker in tickers:
    # Create a directory for the ticker
    if not os.path.exists(get_file_path(f'intraday\\TICKERS\\{ticker}')):
        os.makedirs(get_file_path(f'intraday\\TICKERS\\{ticker}'))

    # Download the intraday data to CSV files
    for x in range(1,3):
        for i in range(1,13):
            CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=15min&slice=year{x}month{i}&adjusted=true&apikey=7V18CQ34ZDK1M3SJ'

            if i == 5 or i == 9:
                print('Sleeping for 90 seconds')
                time.sleep(90)
            elif x == 2 and i == 1:
                print('Sleeping for 90 seconds')
                time.sleep(90)

            if x == 1:
                name = i
            else:
                name = i + 12

            with requests.Session() as s:
                download = s.get(CSV_URL)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                my_list = list(cr)
                
                # Create a new CSV file and write the rows to it
                with open(get_file_path(f'intraday\\TICKERS\\{ticker}', filename=f'{name}.csv'), mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(my_list)
                    print(f'Year: {x} Month: {i} downloaded ({name}.csv)')

    df = pd.DataFrame()
    df = pd.concat([pd.read_csv(f, index_col='time') for f in glob.glob(get_file_path(f'intraday\\TICKERS\\{ticker}', filename='*.csv'))])

    # Drop all rows with NaN values and sort by date and time
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)

    df.to_csv(get_file_path(f'intraday\\TICKERS\\{ticker}', filename=f'{ticker}_COMBINED.csv'))
    df = pd.read_csv(get_file_path(f'intraday\\TICKERS\\{ticker}', filename=f'{ticker}_COMBINED.csv'))

    # Add simple and exponential moving averages to the data
    df['sma50'] = ta.trend.sma_indicator(df['close'], 50)
    df['sma200'] = ta.trend.sma_indicator(df['close'], 200)
    df['ema8'] = ta.trend.ema_indicator(df['close'], 8)
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)

    # Add StochOSC, RSI, StochRSI, and MACD to the data
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['macd'] = ta.trend.macd(df['close'], 12, 26, 9)
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3, 3)

    # Add Volume Weighted Average Price (VWAP) to the data
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

    # Add Aroon Oscillator to the data
    df['aroon_up'] = ta.trend.aroon_up(df['close'], 25)
    df['aroon_down'] = ta.trend.aroon_down(df['close'], 25)

    # Add rate of change (ROC) to the data
    df['roc'] = ta.momentum.roc(df['close'], 12)

    # Add On Balance Volume (OBV) to the data
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # Add Accumulation/Distribution Index (ADI) to the data
    df['adi'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

    # Drop all rows with NaN values
    df.dropna(inplace=True, axis=0, how='any')

    # Set "datetime" header to first column
    df = df.set_index('time')

    train = df.iloc[:int(len(df) * 0.8)]
    test = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    val = df.iloc[int(len(df) * 0.9):]

    # Create a directory for the training data
    if not os.path.exists(get_file_path(f'intraday\\TICKERS\\{ticker}\\training')):
        os.makedirs(get_file_path(f'intraday\\TICKERS\\{ticker}\\training'))

    # Save the data to a csv file
    train.to_csv(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_TRAIN.csv'))
    val.to_csv(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_VAL.csv'))
    test.to_csv(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_TEST.csv'))

    # # Save the data to a parquet file
    train.to_parquet(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_TRAIN.parquet'))
    val.to_parquet(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_VAL.parquet'))
    test.to_parquet(get_file_path(f'intraday\\TICKERS\\{ticker}\\training', filename=f'{ticker}_TEST.parquet'))

    time.sleep(75)