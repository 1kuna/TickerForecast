import csv
import requests
import time
import os
import pandas as pd
import glob
import ta

tickers = ['SPY', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 
           'XOM', 'QQQ', 'AAPL', 'AMD', 'BAC', 'DIS', 'BRK-B', 
           'GOOG', 'IBM', 'JPM', 'JNJ', 'META', 'NFLX', 'PG', 
           'SBUX', 'V', 'TSM', 'WMT']

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
    if not os.path.exists(get_file_path(f'intraday\\TICKERS2\\{ticker}')):
        os.makedirs(get_file_path(f'intraday\\TICKERS2\\{ticker}'))

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
                with open(get_file_path(f'intraday\\TICKERS2\\{ticker}', filename=f'{name}.csv'), mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(my_list)
                    print(f'Year: {x} Month: {i} downloaded ({name}.csv)')

    df = pd.DataFrame()
    df = pd.concat([pd.read_csv(f, index_col='time') for f in glob.glob(get_file_path(f'intraday\\TICKERS2\\{ticker}', filename='*.csv'))])

    # Drop all rows with NaN values and sort by date and time
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)

    df.to_csv(get_file_path(f'intraday\\TICKERS2\\{ticker}', filename=f'{ticker}_COMBINED.csv'))
    df = pd.read_csv(get_file_path(f'intraday\\TICKERS2\\{ticker}', filename=f'{ticker}_COMBINED.csv'))

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

    # Create a directory for the training data
    if not os.path.exists(get_file_path(f'intraday\\TICKERS2\\{ticker}\\training')):
        os.makedirs(get_file_path(f'intraday\\TICKERS2\\{ticker}\\training'))

    # Save the data to a csv file
    df.to_csv(get_file_path(f'intraday\\TICKERS2\\{ticker}\\training', filename=f'{ticker}_TRAIN.csv'))
    
    # # Save the data to a parquet file
    df.to_parquet(get_file_path(f'intraday\\TICKERS2\\{ticker}\\training', filename=f'{ticker}_TRAIN.parquet'), index=True)

    time.sleep(75)

dfs = []

for ticker in tickers:
        # Define the directory where your CSV files are stored
        dir_path = get_file_path(f'intraday\\TICKERS2\\{ticker}\\training')
        # Loop through each file in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith('.csv'):
                # Read the CSV file and append it to the list
                filepath = os.path.join(dir_path, filename)
                suffix = filename.split("_")[1].split(".")[0]
                if suffix == set:
                    ticker = filename.split("_")[0]
                    mx = pd.read_csv(filepath)
                    print(f'Ticker: {ticker}')
                    print(f'File: {filename}')
                    mx['ticker'] = hash(ticker)
                    dfs.append(mx)

        # Combine the list of dataframes into a 3D matrix
        matrix = pd.concat(dfs, axis=0, ignore_index=True)

        # Reorder the columns and sort by time
        matrix = matrix[['time', 'ticker'] + [c for c in matrix.columns if c not in ['time', 'ticker']]]
        matrix.sort_values(by=['time'], inplace=True)

        # Set the time column as datetime
        matrix['time'] = pd.to_datetime(matrix['time'])
        matrix['time'] = matrix['time'].apply(lambda x: x.timestamp())
        matrix['time'] = matrix['time'].astype('float64')

        # Set all columns besides time as float64
        matrix[matrix.columns.difference(['time'])] = matrix[matrix.columns.difference(['time'])].astype('float64')

        # Drop all rows with NaN values
        matrix = matrix.dropna(axis=0, how='any', inplace=True)

        # Print the resulting matrix
        matrix.to_csv(get_file_path(f'intraday\\TICKERS2', filename='COMBINED.csv'), index=False)
        matrix.to_parquet(get_file_path(f'intraday\\TICKERS2', filename='COMBINED.parquet'), index=True)
dfs = []