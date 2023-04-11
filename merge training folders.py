import os
import pandas as pd

main_folder = 'K:\\Git\\TickerForecast\\intraday\\TICKERS2'
output_file = os.path.join(main_folder, 'FINALtest.csv')

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()
i = 1

# Iterate through the subfolders
for ticker in os.listdir(main_folder):
    ticker_folder = os.path.join(main_folder, ticker)
    if os.path.isdir(ticker_folder):
        training_folder = os.path.join(ticker_folder, 'training')
        csv_file = os.path.join(training_folder, f'{ticker}_TRAIN.csv')
        
        # Check if the CSV file exists
        if os.path.isfile(csv_file):
            # Load the CSV file into a DataFrame
            ticker_data = pd.read_csv(csv_file)
            
            # Add the ticker name as a new column
            ticker_data.insert(loc=1, column='ticker', value=i)
            
            # Append the data to the merged_data DataFrame
            merged_data = merged_data.append(ticker_data, ignore_index=True)

            i += 1

# Sort the merged data by the 'time' column
merged_data.sort_values(by='time', inplace=True)

# Save the merged data to the 'FINAL.csv' file
merged_data.to_csv(output_file, index=False)
