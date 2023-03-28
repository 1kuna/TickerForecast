import tensorflow as tf
import numpy as np
import pandas as pd

# Define the feature dictionary
feature_columns = ['time', 'open', 'high', 'low', 'close', 'volume',
                   'sma50', 'sma200', 'ema8', 'ema20', 'rsi', 'macd',
                   'stoch', 'vwap', 'aroon_up', 'aroon_down', 'roc',
                   'obv', 'adi']

# Define the file path to write the TFRecord file
file_path = "O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED.tfrecord"

# Read in the CSV file
data = pd.read_csv("O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED.csv")

# Convert columns to float32
data = data.astype('float32')

# Define the target column
target_col = 'open'

# Get the features and target arrays
features = data.drop([target_col], axis=1)
target = data[target_col]

# Open the TFRecord file for writing
with tf.io.TFRecordWriter(file_path) as writer:
    for i in range(len(features)):
        # Extract the features and target for this row
        row_features = features.iloc[i]
        row_target = target.iloc[i]

        # Convert the features and target to numpy arrays
        features_array = np.array(row_features)
        target_array = np.array(row_target)

        # Create a feature dictionary
        feature = {}
        for col, value in zip(features.columns, features_array):
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        feature['open'] = tf.train.Feature(float_list=tf.train.FloatList(value=[target_array]))

        # Create an example protobuf message
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string and write it to the TFRecord file
        writer.write(example.SerializeToString())