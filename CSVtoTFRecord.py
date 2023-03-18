import os
import tensorflow as tf
import pandas as pd

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Define input and output file paths
input_file = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled.csv')
output_file = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled.tfrecord')

data = pd.read_csv(input_file)

# Define the feature dictionary
feature_dict = {
    'time': tf.float32,
    'ticker': tf.float32,
    'open': tf.float32,
    'high': tf.float32,
    'low': tf.float32,
    'close': tf.float32,
    'volume': tf.float32,
    'sma50': tf.float32,
    'sma200': tf.float32,
    'ema8': tf.float32,
    'ema20': tf.float32,
    'rsi': tf.float32,
    'macd': tf.float32,
    'stoch': tf.float32,
    'vwap': tf.float32,
    'aroon_up': tf.float32,
    'aroon_down': tf.float32,
    'roc': tf.float32,
    'obv': tf.float32,
    'adi': tf.float32
}

# Function to convert a value to a tf.train.Feature
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Function to convert a row in the CSV to a tf.train.Example
def csv_row_to_example(row):
    features = {}
    for col, dtype in feature_dict.items():
        if dtype == tf.float32:
            features[col] = _float_feature(row[col])
    return tf.train.Example(features=tf.train.Features(feature=features))

# Write the TFRecord file
with tf.io.TFRecordWriter(output_file) as writer:
    for _, row in data.iterrows():
        example = csv_row_to_example(row)
        writer.write(example.SerializeToString())