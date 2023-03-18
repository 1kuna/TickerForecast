import pandas as pd
import tensorflow as tf
import os

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Define the feature dictionary
feature_dict = {
    'time': tf.io.FixedLenFeature([], dtype=tf.float32),
    'ticker': tf.io.FixedLenFeature([], dtype=tf.float32),
    'open': tf.io.FixedLenFeature([], dtype=tf.float32),
    'high': tf.io.FixedLenFeature([], dtype=tf.float32),
    'low': tf.io.FixedLenFeature([], dtype=tf.float32),
    'close': tf.io.FixedLenFeature([], dtype=tf.float32),
    'volume': tf.io.FixedLenFeature([], dtype=tf.float32),
    'sma50': tf.io.FixedLenFeature([], dtype=tf.float32),
    'sma200': tf.io.FixedLenFeature([], dtype=tf.float32),
    'ema8': tf.io.FixedLenFeature([], dtype=tf.float32),
    'ema20': tf.io.FixedLenFeature([], dtype=tf.float32),
    'rsi': tf.io.FixedLenFeature([], dtype=tf.float32),
    'macd': tf.io.FixedLenFeature([], dtype=tf.float32),
    'stoch': tf.io.FixedLenFeature([], dtype=tf.float32),
    'vwap': tf.io.FixedLenFeature([], dtype=tf.float32),
    'aroon_up': tf.io.FixedLenFeature([], dtype=tf.float32),
    'aroon_down': tf.io.FixedLenFeature([], dtype=tf.float32),
    'roc': tf.io.FixedLenFeature([], dtype=tf.float32),
    'obv': tf.io.FixedLenFeature([], dtype=tf.float32),
    'adi': tf.io.FixedLenFeature([], dtype=tf.float32)
}

tfrecord_file = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled.tfrecord')
output_csv_file = get_file_path('intraday/TICKERS2', filename='VERIFIEDTF.csv')

def parse_example(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_dict)
    parsed_data = {}
    for k, v in parsed_example.items():
        dtype = feature_dict[k].dtype
        parsed_data[k] = tf.cast(v, dtype)
    return parsed_data

# Read the TFRecord file
dataset = tf.data.TFRecordDataset(tfrecord_file)
parsed_dataset = dataset.map(parse_example)

# Convert the dataset to a DataFrame
rows = [row for row in parsed_dataset.as_numpy_iterator()]
df = pd.DataFrame(rows)

# Save the DataFrame as a CSV file
df.to_csv(output_csv_file, index=False)
