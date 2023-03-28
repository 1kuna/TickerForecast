import os
import tensorflow as tf
import pandas as pd

def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

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

input_file = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled.csv')
output_file_features = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled_features.tfrecord')
output_file_targets = get_file_path('intraday/TICKERS2', filename='COMBINED_scaled_targets.tfrecord')

data = pd.read_csv(input_file)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def csv_row_to_example(row, target_col):
    features = {}
    for col, dtype in feature_dict.items():
        if col != target_col and dtype == tf.float32:
            features[col] = _float_feature(row[col])
    return tf.train.Example(features=tf.train.Features(feature=features))

def csv_row_to_target(row, target_col):
    return _float_feature(row[target_col])

target_col = 'open'

with tf.io.TFRecordWriter(output_file_features) as writer:
    for _, row in data.iterrows():
        example = csv_row_to_example(row, target_col)
        writer.write(example.SerializeToString())

with tf.io.TFRecordWriter(output_file_targets) as writer:
    for _, row in data.iterrows():
        target = csv_row_to_target(row, target_col)
        writer.write(target.SerializeToString())