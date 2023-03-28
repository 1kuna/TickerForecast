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

# Define function to convert tfrecord to csv
def tfrecord_to_csv(input_file, output_file):
    # Read TFRecord file into a TensorFlow Dataset
    dataset = tf.data.TFRecordDataset(input_file)

    # Define the feature dict
    feature_dict = {
        'time': tf.io.FixedLenFeature([], dtype=tf.float32),
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

    # Define function to parse the serialized Example
    def parse_example(example):
        return tf.io.parse_single_example(example, feature_dict)

    # Parse each serialized Example in the TFRecord file
    dataset = dataset.map(parse_example)

    # Convert each row of the TensorFlow Dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Reorder columns to match original order
    column_order = list(feature_dict.keys())
    df = df[column_order]

    # Convert the Pandas DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")

tfrecord_to_csv(get_file_path('intraday/TICKERS2', filename=f'COMBINED.tfrecord'), get_file_path('intraday/TICKERS2', filename=f'RECORD.csv'))
