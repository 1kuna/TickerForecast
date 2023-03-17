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
input_file = get_file_path('intraday', filename=f'TRAIN_COMBINED.csv')
output_file = get_file_path(f'intraday', filename=f'TRAIN_COMBINED.tfrecord')

# Read Parquet file into a Pandas DataFrame
df = pd.read_csv(input_file)

# Set 'time' column as index
df.set_index('time', inplace=True)

# Get 'open' column as target variable
target = df.pop('open')

# Convert Pandas DataFrame to TensorFlow Example
def df_to_example(row):
    # Convert each row to a TensorFlow FeatureDict
    features = {
        col: tf.train.Feature(float_list=tf.train.FloatList(value=[val]))
        for col, val in row.items()
    }
    # Wrap the FeatureDict in a TensorFlow Example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

# Write each row of the DataFrame as a serialized Example to the TFRecord file
with tf.io.TFRecordWriter(output_file) as writer:
    for index, row in df.iterrows():
        example = df_to_example(row)
        writer.write(example.SerializeToString())

# Write target variable to separate TFRecord file
with tf.io.TFRecordWriter(output_file.split('.')[0] + '_target.tfrecord') as writer:
    for val in target:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'target': tf.train.Feature(float_list=tf.train.FloatList(value=[val]))
            }
        ))
        writer.write(example.SerializeToString())
