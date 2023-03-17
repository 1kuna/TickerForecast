import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf

# Read in the CSV file
train = pd.read_csv('K:\\Git\\TickerForecast\\intraday\\TRAIN_COMBINED.csv')
val = pd.read_csv('K:\\Git\\TickerForecast\\intraday\\VAL_COMBINED.csv')

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = train.drop([target_col], axis=1).values.astype('float32')
y_train = train[target_col].values.astype('float32')

x_val = val.drop([target_col], axis=1).values.astype('float32')
y_val = val[target_col].values.astype('float32')

# Convert data to TFRecords format
def serialize_example(feature_dict, label):
    feature = {str(k): tf.train.Feature(float_list=tf.train.FloatList(value=v)) for k, v in feature_dict.items()}
    feature['target'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(file_path, features, labels):
    with tf.io.TFRecordWriter(file_path) as writer:
        for x, y in zip(features, labels):
            example = serialize_example(x, y)
            writer.write(example)

# Write the TFRecords files
write_tfrecord('train.tfrecord', x_train, y_train)
write_tfrecord('val.tfrecord', x_val, y_val)
