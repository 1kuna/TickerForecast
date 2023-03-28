import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf
import tensorflow_io as tfio

# # Set up TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))
# print("Got past initialization")

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set automatic Mixed Precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'TPUAlpha2'
print("Project name: ", project_name)

# Read in the Parquet file
parquet_file = 'O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED_scaled.parquet'

dataset = tfio.IODataset.from_parquet(parquet_file)

tester = pd.read_parquet(parquet_file)

# Define the target column
target_col = {'open': tf.io.FixedLenFeature([], dtype=tf.float32)}

# Define the feature dictionary
feature_dict = {
        'time': tf.io.FixedLenFeature([], dtype=tf.float32),
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

# Define feature and target datasets
features_dataset = tfio.IODataset.from_parquet(
    parquet_file,
    columns=list(feature_dict.keys())
)

targets_dataset = tfio.IODataset.from_parquet(
    parquet_file,
    columns=list(target_col.keys())
)

# Define callbacks
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50
)

checkpoint_callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir='./checkpoints',
    save_freq='epoch'
)

# Define callbacks list
callbacks = [stopping_callback, checkpoint_callback]

# Initialize the model
def run_model():
    clf = ak.TimeseriesForecaster(
        max_trials=100,
        lookback=5120,
        project_name=project_name,
        overwrite=False,
        objective='val_loss',
        directory='O:\\Git\\TickerForecast\\intraday\\TICKERS2\\models',
        metrics='mape',
        loss='mae'
    )
    return clf

clf = run_model()

# with strategy.scope():
#     clf
# print("Past strategy scope")

# Set the steps per epoch
steps_per_epoch = len(tester) / 256

# Fix unknown dataset length
def fix_dataset_length(dataset):
    dataset = dataset.map(lambda x: (x, tf.constant(1, dtype=tf.int64)))
    dataset = dataset.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1)
    return dataset

# Fix unknown dataset length
features_dataset = fix_dataset_length(features_dataset)
targets_dataset = fix_dataset_length(targets_dataset)

# Train the AutoKeras model
clf.fit(features_dataset, targets_dataset, epochs=None, shuffle=False, batch_size=256, steps_per_epoch=steps_per_epoch)
