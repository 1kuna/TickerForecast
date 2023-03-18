import tensorflow as tf
import autokeras as ak
import os

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

project_name = 'TPUAlpha2'
print("Project name: ", project_name)

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

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_dict)
    features = {k: tf.cast(v, dtype) for k, (v, dtype) in parsed_example.items()}
    label = features.pop('open')  # Remove the 'open' column from the features and use it as the label
    return features, label

file_path = "O:\\Git\\TickerForecast\\intraday\\TICKERS2\\COMBINED.tfrecord"
dataset = tf.data.TFRecordDataset(file_path)
dataset = dataset.map(parse_function)

# Batch and prefetch the datasets
batch_size = 1024
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
        directory='./models',
        metrics='mape',
        loss='mae'
    )
    return clf

clf = run_model()

# with strategy.scope():
#     clf
# print("Past strategy scope")

# Train the AutoKeras model
clf.fit(dataset, epochs=None, shuffle=False, callbacks=callbacks)
