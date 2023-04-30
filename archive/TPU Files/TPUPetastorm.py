import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf
import numpy as np
import petastorm

from petastorm.codecs import ScalarCodec
from petastorm.unischema import Unischema, UnischemaField
from petastorm.pytorch import DataLoader
from petastorm import make_batch_reader

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = '3D2'

# Define the target column
target_col = 'open'

# Define the schema for the dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import ScalarCodec

# Define the schema for your dataset
schema = Unischema('TickerForecast', [
    UnischemaField('time', np.float64, ()),
    UnischemaField('ticker', np.float64, ()),
    UnischemaField('open', np.float64, ()),
    UnischemaField('high', np.float64, ()),
    UnischemaField('low', np.float64, ()),
    UnischemaField('close', np.float64, ()),
    UnischemaField('volume', np.float64, ()),
    UnischemaField('sma50', np.float64, ()),
    UnischemaField('sma200', np.float64, ()),
    UnischemaField('ema8', np.float64, ()),
    UnischemaField('ema20', np.float64, ()),
    UnischemaField('rsi', np.float64, ()),
    UnischemaField('macd', np.float64, ()),
    UnischemaField('stoch', np.float64, ()),
    UnischemaField('vwap', np.float64, ()),
    UnischemaField('aroon_up', np.float64, ()),
    UnischemaField('aroon_down', np.float64, ()),
    UnischemaField('roc', np.float64, ()),
    UnischemaField('obv', np.float64, ()),
    UnischemaField('adi', np.float64, ())
])


# Define the PyTorch dataloader
def create_dataloader(data_paths, batch_size, shuffle=True):
    with make_batch_reader(data_paths, schema, reader_pool_type='process', workers_count=8) as reader:
        dataloader = DataLoader(reader, batch_size=batch_size, shuffling_queue_capacity=4096, shuffling_queue_capacity_per_producer=1024, shuffling_multiproducer_mode='interleaved' if shuffle else None)
    return dataloader

# Read in the parquet file
train_paths = ['./TRAIN_COMBINED.parquet']
val_paths = ['./VAL_COMBINED.parquet']

train_dataloader = create_dataloader(train_paths, batch_size=1024)
val_dataloader = create_dataloader(val_paths, batch_size=1024)

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
        loss='huber_loss'
    )
    return clf

clf = run_model()

# Train the AutoKeras model
clf.fit(train_dataloader, validation_data=val_dataloader, epochs=None, shuffle=False, callbacks=callbacks)
