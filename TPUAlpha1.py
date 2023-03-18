import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf
import sklearn.preprocessing as sk

# Set up TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
print("Got past initialization")

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set automatic Mixed Precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'TPUAlpha1'
print("Project name: ", project_name)

# Read in the parquet file
train = pd.read_parquet('./TRAIN_COMBINED.parquet')
val = pd.read_parquet('./VAL_COMBINED.parquet')

# Convert columns to float32
train = train.astype('float32')
val = val.astype('float32')

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = train.drop([target_col], axis=1).values
y_train = train[target_col].values

x_val = val.drop([target_col], axis=1).values
y_val = val[target_col].values

# Scale all data with RobustScaler except datetime columns
scaler = sk.RobustScaler()
x_train[:, 1:] = scaler.fit_transform(x_train[:, 1:])
x_val[:, 1:] = scaler.transform(x_val[:, 1:])

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

with strategy.scope():
    clf
print("Past strategy scope")

# Train the AutoKeras model
clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=None, shuffle=False, batch_size=1024, callbacks=callbacks)