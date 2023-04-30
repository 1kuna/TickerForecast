import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf
from google.cloud import storage

# Set up TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
print("Got past initialization")

# Create a GCS client
client = storage.Client()
bucket_name = "eu-forecast-bucket"
bucket = client.get_bucket(bucket_name)

# Download the parquet file from GCS
parquet_blob = bucket.blob("COMBINED_scaled.parquet")
parquet_blob.download_to_filename("./COMBINED_scaled.parquet")

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set automatic Mixed Precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'TPUAlpha1'
print("Project name: ", project_name)

# Read in the parquet file
train = pd.read_parquet('./COMBINED_scaled.parquet')

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = train.drop([target_col], axis=1).values
y_train = train[target_col].values

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

with strategy.scope():
    clf
print("Past strategy scope")

# Train the AutoKeras model
clf.fit(x_train, y_train, epochs=None, shuffle=False, batch_size=1024, callbacks=callbacks)