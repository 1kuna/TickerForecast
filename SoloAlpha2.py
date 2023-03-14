import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as sk
import json

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = '3D2'

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Read in the parquet file
train = pd.read_parquet(get_file_path('intraday', filename=f'TRAIN_COMBINED.parquet'))
val = pd.read_parquet(get_file_path(f'intraday', filename=f'VAL_COMBINED.parquet'))
test = pd.read_parquet(get_file_path(f'intraday', filename=f'TEST_COMBINED.parquet'))

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = train.drop([target_col], axis=1).values
y_train = train[target_col].values

x_val = val.drop([target_col], axis=1).values
y_val = val[target_col].values

x_test = test.drop([target_col], axis=1).values
y_test= test[target_col].values

# Define callbacks
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=tensorboard_dir, histogram_freq=50, 
#     write_graph=True, write_images=True, update_freq='batch', 
#     write_steps_per_second=False
# )
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=30
)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=get_file_path('models\\checkpoints', filename=f'{project_name}-{currentTime}.h5'),
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    verbose=1,
    save_freq='epoch'
)

# Define callbacks list
callbacks = [stopping_callback, checkpoint_callback]

# Initialize the model
def run_model():
    clf = ak.TimeseriesForecaster(
        # max_trials=250,
        lookback=5120,
        project_name=project_name,
        overwrite=False,
        objective='val_loss',
        directory=get_file_path('models'),
        metrics='mape',
        loss='huber_loss'
    )
    return clf

# Set the environment variables
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['192.168.0.223:2222', '192.168.0.183:2222']
    },
    'task': {'type': 'worker', 'index': 0}
})

# Create a TensorFlow cluster resolver
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

# Define the strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)

# Initialize the AutoKeras model
with strategy.scope():
    clf = run_model()

# Print the distribution strategy
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Train the AutoKeras model
clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, shuffle=False, batch_size=256, callbacks=callbacks)

# Evaluate the model but if there is an error, clear the session and try again
try:
    print("Evaluating model...")
    predictions = clf.predict(x_test)
    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
    print(f"Percentage error: {error:.2f}")
except:
    print("Error evaluating model, clearing session and trying again...")
    tf.keras.backend.clear_session()
    clf = run_model()
    print("Evaluating model...")
    predictions = clf.predict(x_test)
    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
    print(f"Percentage error: {error:.2f}")