import pandas as pd
import datetime
import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split

# Set up TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='eu-pod')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
print("Got past initialization")

# Set environment variables
os.environ['TPU_NAME'] = 'eu-pod'
os.environ['TPU_LOAD_LIBRARY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'KerasAlpha1'
print("Project name: ", project_name)

# Read in the parquet file
train = pd.read_parquet('./COMBINED_scaled2.parquet')

train = train.astype('float32')

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = train.drop([target_col], axis=1).values
y_train = train[target_col].values

lookback = 2560

def create_dataset(x, y, lookback=1):
    x_data, y_data = [], []
    for i in range(len(x) - lookback):
        x_data.append(x[i:(i + lookback), :])
        y_data.append(y[i + lookback])
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

x_train, y_train = create_dataset(x_train, y_train, lookback)

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

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                          activation='relu', input_shape=(lookback, x_train.shape[2])))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
        loss='mae',
        metrics=['mape']
    )

    return model

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=3,
    directory='./models',
    project_name=project_name
)

tuner.search_space_summary()

# Train the Keras model
with strategy.scope():
    tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val), shuffle=False, batch_size=64, callbacks=callbacks)

# Get the best model
best_model = tuner.get_best_models(1)[0]
best_model.summary()

# Save the best model
best_model.save(f'./best_model_{currentTime}.h5')
