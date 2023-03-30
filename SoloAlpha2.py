import pandas as pd
import autokeras as ak
import datetime
import os
import tensorflow as tf

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set automatic Mixed Precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
print(tf.keras.backend.floatx())

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'TF2'

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Read in the parquet file
dataset = pd.read_parquet(get_file_path('intraday/TICKERS2', filename=f'COMBINED_scaled.parquet'))

# Define the target column
target_col = 'open'

# Get the features and target arrays
x_train = dataset.drop([target_col], axis=1).values
y_train = dataset[target_col].values

stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50
)   

checkpoint_callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir=get_file_path(f'models\\checkpoints\\{project_name}'),
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
        directory=get_file_path('models'),
        metrics='mape',
        loss='mae'
    )
    return clf

clf = run_model()

# Train the AutoKeras model
clf.fit(x_train, y_train, epochs=10, shuffle=False, batch_size=256, callbacks=callbacks)