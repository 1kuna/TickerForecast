import pandas as pd
import autokeras as ak
import datetime
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Read in the paraquet file
data = pd.read_parquet(get_file_path('ticker data', filename="combined.parquet"))

# Define the list of tickers
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'SPY', 'QQQ', 'SHOP',
           'TSLA', 'AMD', 'GOOG', 'META', 'BABA', 'BIDU', 'NFLX',
           'INTC', 'MU', 'BAC', 'JPM', 'WMT',
           'DIS', 'T', 'V', 'MA', 'PYPL', 'SQ', 'DIS']

# Define the list of target columns
target_cols = []
for ticker in tickers:
    target_cols = target_cols + [f"{ticker}_Open"]

# Get the features and target arrays
for col in target_cols:
    x = data.drop(columns=[col])
y = data[target_cols].values

# Split the data into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=False)

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

# Define callbacks list
callbacks = [stopping_callback]

# Initialize the model
def run_model():
    clf = ak.TimeseriesForecaster(
        max_trials=250,
        lookback=24,
        project_name='conf_alpha1',
        overwrite=False,
        directory=get_file_path('models')
    )
    return clf

clf = run_model()

# Train the AutoKeras model
clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=None, shuffle=False, callbacks=callbacks)

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