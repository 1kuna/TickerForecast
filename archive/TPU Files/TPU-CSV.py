import autokeras as ak
import datetime
import os
import tensorflow as tf
import pandas as pd

# # Set up TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))
# print("Got past initialization")

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set automatic Mixed Precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

project_name = 'CSVTEST'
print("Project name: ", project_name)

train = pd.read_csv(get_file_path('intraday/TICKERS2', filename=f'COMBINED_scaled.csv'))

# Read in the CSV file using tf.data API
def load_dataset(csv_file, target_col_index, batch_size):
    dataset = tf.data.experimental.CsvDataset(
        csv_file,
        [tf.float32] * (len(train.columns)),  # replace 'train.columns' with the number of columns in your CSV
        field_delim=',',
        header=True,
    )

    def parse_csv(*fields):
        features = list(fields)
        target = features.pop(target_col_index)
        return tf.stack(features), target

    dataset = dataset.map(parse_csv)
    dataset_x = dataset.map(lambda x, y: x)
    dataset_y = dataset.map(lambda x, y: y)
    dataset_x = dataset_x.batch(batch_size)
    dataset_y = dataset_y.batch(batch_size)
    return dataset_x, dataset_y

csv_file = get_file_path('intraday/TICKERS2', filename=f'COMBINED_scaled.csv')
target_col = 1
batch_size = 1024
train_dataset = load_dataset(csv_file, target_col, batch_size)

# Define callbacks
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50
)

checkpoint_callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir=get_file_path('models/checkpoints'),
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

# with strategy.scope():
#     clf
# print("Past strategy scope")

# Train the AutoKeras model
clf.fit(train_dataset, epochs=None, shuffle=False, callbacks=callbacks)
