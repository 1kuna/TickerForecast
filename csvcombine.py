import pandas as pd
import os
import glob

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

df = pd.DataFrame()

df = pd.concat([pd.read_csv(f, index_col='time') for f in glob.glob(get_file_path('intraday\\g\\val', filename='*.csv'))])

# Drop all rows with NaN values and sort by date and time
df.sort_index(inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv(get_file_path('intraday\\HERE', filename='VAL_COMBINED.csv'))
df.to_parquet(get_file_path('intraday\\HERE', filename='VAL_COMBINED.parquet'))