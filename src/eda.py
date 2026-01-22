import pandas as pd
import matplotlib.pyplopt as plt
from data_processing import DataCleaner

def merge():
    dC = DataCleaner("../data/raw")
    files = dC.file_paths
    all_dfs = []
    for file in files:
        df = pd.read_csv(file)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    print(full_df.shape)
    print(full_df.head())


if __name__=="__main__":
    merge()