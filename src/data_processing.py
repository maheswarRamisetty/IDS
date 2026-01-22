import pandas as pd
from pathlib import Path 
import os


class DataCleaner:

    def __init__(self,path):
        self.path = path
        self.file_paths = []
    def _clean(self):

        dir_path = Path(self.path)

        for file_path in dir_path.iterdir():
            if file_path.is_file():
                self.file_paths.append(file_path)
                try:
                    df = pd.read_csv(file_path)
                    clean_cols = df.columns.str.strip().tolist()
                    print("Cleaned Cols : ",len(clean_cols))
                    print(clean_cols)
                except Exception as e:
                    print("Err")


if __name__=="__main__":
    dC=DataCleaner("../data/raw")
    dC._clean()