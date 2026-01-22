import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import os
from pathlib import Path

class DataCleaner:

    def __init__(self,path):
        self.path = path
        self.file_paths = []
        self.cleaned=[]
        self.dfs=[]
        self._data=None
        self.data={
            '1':'../data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            '2':'../data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            '3':'../data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            '4':'../data/raw/Monday-WorkingHours.pcap_ISCX.csv',
            '5':'../data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            '6':'../data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            '7':'../data/raw/Tuesday-WorkingHours.pcap_ISCX.csv',
            '8':'../data/raw/Wednesday-workingHours.pcap_ISCX.csv'
        }

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
                    self.cleaned.append(clean_cols)
                except Exception as e:
                    print("Err")

    def _dbg(self,x,d='\n'):
        print(x,end=f'{d}')

    def _merge(self):
            all_dfs = []    
            print(self.cleaned)
            full_df = pd.concat(self.cleaned, ignore_index=True)

            print(full_df.shape)
            print(full_df.head())


    def _get_dims(self):
        print('Data dimensions: ')
        
        for k,v in self.data.items():
            d = pd.read_csv(v)
            self.dfs.append(d)
            rows, cols = d.shape
            print(f'Data -> {rows} rows, {cols} columns')
        self._data = pd.concat(self.dfs)
        r,c = self._data.shape
        print("NEW DIMENSTION")

        print("ROWS : ",r)
        print("COLS : ",c)
        print("Total : ",r*c)

    def _delete_fs(self,ok=True):
        for d in self.dfs: del d 

    def _preprocess_columns(self):
        col_names = {col : col.strip() for col in self._data.columns}
        self._data.rename(columns = col_names , inplace = True)
        self._dbg(self._data.columns)
        print("DATA INFO : ")
        self._dbg(self._data.info)

    def _describe(self):
        self._get_dims()
        self._delete_fs()
        self._preprocess_columns()
        self._dbg(self._data.describe().transpose())


if __name__=="__main__":
    dC=DataCleaner("../data/raw")
    dC._describe()