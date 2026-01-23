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


    
    def _box(self):
        plt.figure(figsize = (8, 3))
        sns.boxplot(x = self._data['Flow Bytes/s'])
        plt.xlabel('Boxplot of Flow Bytes/s')
        plt.show()

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
        print("Pre Process")
        self._preprocess_columns()
        self._dbg(self._data.describe().transpose())

    def _process(self):
        self._describe()
        dups = self._data[self._data.duplicated()]
        print(len(dups))
        self._data.drop_duplicates(inplace = True)
        missing_val = self._data.isna().sum()
        numeric_cols = self._data.select_dtypes(include = np.number).columns
        inf_count = np.isinf(self._data[numeric_cols]).sum()
        self._data.replace([np.inf, -np.inf], np.nan, inplace = True)
        missing = self._data.isna().sum()
        mis_per = (missing / len(self._data)) * 100
        mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
        mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})
        sns.set_palette('pastel')
        colors = sns.color_palette()

        missing_vals = [col for col in self._data.columns if self._data[col].isna().any()]

        fig, ax = plt.subplots(figsize = (2, 6))
        msno.bar(self._data[missing_vals], ax = ax, fontsize = 12, color = colors)
        ax.set_xlabel('Features', fontsize = 12)
        ax.set_ylabel('Non-Null Value Count', fontsize = 12)
        ax.set_title('Missing Value Chart', fontsize = 12)
        plt.show()
        

if __name__=="__main__":
    dC=DataCleaner("../data/raw")
    dC._process()
    dC._box()