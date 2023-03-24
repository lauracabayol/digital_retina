import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class open_OCTdata():
    def __init__(self, path,drop_cols=None):
        self.path=path
        self.drop_cols=drop_cols
        csv_files = self._sel_files()
        self.data=[]
        self._load_data(csv_files)
        self.N = len(csv_files)
        
        
    def _sel_files(self):
        csv_files = []
        # Loop through all files in the directory
        for filename in os.listdir(self.path):
            # Check if the file is a CSV and starts with the prefix
            if filename.endswith('.csv') and filename.startswith('Num'):
                csv_files.append(os.path.join(self.path, filename))
        return csv_files
    
    def _clear_df(self, df):

        original_list = df.columns.values
        new_list = [element.replace(' ', '').replace('.', '').replace('/', '_') for element in original_list]
        dictionary = dict(zip(original_list, new_list))
        df = df.rename(columns=dictionary)
        df = df.replace(',', '.', regex=True)
        
        self.columns=new_list
        
        
        return df
    
    def _select_columns(self):
        cols = self.columns
        cols = [x for x in cols if x not in self.drop_cols]
        self.columns=cols
    
    
    def _load_data(self,csv_files):
        for i,file in enumerate(csv_files):  
            df_dict = {}
            df = pd.read_csv(file, header=0, sep =';')
            df = self._clear_df(df) 
            self._select_columns()
            #print(df.columns)
            for col in self.columns:
                #print(col)
                df_dict[col] = df.loc[:,str(col)].values[:-2].astype(np.float32)
            self.data.append(df_dict)
        self._store_param_arrays()
            
    def _store_param_arrays(self):
        """ create 1D arrays with all entries for a given parameter. """

        N=len(self.data)
        for col in self.columns:
            setattr(self, f"{col}", np.array([self.data[i][col] for i in range(N)]))
            
            
    def _plot_archive(self,atr1,atr2):
        data_atr1 = getattr(self, atr1)
        data_atr2 = getattr(self, atr2)
        for i in range(self.N):    
            plt.plot(data_atr1[i],data_atr2[i])
            plt.xlabel(f'{atr1}', fontsize = 12)
            plt.ylabel(f'{atr2}', fontsize = 12)
        plt.show()

        return      
        
