import numpy as np
import pandas as pd


from dataset import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, filename, **kwargs) -> None:
        # super().__init__(*args, **kwargs)
        
        sep = kwargs["sep"] if "sep" in kwargs else ","
        header = kwargs["header"] if "sep" in kwargs else 0
        self.data = pd.read_csv(filename, sep=sep, header=header).to_numpy(dtype=np.float)        
    
    def __len__(self,):
        return self.data.shape[0]
    
    def __getitem__(self, key):
        return self.data[key, :]
    
    @property
    def shape(self):
        return self.data.shape
        
