import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __len__(self,):
        pass

    def __getitem__(self, key):
        pass


class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size=None, shuffle=False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(self.dataset)
        if self.batch_size:
            self.size = np.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self):
        if self.shuffle:
            np.random.seed()
            np.random.choice(range(len(self.dataset)), size=len(self.dataset), replace=False)
        self._idx=-1
        return self

    def __next__(self):
        self._idx += 1
        return self.dataset[]


class Iter(object):
    def __init__(self) -> None:
        pass
    
    def __iter__(self):
        self.x = 1
        return self

    def __next__(self):
        self.x +=1
        return self.x

print( np.random.choice(range(10), size=10, replace=False))