# import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __len__(self,):
        pass

    def __getitem__(self, key):
        pass


class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size=None, shuffle=False, random_state=None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_flag = shuffle
        self.size = len(self.dataset)
        if self.batch_size:
            self.batch_number = np.ceil(len(self.dataset) / self.batch_size)
        self.seed(random_state)
        
            
    def seed(self, x):
        np.random.seed(x)
            
    def shuffle(self):
        self.dataset = self.dataset[np.random.choice(range(len(self.dataset)), size=len(self.dataset), replace=False)]
    
    def __iter__(self):
        if self.shuffle:
            self.shuffle()
            # np.random.seed()
            # self.dataset = self.dataset[np.random.choice(range(len(self.dataset)), size=len(self.dataset), replace=False)]
        self._idx=-1
        return self

    def __next__(self):
        self._idx += 1
        
        if self._idx >= self.batch_number:
            raise StopIteration 
        
        sample = self.dataset[self._idx * self.batch_size: min((self._idx+1) * self.batch_size, self.size)]
        return sample[:, :-1],  sample[:,-1]
        
        
        # return self.dataset[]
        
    
    

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