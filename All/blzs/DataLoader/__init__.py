import numpy as np
import math
class Data:
    def __init__(self, data, random_shuffle = False, label_transform = False):
        data_size = len(data)
        input_dim = 1
        d0 = np.array(data[0][0])
        for i in range(len(d0.shape)):
            input_dim *= d0.shape[i]
        
        self.inputs = np.zeros(shape = (input_dim, data_size))
        self.one_hot = label_transform
        self.labels = None
        if self.one_hot == True:
            #labels: like 0-9 (not 1-10)
            label_dim = int(max(data[i][1] for i in range(data_size))) + 1
            self.labels = np.zeros(shape = (label_dim, data_size)) #one-hot
        else:
            self.labels = np.zeros(shape = (1, data_size))
        for i in range(data_size):
            self.inputs[:,i] = np.array(data[i][0]).reshape(1,-1)
            if self.one_hot == True:
                self.labels[data[i][1],i] = 1
            else:
                self.labels[0,i] = data[i][1]
        if random_shuffle == True:
            idx = np.random.permutation(data_size)
            self.inputs = self.inputs[:,idx]
            self.labels = self.labels[:,idx]
class DataLoader(Data):
    def __init__(self, data, *, ratio, random_shuffle, label_transform = False):
        assert type(data) == list,"type of data is not list"
        data_size = len(data)
        self.TrainData = Data(data[:int(ratio*data_size)], random_shuffle, label_transform)
        self.TestData = Data(data[int(ratio*data_size):], random_shuffle, label_transform)
        