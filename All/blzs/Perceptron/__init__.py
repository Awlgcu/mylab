import numpy as np
class Perceptron:
    class Hardlim:
        def F(self,x):
            return np.where(x>=0, 1, 0)
        def dF(self,x):
            return np.zeros_like(x)
    def __init__(self, l1, l2, act_func = "Hardlim", bias = False):
        self.l1 = l1
        self.l2 = l2
        self.w = np.zeros(shape = (l2, l1))
        self.b = None
        self.act_func = None
        if bias == True:
            self.b = np.zeros(shape = (l2, 1))
        if act_func == "Hardlim":
            self.act_func = self.Hardlim()
    def Normalization(self, data):
        mean = np.sum(data, axis = 1)
        std = np.std(data, axis = 1)
        for i in range(data.shape[1]):
            data[:,i] = (data[:,i] - mean)/std
        return data
    def train(self, train_data, train_label, lr = 0.01, num_epoch = 10):
        assert len(train_label.shape) == 2, f"train_label shape error: {train_label.shape}"
        train_size = train_data.shape[1]
        for epoch in range(num_epoch):
            for i in range(train_size):
                a = None
                if self.b != None:
                    a = self.act_func.F(np.dot(self.w, train_data[:,i]) + self.b)
                else:
                    a = self.act_func.F(np.dot(self.w, train_data[:,i]))
                e = train_label[:,i] - a
                if len(e.shape) == 1:
                    e = e.reshape(1,-1)
                self.w += np.dot(e, train_data[:,i].reshape(1,-1))
                if self.b != None:
                    self.b += e
    def Predict(self, data):
        assert len(data.shape) == 2, f"data shape error: {data.shape}"
        assert data.shape[0] == self.l1, f"input dim: {self.l1} != data's row number: {data.shape[0]}"
        if self.b != None:
            a = self.act_func.F(np.dot(self.w, data) + self.b)
        else:
            a = self.act_func.F(np.dot(self.w, data))
        return a