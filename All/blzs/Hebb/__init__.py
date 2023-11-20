import numpy as np

class Hebb:
    class Hardlims:
        def F(self,x):
            return np.where(x>=0, 1, -1)
        def dF(self,x):
            return np.zeros_like(x)
    def __init__(self, l1, l2, act_func = "Hardlims"):
        self.l1 = l1
        self.l2 = l2
        if act_func == "Hardlims":
            self.act_func = self.Hardlims()
        self.w = np.zeros(shape = (self.l2, self.l1))
    def train(self, data, data_labels = None, alpha = 0.01, num_epoch = 10):#feature*num
        for epoch in range(num_epoch):
            for i in range(data.shape[1]):
                if len(data_labels):
                    self.w += alpha*np.dot(data_labels[:,i].reshape(-1,1), data[:,i].reshape(1,-1))
                else:
                    a = self.act_func.F(np.dot(self.w, data[:,i]))
                    self.w += alpha*np.dot(a.reshape(-1,1), data[:,i].reshape(1,-1))
    def prediction(self, data):
        pred_data = np.dot(self.w, data)
        return self.act_func.F(pred_data)



