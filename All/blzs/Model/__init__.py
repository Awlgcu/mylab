
import numpy as np
import math
class Model:
#Loss Function
    class MSE: #ok
        def F(self, outputs,targets):
            return 0.5*np.sum((outputs-targets)**2)
        def dF(self, outputs,targets):
            return (outputs-targets)
    class CrossEntropy: #ok
        def F(self, outputs,targets):
            delta=1e-7       #添加一个微小值可以防止负无限大(np.log(0))的发生。
            f = lambda s:np.exp(s)/np.sum(np.exp(s), axis = 0)
            p = f(outputs)   # 通过 softmax 变为概率分布，并且sum(p) = 1
            return -np.sum(targets*np.log(p+delta))
        def dF(self, outputs,targets):
            f = lambda s:np.exp(s)/np.sum(np.exp(s), axis = 0)
            p = f(outputs)   # 通过 softmax 变为概率分布，并且sum(p) = 1
            return p - targets
     
#Active Function
    class Sigmoid: #ok
        def F(self,x):
            return 1/(1+np.exp(-x))
        def dF(self,x):
            return self.F(x)*(1-self.F(x))
    class ReLU:
        def F(self,x):
            return np.where(x >= 0, x, 0)
        def dF(self,x):
            return np.where(x >= 0, 1, 0)
    class Tanh: #ok
        def F(self,x):
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        def dF(self,x):
            return 1- self.F(x)**2
    class Linear: #ok
        def F(self,x):
            return x
        def dF(self,x):
            return 1
    class Leaky_ReLU:
        def F(self,x):
            return np.where(x>0,x,0.01*x)
        def dF(self,x):
            return np.where(x>0,1,0.01)
    class Hardlims:
        def F(self,x):
            return np.where(x>=0, 1, -1)
        def dF(self,x):
            return np.zeros_like(x)
    class Hardlim:
        def F(self,x):
            return np.where(x>=0, 1, 0)
        def dF(self,x):
            return np.zeros_like(x)
    class Softmax:
        def F(self,x):
            exps = np.exp(x)
            return exps / np.sum(exps)
        def dF(self,x):
            return x - np.sum(x, axis=0)
        
    
    def __init__(self, layers, act_func, weight_define='Gauss'):
        #a:[0,layersize-1], w:[1,layersize-1], act_func:[0,layersize-2]
        self.layers = layers
        self.act_func = {}
        self.w = {}
        self.layer_size = 0
        assert type(layers) == list, "layers is not list"
        assert type(act_func) == list, "act_func size not list"
        self.layer_size = len(layers)
        assert self.layer_size == len(act_func) + 1, f"layer_size: {self.layer_size} != act_func size :{len(act_func)}"
        for l in range(self.layer_size-1):
            if act_func[l] == 'Sigmoid':
                self.act_func[l] = self.Sigmoid()
            elif act_func[l] == "ReLU":
                self.act_func[l] = self.ReLU()
            elif act_func[l] == "Tanh":
                self.act_func[l] = self.Tanh()
            elif act_func[l] == "Linear":
                self.act_func[l] = self.Linear()
            elif act_func[l] == "Leaky_ReLU":
                self.act_func[l] = self.Leaky_ReLU()
            elif act_func[l] == "Softmax":
                self.act_func[l] = self.Softmax()
            elif act_func[l] == "Hardlims":
                self.act_func[l] = self.Hardlims()
            elif act_func[l] == "Hardlim":
                self.act_func[l] = self.Hardlim()
            else:
                assert False, f"act_func name error: {act_func[l]}"
        for l in range(1, self.layer_size):
            if weight_define == 'Zeros':
                self.w[l] = np.zeros((layers[l], layers[l - 1]))
            elif weight_define == 'Gauss':
                self.w[l] = np.random.randn(layers[l], layers[l - 1])
            elif weight_define == 'Xavier':
                self.w[l] = np.random.randn(layers[l], layers[l - 1]) * math.sqrt(6 / (layers[l] + layers[l - 1]))
            elif weight_define == 'He':
                self.w[l] = np.random.randn(layers[l], layers[l - 1]) * math.sqrt(2 / layers[l - 1])
            elif weight_define == 'Uniform':
                self.w[l] = np.random.uniform(0, 1, size = (layers[l], layers[l - 1]))
            else:
                assert False, f"weight_define name error: {weight_define}"
        
    def forward(self, data): #ok
        a = {}
        z = {}
        a[0] = data
        for i in range(1, self.layer_size):
            z[i] = np.dot(self.w[i], a[i - 1])
            a[i] = self.act_func[i-1].F(z[i])
        return a, z
    def backward(self, a, y, z): #ok
        #delta: [2,layer_size]
        delta = {}
        delta[self.layer_size] = self.loss_func.dF(a[self.layer_size-1], y)*self.act_func[self.layer_size-1-1].dF(z[self.layer_size-1])
        for i in range(1, self.layer_size-1):
            delta[self.layer_size-i] = np.dot(self.w[self.layer_size-i].T, delta[self.layer_size-i+1]) * self.act_func[self.layer_size-2-i].dF(z[self.layer_size-i-1])
        return delta
    def compute_gradient(self, a, delta):
        w_grad = {}
        for i in range(1, len(self.w) + 1):
            w_grad[i] = np.dot(delta[i + 1], a[i - 1].T) / a[i-1].shape[1]
        return w_grad
    def gradient_descent(self, w_grad, alpha):
        for i in range(1, len(self.w) + 1):
            self.w[i] -= alpha * w_grad[i]
#BP algorithm
    def TrainBP(self, train_data, train_label, *, batch_size = 1, alpha = 0.01, num_epoch = 10, validate_ratio=0.0, show_ratio=0.0, loss_func='MSE'): #ok
        #Loss Function
        if loss_func == "MSE":
            self.loss_func = self.MSE()
        elif loss_func == "CrossEntropy":
            self.loss_func = self.CrossEntropy()
        else:
            assert False, f"loss_func name error: {loss_func}"
        validate_data = None
        validate_label = None
        #validate不等于0表示要进行验证集划分
        if validate_ratio != 0:
            knife = math.ceil(train_data.shape[1]*0.9)
            validate_data = train_data[:, knife:]
            validate_label = train_label[:, knife:]
            train_data = train_data[:, 0:knife]
            train_label = train_label[:, 0:knife]
        data_size = train_data.shape[1]
        for epoch in range(num_epoch):
            indexes = np.random.permutation(data_size)
            num_batch = math.ceil(data_size / batch_size)
            train_j = 0
            train_acc = 0
            for i in range(num_batch):
                #随机选择样本,还可以优化
                start = i * batch_size
                end = min((i + 1) * batch_size, data_size)
                indice = indexes[start:end]
                batch_data = train_data[:, indice]
                batch_label = train_label[:, indice]
                a, z = self.forward(batch_data)
                delta = self.backward(a, batch_label, z)
                w_grad = self.compute_gradient(a, delta)
                self.gradient_descent(w_grad, alpha)
                train_j += self.loss_func.F(a[self.layer_size - 1], batch_label)
                train_acc += self.Correct(a[self.layer_size - 1], batch_label)

            if (show_ratio != 0) and (epoch % (num_epoch * show_ratio) == 0):
                train_j = train_j/data_size
                train_acc = train_acc/data_size
                print("epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}%".format(epoch, train_j, train_acc*100))
            if (validate_ratio != 0) and (show_ratio != 0) and (epoch % (num_epoch * show_ratio) == 0):
                validate_a, _ = self.forward(validate_data)
                validate_j = self.loss_func.F(validate_a[self.layer_size-1], validate_label)
                validate_acc = self.Accuracy(validate_a[self.layer_size-1], validate_label)
                print("Validate loss: {:.4f} , Validate accuracy: {:.4f}%".format(validate_j/validate_data.shape[1], validate_acc*100))
    def Test(self, test_data, test_label): #ok
        a, _ = self.forward(test_data)
        acc = self.Accuracy(a[self.layer_size-1], test_label)
        print("Test accuracy: {:.4f}%".format(acc*100))
    def Predict(self, data): #ok
        a, _ = self.forward(data)
        pred_size = a[self.layer_size-1].shape[1]
        pred = np.zeros(pred_size)
        for i in range(pred_size):
            pred[i] = a[self.layer_size-1][:, i].argmax()
        return pred

# Indicators
    def Correct(self, x, y): #ok
        correct = 0
        for i in range(x.shape[1]):
            if x[:, i].argmax() == y[:, i].argmax():
                correct += 1
        return correct
    def Accuracy(self, x, y): #ok
        correct = 0
        for i in range(x.shape[1]):
            if x[:, i].argmax() == y[:, i].argmax():
                correct += 1
        acc = correct/x.shape[1]
        return acc


    def Normalization(self, data):
        mean = np.sum(data, axis = 1)
        std = np.std(data, axis = 1)
        for i in range(data.shape[1]):
            data[:,i] = (data[:,i] - mean)/std
        return data