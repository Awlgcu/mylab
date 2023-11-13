import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat


######################################################定义不同模块
def sigmoid(x):
    # return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    # return x*(1-x)
    return sigmoid(x)*(1-sigmoid(x))
def relu(x):
    return np.where(x >= 0, x, 0)
def d_relu(x):
    return np.where(x >= 0, 1, 0)
#定义损失
def mse(a, y):
    return np.sum((a-y)**2)/2
def d_mse(a, y):
    return a-y

class dataloader():
    def __init__(self, data, *, ratio, shuffle):
        x, y = data
        self.x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        self.y = y
        total_num = self.x.shape[1]
        knife = math.ceil(total_num*ratio[0])
        if shuffle:
            indexes = np.random.permutation(total_num)
        else:
            indexes = np.arange(0, total_num, 1)
        train_slice = indexes[0:knife]
        test_slice = indexes[knife: total_num]
        self.train_data = self.x[:, train_slice]
        self.train_label = self.y[:, train_slice]
        self.test_data = self.x[:, test_slice]
        self.test_label = self.y[:, test_slice]

class model:
    def __init__(self, layers=None, act_func=None, weight_define='Gauss'):
        self.for_bp_train = False
        self.for_perceptron = False
        self.for_hebb = False
        if layers != None and act_func != None and (len(layers) != (len(act_func)+1)):
            print("层数和激活函数不匹配")
            sys.exit(0)
        self.layers = layers
        self.act_func = act_func
        self.w = {}
        self.layer_size = 0
        if self.layers != None:
            #layers存在,则打开bp_train的接口
            self.for_bp_train = True
            self.layer_size = len(layers)
        else:
            #否则打开其他两个接口
            self.for_perceptron = True
            self.for_hebb = True
        if weight_define == 'Zeros':
            for l in range(1, self.layer_size):
                self.w[l] = np.zeros((layers[l], layers[l - 1]))
        elif weight_define == 'Gauss':
            for l in range(1, self.layer_size):
                self.w[l] = np.random.randn(layers[l], layers[l - 1])
        elif weight_define == 'Xavier':
            for l in range(1, self.layer_size):
                self.w[l] = np.random.randn(layers[l], layers[l - 1]) * math.sqrt(1 / layers[l - 1])
                # self.w[l] = np.random.randn(layers[l], layers[l - 1]) * math.sqrt(6 / (layers[l] + layers[l - 1]))
        elif weight_define == 'He':
            for l in range(1, self.layer_size):
                self.w[l] = np.random.randn(layers[l], layers[l - 1]) * math.sqrt(2 / layers[l - 1])
#bp算法的实现
    def forward(self, data):
        a = {}
        z = {}
        a[0] = data
        for i in range(1, self.layer_size):
            z[i] = np.dot(self.w[i], a[i - 1])
            f = None
            if self.act_func[i-1] == 'sigmoid':
                f = sigmoid
            elif self.act_func[i-1] == 'relu':
                f = relu
            a[i] = f(z[i])
        return a, z
    def backward(self, a, y, z):
        delta = {}
        d_f = None
        if self.act_func[self.layer_size-1-1] == 'sigmoid':
            d_f = d_sigmoid
        elif self.act_func[self.layer_size-1-1] == 'relu':
            d_f = d_relu
        delta[self.layer_size] = d_mse(a[self.layer_size-1], y)*d_f(z[self.layer_size-1])
        for i in range(1, self.layer_size-1):
            if self.act_func[self.layer_size - 2 - i] == 'sigmoid':
                d_f = d_sigmoid
            elif self.act_func[self.layer_size - 2 - i] == 'relu':
                d_f = d_relu
            delta[self.layer_size-i] = np.dot(self.w[self.layer_size-i].T, delta[self.layer_size-i+1]) * d_f(z[self.layer_size-i-1])
        return delta
    # 开始更新参数
    def update_weight(self, a, delta, alpha):
        new_w = {}
        for i in range(1, len(self.w) + 1):
            new_w[i] = self.w[i] - alpha * np.dot(delta[i + 1], a[i - 1].T)
        return new_w
    def bp_train(self, train_data, train_label, *, alpha, num_epoch, show_ratio=0.0, batch_size, validate_ratio=0.0):
        if self.for_bp_train == False:
            print("网络定义不匹配bp算法需要的内容")
            sys.exit(0)
        self.for_perceptron = False
        self.for_hebb = False
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
            train_precession = 0
            for i in range(num_batch):
                #随机选择样本,还可以优化
                start = i * batch_size
                end = min((i + 1) * batch_size, data_size)
                indice = indexes[start:end]
                batch_data = train_data[:, indice]
                batch_label = train_label[:, indice]
                a, z = self.forward(batch_data)
                delta = self.backward(a, batch_label, z)
                self.w = self.update_weight(a, delta, alpha)
                train_j += mse(a[self.layer_size - 1], batch_label)
                train_precession += self.validate(a[self.layer_size - 1], batch_label)

            if (show_ratio != 0) and (epoch % (num_epoch * show_ratio) == 0):
                train_j = train_j/data_size
                train_precession = train_precession/num_batch
                print("训练集损失:{},训练集准确率{}".format(train_j, train_precession))
            if (validate_ratio != 0) and (show_ratio != 0) and (epoch % (num_epoch * show_ratio) == 0):
                validate_a, _ = self.forward(validate_data)
                validate_j = mse(validate_a[self.layer_size-1], validate_label)
                validate_precession = self.validate(validate_a[self.layer_size-1], validate_label)
                print("验证集损失:{},验证集准确率{}".format(validate_j/validate_data.shape[1], validate_precession))
#用于计算正确率的函数
    def validate(self, x, y):
        correct = 0
        for i in range(x.shape[1]):
            if x[:, i].argmax() == y[:, i].argmax():
                correct += 1
        precession = correct/x.shape[1]
        return precession
#感知机算法的实现
    def perceptron(self, data, label, alpha, num_epoch, show_ratio=0.0):
        if self.for_perceptron == False:
            print("网络定义不匹配感知机算法需要的内容")
            sys.exit(0)
        self.for_bp_train = False
        self.for_hebb = False
        #归一化处理
        data = self.normalization(data)
        #增加一列b
        data = np.hstack((data, np.ones((data.shape[0], 1))))
        self.w = np.random.randn(data.shape[1], 1)
        #开始训练
        for i in range(num_epoch):
            s = np.dot(data, self.w)  # 40*3, 3*1
            y_pred = np.ones((data.shape[0], 1))
            index = np.where(s < 0)[0]
            y_pred[index, :] = -1
            num = len(np.where(label != y_pred)[0])
            if (show_ratio != 0) and (i % (num_epoch*show_ratio) == 0):
                precession = (label.shape[0]-num)/label.shape[0]
                print("第{}轮测试, 正确率{}".format(i/(num_epoch*show_ratio)+1, precession))
            if num == 0:
                print("任务完成")
                return self.w
            else:
                t = np.where(label != y_pred)[0][0]
                self.w += alpha * label[t] * data[t, :].reshape(3, 1)
            if(i==num_epoch-1):
                print("迭代完成,未能完全分类")
                return self.w
    def perceptron_forward(self, data):
        # 归一化处理
        data = self.normalization(data)
        # 增加一列b
        data = np.hstack((data, np.ones((data.shape[0], 1))))
        s = np.dot(data, self.w)  # 40*3, 3*1
        y_pred = np.ones((data.shape[0], 1))
        index = np.where(s < 0)[0]
        y_pred[index, :] = -1
        return y_pred
    def normalization(self, data):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        return (data-mean)/std
#hebb算法的实现



#用于测试集的测试
    def eval(self, x, y):
        if self.for_perceptron == True:
            #感知机的预测
            x = self.normalization(x)
            x = np.hstack((x, np.ones((x.shape[0], 1))))
            s = np.dot(x, self.w)  # 40*3, 3*1
            y_pred = np.ones((x.shape[0], 1))
            index = np.where(s < 0)[0]
            y_pred[index, :] = -1
            num = len(np.where(y != y_pred)[0])
            precession = (y.shape[0] - num) / y.shape[0]
            print("测试集正确率为:{}".format(precession))
        elif self.for_bp_train == True:
            #bp预测
            a, _ = self.forward(x)
            j = mse(a[self.layer_size - 1], y)
            precession = self.validate(a[self.layer_size - 1], y)
            print("测试集损失:{},测试集准确率{}".format(j/x.shape[1], precession))
        elif self.for_hebb == True:
            pass

#####################################################################  测试
#bp_trian
#准备数据
m = loadmat("./mnist_small_matlab.mat")
x, y = m["trainData"], m['trainLabels']
# print(x.shape, y.shape)
data = (x, y)
processed_data = dataloader(data,  ratio=[0.9, 0.1], shuffle=True)
train_data = processed_data.train_data
train_label = processed_data.train_label
test_data = processed_data.test_data
test_label = processed_data.test_label
#定义网络测试
net = model([784, 256, 64, 10], ['sigmoid', 'sigmoid', 'sigmoid'], weight_define='Gauss')
net.bp_train(train_data, train_label, alpha=0.05, num_epoch=100, show_ratio=0.1, batch_size=128, validate_ratio=0)
net.eval(test_data, test_label)

#perceptron
#准备数据
data_train = np.zeros((100, 2))
data_train[0:50, 0] = np.random.randint(0, 10, 50)
data_train[0:50, 1] = np.random.randint(0, 10, 50)
data_train[50:, 0] = np.random.randint(10, 20, 50)
data_train[50:, 1] = np.random.randint(10, 20, 50)
y_train = np.zeros((100, 1))
y_train[:50, 0] = np.ones(50)
y_train[50:, 0] = -1*np.ones(50)

data_test = np.zeros((40, 2))
data_test[0:20, 0] = np.random.randint(0, 10, 20)
data_test[0:20, 1] = np.random.randint(0, 10, 20)
data_test[20:, 0] = np.random.randint(10, 20, 20)
data_test[20:, 1] = np.random.randint(10, 20, 20)
y_test = np.zeros((40, 1))
y_test[:20, 0] = np.ones(20)
y_test[20:, 0] = -1*np.ones(20)
y_test[3, 0] = -1
y_test[4, 0] = -1
y_test[21, 0] = 1
# #定义网络测试
# net1 = model()
# w = net1.perceptron(data_train, y_train, alpha=0.1, num_epoch=100, show_ratio=0.1)
# net1.eval(data_test, y_test)
# #尝试打印结果
# y_pred = net1.perceptron_forward(data_test)

# #二维可视化
# x1, x2 = -2, 2
# y1 = -1/w[1]*(w[2]*1+w[0]*x1)
# y2 = -1/w[1]*(w[2]*1+w[0]*x2)
# data_train = net1.normalization(data_train)
# data_test = net1.normalization(data_test)
# plt.scatter(data_train[0:50, 0], data_train[0:50, 1], color='b', marker='o', label='train blue points')
# plt.scatter(data_train[50:, 0], data_train[50:, 1], color='r', marker='o', label='train red points')
# plt.scatter(data_test[0:20, 0], data_test[0:20, 1], color='b', marker='x', label='test blue points')
# plt.scatter(data_test[20:, 0], data_test[20:, 1], color='r', marker='x', label='test red points')
# plt.plot([x1, x2], [y1, y2], color='r')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(loc='upper left')
# plt.show()