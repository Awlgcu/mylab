import mytorch
from mytorch import dataloader
from mytorch import model
from scipy.io import loadmat

sig = mytorch.sigmoid(1)

m = loadmat("./mnist_small_matlab.mat")
x, y = m["trainData"], m['trainLabels']
# print(x.shape, y.shape)
data = (x, y)
processed_data = dataloader(data,  ratio=[0.9, 0.1], shuffle=True)
train_data = processed_data.train_data
train_label = processed_data.train_label

net = model([784, 256, 64, 10], ['sigmoid', 'sigmoid', 'sigmoid'], weight_define='Gauss')
net.bp_train(train_data, train_label, batch_size=128, alpha=0.05, num_epoch=100, validate_ratio=0, show_ratio=0.1)

test_data = processed_data.test_data
test_label = processed_data.test_label
net.eval(test_data, test_label)