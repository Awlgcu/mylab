import blzs.Perceptron as Perceptron
import blzs.DataLoader as DataLoader
import numpy as np
import matplotlib.pyplot as plt
size1 ,size2 = 50, 30
p1 = np.random.rand(2,size1) + 0.5
p2 = -np.random.rand(2,size2) + 0.5

train_dataset = []
for i in range(p1.shape[1]):
    train_dataset.append((p1[:,i], 0))
for i in range(p2.shape[1]):
    train_dataset.append((p2[:,i], 1))

train_data = DataLoader.Data(train_dataset,random_shuffle=True)
net = Perceptron.Perceptron(2,1,bias=True)
net.train(train_data.inputs,train_data.labels,lr = 0.01, num_epoch = 10)

pred = net.Predict(train_data.inputs)
for i in range(size1+size2):
    if int(pred[0,i])!= int(train_data.labels[0,i]):
        print(i)
print(net.w)
plt.figure()
plt.scatter(p1[0,:], p1[1,:], color='red')
plt.scatter(p2[0,:], p2[1,:], color='blue')
x = np.linspace(-2,2,100)
plt.plot(x,(-net.w[0,0]*x-net.b[0,0])/net.w[0,1])
plt.show()