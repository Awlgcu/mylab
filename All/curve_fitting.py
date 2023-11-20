
import blzs.DataLoader as DataLoader
import blzs.Model as Model
import numpy as np
import matplotlib.pyplot as plt

#just using Model,not Data
sample_num=1000
x=np.linspace(-4,4,sample_num)
y=np.sin(x)
train_size=50
train_x=np.linspace(-4,4,train_size)
train_y=np.sin(train_x)
train_x=train_x.reshape(-1,train_size)
train_y=train_y.reshape(-1,train_size)

net=Model.Model([1,20,8,1],['Tanh','Tanh','Linear'],weight_define='Xavier')
net.TrainBP(train_x,train_y,alpha=0.01,batch_size=1,num_epoch=500)
a,z=net.forward(np.linspace(-4,4,sample_num).reshape(-1,sample_num))
plt.figure()
plt.plot(x,a[3].reshape(-1),color='blue',label='predict curve')
plt.plot(x,y,color='red',label='true curve')
plt.legend()
plt.show()


sample_num=1000
x=np.linspace(-4,4,sample_num)
y=np.exp(x)
train_size=100
train_x=np.linspace(-4,4,train_size)
train_y=np.exp(train_x)
train_x=train_x.reshape(-1,train_size)
train_y=train_y.reshape(-1,train_size)
net=Model.Model([1,15,25,12,1],['Sigmoid','Sigmoid','Sigmoid','Linear'],weight_define='Xavier')
net.TrainBP(train_x,train_y,alpha=0.001,batch_size=1,num_epoch=800)
a,z=net.forward(np.linspace(-4,4,sample_num).reshape(-1,sample_num))
plt.figure()
plt.plot(x,a[net.layer_size-1].reshape(-1),color='blue',label='predict curve')
plt.plot(x,y,color='red',label='true curve')
plt.legend()
plt.show()

