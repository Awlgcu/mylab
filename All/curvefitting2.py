import blzs.Model as Model
import blzs.DataLoader as DataLoader
import numpy as np
import matplotlib.pyplot as plt
sample_num=1000
x=np.linspace(-4,4,sample_num)
y=np.sin(x)

train_size=50
train_x=np.linspace(-4,4,train_size)
train_y=np.sin(train_x)

train_dataset=[(train_x[i],train_y[i]) for i in range(train_size)]
train_data=DataLoader.Data(train_dataset)


net=Model.Model([1,20,8,1],['Tanh','Tanh','Linear'],weight_define='Xavier')
net.TrainBP(train_data.inputs,train_data.labels,alpha=0.01,batch_size=1,num_epoch=500)
a,z=net.forward(np.linspace(-4,4,sample_num).reshape(-1,sample_num))
plt.figure()
plt.plot(x,a[3].reshape(-1),color='blue',label='predict curve')
plt.plot(x,y,color='red',label='true curve')
plt.legend()
plt.show()