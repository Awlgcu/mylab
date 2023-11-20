import blzs.Model as Model
import numpy as np
import matplotlib.pyplot as plt
import blzs.DataLoader as DataLoader

from torchvision.datasets import MNIST
import PIL
train_dataset=MNIST(root=r"C:\Users\lyz\Desktop\Python\NNDP\All",download=True,train=True)
test_dataset=MNIST(root=r"C:\Users\lyz\Desktop\Python\NNDP\All",download=True,train=False)
# train_size = len(train_dataset)
# test_size = len(test_dataset)
# train_inputs = np.zeros(shape=(784,train_size))
# train_targets = np.zeros(shape=(10,train_size))
# for i in range(train_size):
#     train_inputs[:,i] = np.array(train_dataset[i][0]).reshape(1,-1)
#     train_targets[train_dataset[i][1],i] = 1
#net.TrainBP(train_data=train_inputs,train_label=train_targets,num_epoch=50,alpha=0.05,batch_size=128,show_ratio=0.1,validate_ratio=0.1)

#data
train_data=DataLoader.Data(train_dataset,random_shuffle=False,label_transform=True)
test_data=DataLoader.Data(test_dataset,random_shuffle=True,label_transform=True)
#define NN
net=Model.Model(layers=[784,128,32,10],act_func=["Sigmoid","Sigmoid","Sigmoid"],weight_define="Gauss")
net.TrainBP(train_data=train_data.inputs,train_label=train_data.labels,num_epoch=50,alpha=0.05,batch_size=128,show_ratio=0.1,validate_ratio=0.1,loss_func="CrossEntropy")
net.Test(test_data=test_data.inputs,test_label=test_data.labels)
subsize = 10
pred = net.Predict(train_data.inputs[:,0:subsize])
tr = [train_dataset[i][1] for i in range(subsize)]
print(pred,tr)