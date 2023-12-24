#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib as plt


# In[40]:


x_test = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtest__FOLD_1.txt")
x_train = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtrain__FOLD_1.txt")
y_test = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytest__FOLD_1.txt")
y_train = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytrain__FOLD_1.txt")


# In[41]:


print(x_test.shape,y_test.shape)


# In[42]:


import torch
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sklearn
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import make_friedman2

# Set random seed for reproducibility
np.random.seed(1995)
torch.manual_seed(1995)
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.nn import PyroParam


# In[77]:


from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        torch.manual_seed(1)
        self.linear = torch.nn.Linear(13,10)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 10),
#             nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
#             nn.ReLU(),
            nn.Linear(10,1)
        )
        torch.nn.init.uniform_(self.linear.weight)
    def forward(self, x):
#         x = x.reshape(-1, 1)
#         print(x)
        x1 = self.linear(x)
        out = self.linear_relu_stack(x1)
        #out = self.linear(x)
        return out


# In[78]:


# Convert data to PyTorch tensors
x_train_d = torch.from_numpy(x_train).float()
y_train_d = torch.from_numpy(y_train).float()
x_train_dnn = x_train_d#.reshape(-1,1)
y_train_dnn = y_train_d#.reshape(-1,1)
x_train.shape


# In[79]:


learningRate = 0.001
epochs = 10000

model = linearRegression()
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


# In[80]:


for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(x_train_dnn.cuda())
        labels = Variable(y_train_dnn.cuda())
        #labels = labels.reshape(-1, 1)
        #inputs = Variable(torch.from_numpy(x_train).cuda())
        #labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(x_train)
        labels = Variable(y_train)
        #labels = labels.reshape(-1, 1)
        #inputs = Variable(torch.from_numpy(x_train))
        #labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)
#     print(outputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    #print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()
    if (epoch-1)%1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[81]:


x_test_dnn = torch.from_numpy(x_test).float()
y_test_dnn = torch.from_numpy(y_test).float()


# In[68]:


with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(x_test_dnn.cuda()).cpu().data.numpy()
    else:
        predicted = model(x_test).data.numpy()
    #print(predicted)

plt.clf()
plt.plot(x_test_dnn, y_test_dnn, '.', label='True data', alpha=0.5)
plt.plot(x_test_dnn, predicted, 'rx', label='Predictions', alpha=0.5)

plt.legend(loc=4)
plt.xlabel("input")
plt.ylabel("output")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




