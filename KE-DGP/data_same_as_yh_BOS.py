#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib as plt


# In[51]:


import pandas as pd
import numpy as np

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]


# In[52]:


fxtes = r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtest__FOLD_1.txt"
# fxtr = "C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtrain__FOLD_1"
# fytes = "C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytest__FOLD_1"
# fytr = "C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytrain__FOLD_1"


# In[53]:


x_test = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtest__FOLD_1.txt")
x_train = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_Xtrain__FOLD_1.txt")
y_test = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytest__FOLD_1.txt")
y_train = np.loadtxt(r"C:\Users\yuanq\OneDrive\Desktopold\SB\research\DGP\program\BOS_ARD_ytrain__FOLD_1.txt")


# In[54]:


# data.shape
# X = data
# mean_values = np.mean(X, axis=0)
# std_values = np.std(X, axis=0)

# X_normalized = (X - mean_values) / std_values

# data = X_normalized


# In[55]:


# X = target
# mean_values = np.mean(X, axis=0)
# std_values = np.std(X, axis=0)

# X_normalized = (X - mean_values) / std_values

# target = X_normalized


# In[56]:


# np.set_printoptions(threshold=np.inf)
# print(data)


# In[57]:


# x_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.3)


# In[58]:


# scaler = preprocessing.StandardScaler().fit(x_train)
# # x_train_scaled = scaler.transform(x_train)
# # x_test_scaled = scaler.transform(x_test)
# x_train_scaled = x_train
# x_test_scaled = x_test
# x_test,x_vali,y_test,y_vali = train_test_split(x_test_scaled,y_test,test_size = 0.4)


# In[59]:


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

# Generate data


# In[60]:


import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.nn import PyroParam


# In[61]:


x_train_torch = torch.from_numpy(x_train).float()
y_train_torch = torch.from_numpy(y_train).float()
x_test_torch = torch.from_numpy(x_test).float()
y_test_torch = torch.from_numpy(y_test).float()
x_train.shape


# In[182]:


class BNN(PyroModule):
    def __init__(self, in_dim=13, out_dim=1, feature_number=5,hid_dim=10, n_hid_layers=2, prior_scale=5., x_scale=4):
        super().__init__()
        # Set random seed for reproducibility
        np.random.seed(1995)
        torch.manual_seed(1995)
        
        self.sigome0 = PyroParam(torch.tensor(1.0)/x_scale)
        self.sigome1 = PyroParam(torch.tensor(1.0)/x_scale)
        self.sigome2 = PyroParam(torch.tensor(1.0)/x_scale)
        self.sigome3 = PyroParam(torch.tensor(1.0)/x_scale)
        self.sigome4 = PyroParam(torch.tensor(1.0)/x_scale)
        
        omega00 = torch.randn([in_dim,feature_number])
        omega01 = torch.randn([in_dim,feature_number])
        omega02 = torch.randn([2*hid_dim,feature_number])
        #omega02 = torch.tensor(np.random.standard_cauchy([2*hid_dim,feature_number]))
        omega03 = torch.randn([2*hid_dim,feature_number])
        omega04 = torch.rand([2*hid_dim,feature_number])
        self.omega0 = omega00
        self.omega1 = omega01
        self.omega2 = omega02
        self.omega3 = omega03
        self.omega4 = omega04
    
        self.activation0 = lambda x :torch.hstack((torch.cos(torch.mm(x,self.omega0)*self.sigome0),torch.sin(torch.mm(x,self.omega0)*self.sigome0)))
        self.activation1 = lambda x :torch.hstack((torch.cos(torch.mm(x,self.omega1)*self.sigome1),torch.sin(torch.mm(x,self.omega1)*self.sigome1)))
        self.activation2 = lambda x :torch.hstack((torch.cos(torch.mm(x,self.omega2)*self.sigome2),torch.sin(torch.mm(x,self.omega2)*self.sigome2)))
        self.activation3 = lambda x :torch.hstack((torch.cos(torch.mm(x,self.omega3)*self.sigome3),torch.sin(torch.mm(x,self.omega3)*self.sigome3)))
        self.activation4 = lambda x :torch.hstack((torch.cos(torch.mm(x,self.omega4)*self.sigome4),torch.sin(torch.mm(x,self.omega4)*self.sigome4)))
        
        assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0  # make sure the dimensions are valid

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes_start = [2*feature_number] * 5 + [out_dim]
        self.layer_sizes = [in_dim] + 4*[hid_dim] + [out_dim]

        self.sigW0 = PyroParam(torch.tensor([1.0]*hid_dim))
        self.sigW1 = PyroParam(torch.tensor([1.0]*hid_dim))
        self.sigW2 = PyroParam(torch.tensor([1.0]*hid_dim))
        self.sigW3 = PyroParam(torch.tensor([1.0]*hid_dim))
        self.sigW4 = PyroParam(torch.tensor([1.0]*out_dim))
        
        self.W0 = torch.randn([2*feature_number,hid_dim])
        self.W1 = torch.randn([2*feature_number,hid_dim]).float()
        #self.W1 = torch.from_numpy(np.random.standard_cauchy([2*feature_number,hid_dim])).float()
        self.W2 = torch.randn([2*feature_number,hid_dim])
        self.W3 = torch.randn([2*feature_number,hid_dim]).float()
        #self.W3 = torch.from_numpy(np.random.standard_cauchy([2*feature_number,hid_dim])).float()
        self.W4 = torch.randn([2*feature_number,out_dim])
        
        self.bias0 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias1 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias2 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias3 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))

            
    def forward(self, x, y=None):
        #x = x.reshape(-1, 1)
        x1 = (torch.mm(self.activation0(x),self.W0)*self.sigW0)+self.bias0
        x2 = (torch.mm(self.activation1(x),self.W1)*self.sigW1)+self.bias1
        x = torch.hstack((x1,x2))
        x3 = (torch.mm(self.activation2(x),self.W2)*self.sigW2)+self.bias2
        x4 = (torch.mm(self.activation3(x),self.W3)*self.sigW3)+self.bias3
        x = torch.hstack((x3,x4))
        mu = (torch.mm(self.activation4(x),self.W4)*self.sigW4 ).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(1, 1))  # infer the response noise
        
        with pyro.plate("data", x.shape[0]):
             obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu


# In[232]:


from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange
from pyro.infer import Predictive
pyro.clear_param_store()

model = BNN(hid_dim=5, feature_number=5, n_hid_layers=2, prior_scale=4)
mean_field_guide = AutoDiagonalNormal(model)
#optimizer = pyro.optim.Adam({"lr": .01})

num_steps = 30000
initial_lr = 0.05
gamma = 0.001  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
optimizer = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
pyro.clear_param_store()

num_epochs = num_steps
progress_bar = trange(num_epochs)


mse = nn.MSELoss()
testinput = torch.from_numpy(x_test).float()
target = torch.from_numpy(y_test).float()
predictive = Predictive(model, guide=mean_field_guide, num_samples=500)

mselist = []
losslist = np.array([0]*num_epochs)
interval = 100
for epoch in progress_bar:
    loss = svi.step(x_train_torch, y_train_torch)
    losslist[epoch] = loss
    
    if epoch%interval == 0:
        preds = predictive(testinput)
        pred_y = preds['obs'].T.detach().mean(axis=1)
        output = mse(pred_y, target)
        mselist.append(output)
    
    progress_bar.set_postfix(loss=f"{loss / x_train_torch.shape[0]:.3f}")


# In[233]:


x_train.shape[0]


# In[234]:


print(mselist)


# In[235]:


model.sigW0_unconstrained
fig, ax = plt.subplots(figsize=(10, 5))
#ax.plot(range(num_epochs), losslist/x_train.shape[0])
ax.plot(np.array(range(len(mselist)))*interval, mselist)#/x_train.shape[0])
ax.set_ylim([0,2])


# In[236]:


from pyro.infer import Predictive

k = 7
x_obs = x_test_torch[:,k]
y_obs = y_test_torch


# In[237]:


def plot_predictions(preds):
    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
#    xlims = [-0.5, 1.5]
#    ylims = [-1.5, 2.5]
#     plt.xlim(xlims)
#     plt.ylim(ylims)
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)
    
    plt.title("prediction by two kernel")

    #ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    ax.plot(x_obs, y_obs, 'ko', markersize=4, label="observations")
    #ax.plot(x_obs, y_obs, 'ko', markersize=3)
    ax.plot(x_test_torch[:,k], y_pred, 'o', markersize=4, color="C0", label="predictive mean")
    #ax.fill_between(x_test[:,1], y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='C0', zorder=5)

    plt.legend(loc=1, fontsize=15, frameon=False)


# In[238]:


x_test_torch = torch.from_numpy(x_test).float()
y_test_torch = torch.from_numpy(y_test).float()


# In[239]:


predictive = Predictive(model, guide=mean_field_guide, num_samples=500)
preds = predictive(x_test_torch)
plot_predictions(preds)
plt.rcParams.update({'font.size': 22})
plt.xlabel("input")
plt.ylabel("output")
ax.set_xticks([-1,-.5,0,.5,1])
ax.set_yticks([-.5,0,0.5,1,1.5])


# In[ ]:





# In[ ]:




