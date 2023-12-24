#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1995)
torch.manual_seed(1995)


# Generate data
x_obs = np.linspace(-10, 10, 1000)
noise = 0.1 * np.random.randn(x_obs.shape[0])
y_obs = x_obs/25 + 2*x_obs*np.cos(x_obs)/(1+x_obs*x_obs) + noise

x_true = np.linspace(-10, 10, 1000)
y_true = x_true/25 + 2*x_true*np.cos(x_true)/(1+x_true*x_true)

# Set plot limits and labels
xlims = [-10, 10]
ylims = [-1, 1]


data = np.vstack((x_obs,y_obs))
train_set, val_set = torch.utils.data.random_split(data.T, [500,500])


# Create plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_true, y_true, 'b-', linewidth=3, label="True function")
ax.plot(x_obs, y_obs, 'ko', markersize=4, label="Observations")
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_xlabel("X", fontsize=30)
ax.set_ylabel("Y", fontsize=30)
ax.legend(loc=4, fontsize=15, frameon=False)

plt.show()


# In[2]:


import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.nn import PyroParam


# In[37]:


# Convert data to PyTorch tensors
x_train = torch.from_numpy(train_set[:][:,0].T).float()
y_train = torch.from_numpy(train_set[:][:,1].T).float()
x_train_dnn = x_train.reshape(-1,1)
y_train_dnn = y_train.reshape(-1,1)
x_train.shape


# In[127]:


from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        torch.manual_seed(1)
        self.linear = torch.nn.Linear(1,1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 10),
#             nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(10, 10),
#             nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        torch.nn.init.uniform_(self.linear.weight)
    def forward(self, x):
#         x = x.reshape(-1, 1)
#         print(x)
        out = self.linear_relu_stack(x)
        #out = self.linear(x)
        return out


# In[128]:


learningRate = 0.25
epochs = 30000

model = linearRegression()
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


# In[129]:


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


# In[130]:


x_test_dnn = torch.from_numpy(val_set[:][:,0].reshape(-1,1)).float()
#print(x_test)
y_test_dnn = torch.from_numpy(val_set[:][:,1].reshape(-1,1)).float()


# In[131]:


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


# In[132]:


class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, feature_number=5,hid_dim=10, n_hid_layers=2, prior_scale=5., x_scale=2):
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
        #omega02 = torch.from_numpy(np.random.standard_cauchy([2*hid_dim,feature_number])).float()
        #omega02 = torch.rand([2*hid_dim,feature_number])
        omega02 = torch.from_numpy(np.random.laplace(0,1, [2*hid_dim,feature_number])).float()
        omega03 = torch.randn([2*hid_dim,feature_number])
        #omega04 = torch.from_numpy(np.random.standard_cauchy([2*hid_dim,feature_number])).float()
        omega04 = torch.from_numpy(np.random.laplace(0,1, [2*hid_dim,feature_number])).float()
        #omega04 = torch.rand([2*hid_dim,feature_number])
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
        #self.W1 = torch.from_numpy(np.random.standard_cauchy([2*feature_number,hid_dim])).float()
        self.W1 = torch.randn([2*feature_number,hid_dim])
        self.W2 = torch.randn([2*feature_number,hid_dim])
        #self.W3 = torch.from_numpy(np.random.standard_cauchy([2*feature_number,hid_dim])).float()
        self.W3 = torch.randn([2*feature_number,hid_dim])
        self.W4 = torch.randn([2*feature_number,out_dim])
        
        self.bias0 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias1 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias2 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.bias3 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))

            
    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
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


# In[133]:


from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange
from pyro.infer import Predictive
pyro.clear_param_store()

model = BNN(hid_dim=2, feature_number=5, n_hid_layers=1, prior_scale=5)
mean_field_guide = AutoDiagonalNormal(model)
#optimizer = pyro.optim.Adam({"lr": .01})

num_steps = 30000
initial_lr = 0.02
gamma = 0.01  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
optimizer = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
pyro.clear_param_store()

num_epochs = num_steps
progress_bar = trange(num_epochs)


mse = nn.MSELoss()
testinput = torch.from_numpy(val_set[:][:,0].T).float()
target = torch.from_numpy(val_set[:][:,1].T).float()
predictive = Predictive(model, guide=mean_field_guide, num_samples=500)

mselist = []
losslist = np.array([0]*num_epochs)

for epoch in progress_bar:
    loss = svi.step(x_train, y_train)
    losslist[epoch] = loss
    
    if epoch%100 == 0:
        preds = predictive(testinput)
        pred_y = preds['obs'].T.detach().mean(axis=1)
        output = mse(pred_y, target)
        mselist.append(output)
    
    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")


# In[134]:


from pyro.infer import Predictive
x_test = torch.linspace(xlims[0], xlims[1], 3000)


# In[135]:


def plot_predictions(preds):
    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
#    xlims = [-0.5, 1.5]
#    ylims = [-1.5, 2.5]
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)
    
#     plt.title("prediction by two kernel")

    #ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    ax.plot(x_obs, y_obs, 'ko', markersize=4, label="observations")
    ax.plot(x_obs, y_obs, 'ko', markersize=3)
    ax.plot(x_test, y_pred, '-', linewidth=3, color="C0", label="predictive mean by KE-DGP")
#     ax.fill_between(x_test, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='C0', zorder=5)

    plt.legend(loc=4, fontsize=15, frameon=False)


# In[136]:


predictive = Predictive(model, guide=mean_field_guide, num_samples=500)
preds = predictive(x_test)
plot_predictions(preds)
plt.plot(x_test_dnn, predicted, 'rx', label='Predictions by DNN', alpha=0.5)
plt.legend(loc=4, fontsize=12, frameon=False)
plt.rcParams.update({'font.size': 22})
plt.xlabel("input")
plt.ylabel("output")
ax.set_xticks([-1,-.5,0,.5,1])
ax.set_yticks([-.5,0,0.5,1,1.5])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




