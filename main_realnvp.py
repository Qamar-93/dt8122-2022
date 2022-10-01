### sampe script as in jupyter notebook but as a python main file

import torch
import torch.nn as nn
from torch.distributions import Normal

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_load import *
from utils.plot import *
from nets import make_net
from flows import RealNVP, ReverseBijection, CouplingBijection, Flow, CNF



dataset_name = "two_blobs"    
train_loader, test_loader = data_load_split(dataset_name, "./data/" ,  32, 0.8)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



flow = RealNVP(bijections=[
  CouplingBijection(make_net()), ReverseBijection(),
  CouplingBijection(make_net()), ReverseBijection(),
  CouplingBijection(make_net()), ReverseBijection(),
  CouplingBijection(make_net()), ReverseBijection(),
  CouplingBijection(make_net()), ReverseBijection(),
  CouplingBijection(make_net()),
], net=make_net()).to(device)



test_data = None
for i,data in enumerate(test_loader):
    if test_data is not None:
        test_data=torch.cat((test_data,data),dim=0)
    else:
        test_data=data.clone()
        
initial_flow_samples=flow.sample(int(len(test_data)))


optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
epochs = 200

print('Training...')
flow = flow.train()
for epoch in range(epochs):
    loss_sum = 0.0
    for i, x in enumerate(train_loader):
        x = x.to(device).float()
        optimizer.zero_grad()
        loss = -flow.flow_outputs(x)[1]
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().cpu().item()
    print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch+1, epochs, loss_sum/len(train_loader)))

# torch.save(flow, "./models/realnvp-{dataset_name}.pt")
# flow = torch.load("./results/realnvp/models/realnvp-{dataset_name}.pt")
flow = flow.eval()


test =None
for i,data in enumerate(test_loader):
    if test is not None:
        test=torch.cat((test,data),dim=0)
    else:
        test=data.clone()

        
x_dash = flow.sample(len(test_data))
z_dash,_ = flow.flow_outputs(test_data.type(torch.float32))

plot_samples(test, "realnvp-{dataset_name}", dataset_name, initial_flow_samples, x_dash, z_dash, range_min=[-4,4], range_max=[-4,4])
