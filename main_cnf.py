import glob
from pydoc import visiblename
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.distributions import Normal

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flows import CNF
import torch
import torch.nn as nn
import time 

from utils.plot import *
from utils.data_load import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from nets import HyperNetwork
    

# model and flow parameters
hidden_dim = 32
width= 64
t0 = 0  # flow start time
t1 = 1  # flow end time

# optimization parameters
lr= 3e-3

# model
cnf= CNF(in_out_dim=2, hidden_dim=hidden_dim, width=width)
print(cnf)
ts= torch.tensor([t1, t0]).type(torch.float32)# for training, we flow the samples backward (in time)



def prior():
    return Normal(
        loc=torch.zeros(2),
        scale=torch.ones(2),
    )
p_z0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]),
    covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]])
)


dataset_name= "boomerang"
train_loader, test_loader = data_load_split(dataset_name=dataset_name, path="./data/", batch_size=32, training_frac= 0.8)

#Training
optimizer = torch.optim.Adam(cnf.parameters(), lr=lr)

epochs = 500 # 300

print('Training...')
cnf = cnf.train()
a = 0
z0 = []
start  = time.time()

for epoch in range(1, epochs):
    loss_sum = 0.0
    z_epoch= None
    for i, data in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        x1 = data
        optimizer.zero_grad()
        a += x1.size()[0]
        # initialize initial densities
        logp_diff_t1 = torch.zeros(x1.size()[0], 1).type(torch.float32)

        # compute the backward solutions
        z_t, logp_diff_t = cnf(ts, x1, logp_diff_t1)  # outputs time first
        ####################################################################
        ## z_t is z_dash because ts is t1 t0
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
        z0.append(z_t0.detach().cpu())
        if z_epoch is not None:
            z_epoch=torch.cat((z_epoch,z_t0),dim=0)
        else:
            z_epoch=z_t0.clone()
        # compute the density of each sample
        logp_x = p_z0.log_prob(z_t0) - logp_diff_t0  # .view(-1)
        loss = -logp_x.mean(0)
        loss.sum().backward()
        optimizer.step()
        loss_sum += loss.sum().detach().cpu().item()

    if epoch % 1 == 0:
        print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch, epochs, loss_sum / len(train_loader)))

make_gif("./gif-res-boomerange", "./cnf-res-boomerange")

torch.save(cnf, "./models/cnf_boomerange.pt")
# cnf = torch.load("./models/cnf_two_blobs.pt")
cnf = cnf.eval()

end = time.time()
print('Training finished in: ', (end - start) / 60, 'minutes')
cnf = torch.load("./results/cnf/models/cnf_boomerang.pt")
cnf.eval()

test =None
for i,data in enumerate(test_loader):
    if test is not None:
        test=torch.cat((test,data),dim=0)
    else:
        test=data.clone()


viz_samples = test.size()[0]
## samples from the flow
z_t0 = p_z0.sample([viz_samples]).to(device)

logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)
viz_timesteps = 20

### forward
ts = torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device)
z_t_samples, _  = cnf(ts, z_t0, logp_diff_t0)

z_t = z_t_samples[-1]

ts_inverse = torch.tensor(np.linspace(t1, t0, viz_timesteps)).type(torch.float32)# for training, we flow the samples backward (in time)

z_inverse, _ =  cnf(ts_inverse, z_t0, logp_diff_t0)

plot_samples(test, "cnf", dataset_name, z_t0, z_t, z_inverse[-1], range_min=[-8,8], range_max=[-8,8])
