import torch
import torch.nn as nn
import pandas as pd

import torch
# refer to https://stackoverflow.com/questions/54316053/update-step-in-pytorch-implementation-of-newtons-method
# inital x
initial_x = torch.tensor([4.], requires_grad = True) 

# function to want to solve
def solve_func(x): 
    return torch.exp(x) - 2

def newton_method(function, initial, iteration=10, convergence=0.0001):
    for i in range(iteration): 
        previous_data = initial.clone()
        value = function(initial)
        value.backward()
        # update 
        initial.data -= (value / initial.grad).data
        # zero out current gradient to hold new gradients in next iteration 
        initial.grad.data.zero_() 
        print("epoch {}, obtain {}".format(i, initial))
        # Check convergence. 
        # When difference current epoch result and previous one is less than 
        # convergence factor, return result.
        if torch.abs(initial - previous_data) < torch.tensor(convergence):
            print("break")
            return initial.data
    return initial.data # return our final after iteration



class Planar(nn.Module):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self,x):
        
        # g = f^-1
        z = self.net(x)
            
        for name, param in self.net.named_parameters():
            if name == 'u' : 
                self.u = param
            elif name == 'w' : 
                self.w = param
            elif name == 'b' : 
                self.b = param
        
        affine = torch.mm(x, self.w.T) + self.b          # 2*1
        psi = (1 - nn.Tanh()(affine) ** 2) * self.w      # 2*2
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()    # 1*2
        log_det = torch.log(1e-4 + abs_det).squeeze(0)   # 2
        
        return z, log_det
    

    
    
    
    def inverse(self, z):
        def newton_method(function, initial, iteration=100, convergence=[0.01, 0.01]):
            for i in range(iteration): 
                previous_data = initial.clone()
                value = function(initial)
                value.backward()
                # update 
                initial.data -= (value / initial.grad).data
                # zero out current gradient to hold new gradients in next iteration 
                initial.grad.data.zero_() 
                print("epoch {}, obtain {}".format(i, initial))
                # Check convergence. 
                # When difference current epoch result and previous one is less than 
                # convergence factor, return result.
                comp = torch.le(torch.abs(initial - previous_data).data, torch.tensor(convergence))
                
                if comp.all() == True:
                    print("break")
                    return initial.data
            return initial.data # return our final after iteration
        sol = None
        for idx, sample in enumerate(z):
            sample.requires_grad_()
            s = newton_method(self.net, sample)
            if sol is not None:
                sol=torch.cat((sol,s),dim=1)
            else:
                sol=s
            
        return sol
    
    

from torch.nn import BCEWithLogitsLoss
from torch.distributions import Normal

import math
import time

from Data import *
from nets.ltu import LTU
from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.df[item]



class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device  
        
    @property    
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2, device=self.device),
            scale=torch.ones(2, device=self.device),
        )
      
        
    def build(self): 
        
        return NotImplemented
        
    def flow_outputs(self, x):
        
        log_det = torch.zeros(x.shape[0], device=self.device)
        z = x
        for bijection in self.flow:
            z, ldj = bijection(z)
            log_det += ldj
            
        return z, log_det
   
    
    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for bijection in reversed(self.flow):
            print("qqq", bijection.inverse(z))
            z = bijection.inverse(z)
        return z
    


class PlanarFlow(Flow):   

    def __init__(self, net = LTU, dim=5, device='cuda'):
        Flow.__init__(self) 
        self.net = net
        self.dim = dim
        self.device = device
        self.bijections = []
        self.build()
        self.flow = nn.ModuleList(self.bijections)  

    def build(self): 
        for i in range(self.dim):
            self.bijections += [Planar(self.net)]
            
              
def split_data(df, batch_size, training_frac=0.8):
    train_df = df.sample(frac = training_frac)
    test_df = df.drop(train_df.index)

    train_df_tensor = torch.tensor(list(train_df.values))
    test_df_tensor = torch.tensor(list(test_df.values))

    train_loader= DataLoader(MyData(train_df_tensor), batch_size=batch_size, shuffle=True)

    test_loader= DataLoader(MyData(test_df_tensor), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def base_dist():
    return Normal(
        loc=torch.zeros(2, device=device),
        scale=torch.ones(2, device=device),
    )
    
    
batch_size = 32
df = pd.read_csv('./data/two_moons.csv')

train_loader, test_loader = split_data(df, 32, 0.7)

num_batches = train_loader.__len__()


net = LTU()



flow_planar = PlanarFlow(net = net, dim= 12, device = device).to(device)
print(flow_planar.flow)

test =None
for i,data in enumerate(test_loader):
    if test is not None:
        test=torch.cat((test,data),dim=0)
    else:
        test=data.clone()


z_samples = flow_planar.sample(test.size()[0])
