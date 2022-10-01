from abc import abstractmethod
import torch
import torch.nn as nn
from torch.distributions import Normal

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Flow(nn.Module):

    def __init__(self, bijections, net, dim=1):
        super().__init__()
        self.bijections = nn.ModuleList(bijections)
        self.net = net
        self.dim = dim

    @property
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2, device=device),
            scale=torch.ones(2, device=device),
        )
        
    @abstractmethod
    def flow_outputs(self, x):
        raise NotImplementedError(
            "you called the train function on the abstract model class."
        )

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], device=device)
        for bijection in self.bijections:
            x, ldj = bijection(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x).sum(1)
        return log_prob
    
    # def prob(self, x):
    #     log_prob = torch.zeros(x.shape[0], device=device)
    #     for bijection in self.bijections:
    #         x, ldj = bijection(x)
        
    #     return x

    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for bijection in reversed(self.bijections):
            z = bijection.inverse(z)
        return z
