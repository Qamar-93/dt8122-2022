import torch
import torch.nn as nn

from . import Flow
class RealNVP(Flow):
    def __init__(self, bijections, net):
        super().__init__(bijections, net)
        
        
    def flow_outputs(self, x):
        
        log_prob = torch.zeros(x.shape[0])
        for bijection in self.bijections:
            x, ldj = bijection(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x).sum(1)
        
        return x, log_prob
    
    
class ReverseBijection(nn.Module):

    def forward(self, x):
        return x.flip(-1), x.new_zeros(x.shape[0])

    def inverse(self, z):
        return z.flip(-1)


class CouplingBijection(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        id, x2 = torch.chunk(x, 2, dim=-1)
        p = self.net(id)
        log_s, b = torch.chunk(p, 2, dim=-1)
        z2 = x2 * log_s.exp() + b
        z = torch.cat([id, z2], dim=-1)
        ldj = log_s.sum(-1)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            id, z2 = torch.chunk(z, 2, dim=-1)
            p = self.net(id)
            log_s, b = torch.chunk(p, 2, dim=-1)
            x2 = (z2 - b) * (-log_s).exp()
            x = torch.cat([id, x2], dim=-1)
        return x
