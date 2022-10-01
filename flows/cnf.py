from . import Flow
import torch
import torch.nn as nn 
from nets import HyperNetwork

from torchdiffeq  import odeint

# class CNF(Flow):
#     def __init__(self, bijections, net):
#         super().__init__(bijections,net)
        
#     def flow_outputs(self, x):
#         log_det = torch.zeros(torch.Size([x[0].shape[0],x[1].shape[0],1]), device=self.device)    
#         z = x
#         for bijection in self.flow:
#             z, ldj = bijection(z)
#             log_det += ldj
            
#         return z, log_det
    
#     def sample(self, num_samples):
#         ts = torch.tensor([0, 1]).type(torch.float32).to(self.device)
#         z0 = self.base_dist.sample((num_samples,))
#         logp_diff_t0 = torch.zeros(z0.size()[0], 1).type(torch.float32).to(self.device)
#         z = (ts, z0, logp_diff_t0)
#         for bijection in reversed(self.flow):
#             z = bijection.inverse(z)
    
#         return z


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.f = HyperNetwork(in_out_dim, hidden_dim, width)

    def ode_rhs(self, t, states):
        ''' Differential function implementation. states is (x1,logp_diff_t1) where
                x1 - [N,d] initial values for ODE states
                logp_diff_t1 - [N,1] initial values for density changes
        '''
        z,logp_z = states # [N,d], [N,1]
        N = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt      = self.f(t,z) # [N,d] 
            dlogp_z_dt = -self.trace_df_dz(dz_dt, z).view(N, 1)
        return (dz_dt, dlogp_z_dt)
    
    def forward(self, ts, z0, logp_diff_t0, method='dopri5'):
        ''' Forward integrates the CNF system. Returns state and density change solutions.
            Input
                ts - [T]   time points
                z0 - [N,d] initial values for ODE states
                logp_diff_t0 - [N,1] initial values for density changes
            Retuns:
                zt -     [T,N,...]  state trajectory computed at t
                logp_t - [T,N,1]    density change computed over time
        '''
        zt, logp_t = odeint(self.ode_rhs, (z0, logp_diff_t0), ts, method=method)
        return zt, logp_t
    
    def trace_df_dz(self, f, z):
        """Calculates the trace of the Jacobian df/dz.
        Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        Input:
            f - function output [N,d]
            z - current state [N,d]
        Returns:
            tr(df/dz) - [N]
        """
        sum_diag = 0.
        for i in range(z.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()
    
