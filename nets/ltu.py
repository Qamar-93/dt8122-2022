
import torch
from torch import nn
from typing import List

class Matmul(nn.Module):
    def forward(self, *args):
        return torch.matmul(*args)

class Matadd(nn.Module):
    def forward(self, *args):
        return torch.add(*args)
    
class LTU(nn.Module):
    """
    A Linear Transition Unit with a tanh nonlinearity
    """
    def __init__(self, ):
        super().__init__()
                
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))
        
        if (torch.mm(self.u, self.w.T)< -1).any():   
            self.get_u_hat()

        self.layer1 = Matmul()
        self.layer2 = Matadd() 
        self.layer3 = nn.Tanh()
        self.layer4 = Matmul()
        
    def get_u_hat(self):
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition 
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2)
   
    def forward(self, x):
        
        z = self.layer1(x, self.w.T) #self.w     # 2*1
        z = self.layer2(z, self.b)               # 2*1
        z = self.layer3(z)                       # 2*1
        z = self.layer4(z, self.u) #u.T          # 2*2
        z = z + x                                # 2*2
   
        return z