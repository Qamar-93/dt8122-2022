import torch.nn as nn


def make_net(layers_no=3, hidden_units=32):
  modules = [nn.Linear(1,hidden_units), nn.GELU()]
  middle_layers = layers_no - 2
  for i in range(middle_layers):
      modules.append(nn.Linear(hidden_units,hidden_units))
      modules.append(nn.GELU())
  modules.append(nn.Linear(hidden_units, 2))    
  return nn.Sequential(*modules)
