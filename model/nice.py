import torch 
import torch.nn as nn
from .distribution import LogisticDistribution 
from torch import Tensor, Size

class CouplingLayer(nn.Module): 
  """
  A simple coupling layer.
  """

  class MLP(nn.Module): 

    def __init__(self, dim: Size, hdim: Size, depth: int): 
      super().__init__()

      assert len(dim) == 1 and len(hdim) == 1
      assert depth >= 2
      self.idim = dim 
      self.hdim = hdim

      layers = [nn.Linear(self.idim[0], hdim[0]), nn.LeakyReLU(0.2)]

      for _ in range(depth - 2): 
        layers.append(nn.Linear(self.hdim[0], self.hdim[0])) 
        layers.append(nn.LeakyReLU(0.2))

      layers.append(nn.Linear(self.hdim[0], self.idim[0]))  

      self.layers = nn.Sequential(*layers) 

    def forward(self, x): 
      return self.layers(x)

  def __init__(self, dim: Size, neural_net_layers: int, hdim: Size, mask_first: bool): 
    super().__init__()
    self.idim = dim  
    self.odim = dim
    self.register_buffer("mask", self._init_mask(mask_first))
    
    self.mlp = self.MLP(dim, hdim, depth=neural_net_layers)

  def _init_mask(self, mask_first: bool): 
    mask = torch.zeros(self.idim) 
    if mask_first: 
      mask[::2] += 1 
    else: 
      mask[1::2] += 1 
    return mask

  def forward(self, x: Tensor, logdet_accum: Tensor): 
    """
    Return output f(x) and log det, which is log(det(I)) = log(1) = 0
    """
    x1, x2 = self.mask * x, (1 - self.mask) * x 
    y1 = x1
    y2 = x2 + (self.mlp(x1) * (1. - self.mask))
    return y1 + y2, logdet_accum

  def inverse(self, z: Tensor): 
    """
    The inverse is easy to calculate
    """
    z1, z2 = self.mask * z, (1 - self.mask) * z 
    x1 = z1 
    x2 = z2 - (self.mlp(z1) * (1 - self.mask))
    return x1 + x2

class ScalingLayer(nn.Module): 
  """
  Use this to make NICE non volume preserving. 
  """

  def __init__(self, idim: Size):
    super().__init__()
    # Initialize with small values to prevent numerical instability
    self.log_scale_vector = nn.Parameter(torch.randn(*idim, requires_grad=True))

  def forward(self, x: Tensor, logdet: int):
    log_det_jacobian = torch.sum(self.log_scale_vector)
    return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian 

  def inverse(self, y: Tensor): 
    return torch.exp(- self.log_scale_vector) * y

class NICE(nn.Module): 
  """
  Simplest finite normalizing flow model by Dinh 2014. 
  """

  def __init__(self, idim: Size, n_coupling_layers: int, neural_net_layers: int, hdim: Size, device): 
    super().__init__()
    self.idim = idim

    coupling_layers = []
    for i in range(n_coupling_layers): 
      mask_first = bool(i % 2)
      coupling_layers.append(CouplingLayer(idim, neural_net_layers, hdim, mask_first))
    self.coupling_layers = nn.ModuleList(coupling_layers)

    self.scaling_layer = ScalingLayer(idim)
    self.latent_prior = LogisticDistribution()
    self.device = device 
    self.to(device)

  def forward(self, x: Tensor): 
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    # x, logdet_accum = self.scaling_layer(x, logdet_accum) 
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=1) + logdet_accum
    return x, log_likelihood

  def inverse(self, z: Tensor): 
    z = self.scaling_layer.inverse(z)
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, *self.idim]).to(self.device)
    return self.inverse(z)

