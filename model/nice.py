import torch 
import torch.nn as nn
from torch.distributions import Distribution, Uniform
import torch.nn.functional as F

class LogisticDistribution(Distribution):

  def __init__(self):
    super().__init__()

  def log_prob(self, x):
    return -(F.softplus(x) + F.softplus(-x))

  def sample(self, size):
    z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)
    return torch.log(z) - torch.log(1. - z)

class MLP(nn.Module): 

  def __init__(self, img_dim, hidden_dim, n_hidden_layers: int): 
    super().__init__()
    assert n_hidden_layers >= 2

    layers = [nn.Linear(img_dim, hidden_dim), nn.LeakyReLU(0.2)]
    for _ in range(n_hidden_layers - 2): 
      layers.append(nn.Linear(hidden_dim, hidden_dim)) 
      layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Linear(hidden_dim, img_dim))  
    self.layers = nn.Sequential(*layers) 

  def forward(self, x): 
    return self.layers(x)

class CouplingLayer(nn.Module): 

  def __init__(self, dim:int, neural_net_layers, hidden_dim, mask_first): 
    super().__init__()
    assert dim % 2 == 0

    self.mask = torch.zeros(dim) 
    if mask_first: 
      self.mask[::2] += 1 
    else: 
      self.mask[1::2] += 1 

    self.mlp = MLP(img_dim=dim, hidden_dim=hidden_dim, n_hidden_layers=neural_net_layers)

  def forward(self, x, logdet_accum): 
    """
    Return output f(x) and log det, which is log(det(I)) = log(1) = 0
    """
    x1, x2 = self.mask * x, (1 - self.mask) * x 
    y1, y2 = x1, x2 + (self.mlp(x1) * (1. - self.mask))
    return y1 + y2, logdet_accum

  def inverse(self, y): 
    """
    The inverse is easy to calculate
    """
    y1, y2 = self.mask * y, (1 - self.mask) * y 
    x1, x2 = y1, y2 - self.mlp(y1) * (1 - self.mask)
    return x1 + x2

class ScalingLayer(nn.Module): 
  """
  Not sure if this was part of original implementation, but without this 
  NICE is a perfect volume preserving flow. 
  """

  def __init__(self, dim: int): 
    super().__init__()
    self.log_scale_vector = nn.Parameter(torch.randn(1, dim, requires_grad=True))

  def forward(self, x, logdet):
    log_det_jacobian = torch.sum(self.log_scale_vector)
    return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian 

  def inverse(self, y): 
    return torch.exp(- self.log_scale_vector) * y

class NICE(nn.Module): 
  """
  Simplest volume-preserving, finite normalizing flow model by Dinh 2014. 
  """

  def __init__(self, input_dim: int, n_coupling_layers: int, neural_net_layers: int, hidden_dim: int): 
    super().__init__()
    self.input_dim = input_dim
    coupling_layers = []
    for i in range(n_coupling_layers): 
      mask_first = bool(i % 2)
      coupling_layers.append(CouplingLayer(input_dim, neural_net_layers, hidden_dim, mask_first))
    self.coupling_layers = nn.ModuleList(coupling_layers)
    self.scaling_layer = ScalingLayer(dim=input_dim)
    self.latent_prior = LogisticDistribution()


  def forward(self, x): 
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    x, logdet_accum = self.scaling_layer(x, logdet_accum)
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=1) + logdet_accum
    return x, log_likelihood

  def inverse(self, z: torch.Tensor): 
    z = self.scaling_layer.inverse(z)
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, self.input_dim]).view(n_samples, self.input_dim)
    return self.inverse(z)

