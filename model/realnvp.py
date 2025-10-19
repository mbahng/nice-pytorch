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

class ResidualBlock(nn.Module):
  """
  Residual block with layer normalization
  """
  def __init__(self, hidden_dim):
    super().__init__()
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.linear1 = nn.Linear(hidden_dim, hidden_dim)
    self.activation = nn.LeakyReLU(0.2)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x):
    residual = x
    out = self.norm1(x)
    out = self.linear1(out)
    out = self.activation(out)
    out = self.norm2(out)
    out = self.linear2(out)
    out = out + residual  # Skip connection
    return self.activation(out)

class MLP(nn.Module):
  """
  ResNet-style MLP with residual connections and layer normalization
  """

  def __init__(self, img_dim, hidden_dim, n_hidden_layers: int):
    super().__init__()
    assert n_hidden_layers >= 2

    # Input projection
    self.input_layer = nn.Sequential(
      nn.Linear(img_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.LeakyReLU(0.2)
    )

    # Residual blocks
    self.res_blocks = nn.ModuleList([
      ResidualBlock(hidden_dim) for _ in range(n_hidden_layers - 2)
    ])

    # Output projection
    self.output_layer = nn.Sequential(
      nn.LayerNorm(hidden_dim),
      nn.Linear(hidden_dim, img_dim)
    )

  def forward(self, x):
    x = self.input_layer(x)
    for block in self.res_blocks:
      x = block(x)
    return self.output_layer(x)

class AffineCouplingLayer(nn.Module): 
  """
  Affine Coupling Layer that contains neural networks s, t 
  that act as a scaling and translation factor. 
  """

  def __init__(self, dim:int, neural_net_layers, hidden_dim, mask_first): 
    super().__init__()
    assert dim % 2 == 0

    self.mask = torch.zeros(dim) 
    if mask_first: 
      self.mask[::2] += 1 
    else: 
      self.mask[1::2] += 1 

    self.s = MLP(img_dim=dim, hidden_dim=hidden_dim, n_hidden_layers=neural_net_layers)
    self.t = MLP(img_dim=dim, hidden_dim=hidden_dim, n_hidden_layers=neural_net_layers)

  def forward(self, x, logdet_accum): 
    """
    Return output f(x), and be careful with the masks 
    Determinant calclation in section 3.3 of paper
    """
    x1, x2 = self.mask * x, (1 - self.mask) * x 
    y1 = x1 
    logscale = self.s(x1) 
    translation = self.t(x1) 
    y2 = (x2 * torch.exp(logscale) + translation) * (1 - self.mask) 
    # make sure to sum over only the samples! not the batch dimension! Also mask it 
    return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=-1)

  def inverse(self, y): 
    """
    The inverse is easy to calculate
    """
    y1, y2 = self.mask * y, (1 - self.mask) * y 
    x1 = y1 
    # just masking at the end should suffice? 
    x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
    return x1 + x2


class RealNVP(nn.Module): 
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE. 
  """

  def __init__(self, input_dim: int, n_coupling_layers: int, neural_net_layers: int, hidden_dim: int): 
    super().__init__()
    self.input_dim = input_dim
    coupling_layers = []
    for i in range(n_coupling_layers): 
      mask_first = bool(i % 2)
      coupling_layers.append(AffineCouplingLayer(input_dim, neural_net_layers, hidden_dim, mask_first))
    self.coupling_layers = nn.ModuleList(coupling_layers)
    self.latent_prior = LogisticDistribution()

  def forward(self, x): 
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=1) + logdet_accum
    return x, log_likelihood

  def inverse(self, z: torch.Tensor): 
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, self.input_dim]).view(n_samples, self.input_dim)
    return self.inverse(z)

