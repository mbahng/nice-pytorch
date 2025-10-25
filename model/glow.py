import torch 
import torch.nn as nn
from .distribution import LogisticDistribution 
from torch import Tensor, Size  

class ActNorm(nn.Module):
  """
  Scales + Shifts the flow by (learned) constants per dimension.
  In NICE paper there is a Scaling layer which is a special case of this where t is None
  Really an AffineConstantFlow but with a data-dependent initialization,
  where on the very first batch we clever initialize the s,t so that the output
  is unit gaussian. As described in Glow paper.
  """
  def __init__(self, idim: Size): 
    super().__init__()
    # need this so that we can initialize the weights using first batch of training data. 
    self.data_dep_init_done = False
    self.s = nn.Parameter(torch.randn(idim, requires_grad=True))
    self.b = nn.Parameter(torch.randn(idim, requires_grad=True))

  def forward(self, x):
    # first batch is used for init
    if not self.data_dep_init_done:
      self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
      self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
      self.data_dep_init_done = True
    z = x * torch.exp(self.s) + self.t
    log_det = torch.sum(self.s, dim=1)
    return z, log_det

  def inverse(self, z):
    x = (z - self.b) * torch.exp(-self.s)
    # log_det = torch.sum(-self.s, dim=1)
    return x

class Invertible1x1Conv(nn.Module): 
  """
  Invertible 1x1 Convolution in Section 3.2
  Calculating the log determinant of this linear map W is O(c^3), but can be reduced to 
  O(c) by directly parameterizing the $W$ in its LU-decomposition. 

    W = PL( U + diag(s))

  where P is a permutation matrix, L lower triangular, U upper triangular with 0s on 
  diagonal, and s is a vector. Then, the log determinant is 
  
    log |det(W)| = \sum log()|s|)
  """

  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
    P, L, U = torch.lu_unpack(*Q.lu())
    self.P = P # remains fixed during optimization
    self.L = nn.Parameter(L) # lower triangular portion
    self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
    self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

  def _assemble_W(self):
    """ assemble W from its pieces (P, L, U, S) """
    L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
    U = torch.triu(self.U, diagonal=1)
    W = self.P @ L @ (U + torch.diag(self.S))
    return W

  def forward(self, x):
    W = self._assemble_W()
    z = x @ W
    log_det = torch.sum(torch.log(torch.abs(self.S)))
    return z, log_det

  def inverse(self, z):
    W = self._assemble_W()
    W_inv = torch.inverse(W) # this is the expensive operation! 
    x = z @ W_inv
    # log_det = -torch.sum(torch.log(torch.abs(self.S)))
    return x

class Squeeze(nn.Module): 
  """
  Squeezing operation used in each level to convert shape (C, H, W) to 
  shape (4C, H/2, W/2). See 2016 Dinh Section 3.6 or Figure 3. 
  """

  def __init__(self, idim: Size): 
    self.idim = idim
    assert self.idim[-1] % 2 == 0 and self.idim[-2] % 2 == 0 
    self.odim = idim  
    self.odim[-1] = self.odim[-1] // 2 
    self.odim[-2] = self.odim[-2] // 2 
    self.odim[-3] = 4 * self.odim[-3] 

  def forward(self, x: Tensor, logdet_accum): 
    if len(x.size()) == 3: 
      C, W, H = x.size()
      return x.reshape(4 * C, W // 2, H // 2), logdet_accum
    elif len(x.size()) == 4: 
      B, C, W, H = x.size()
      return x.reshape(B, 4 * C, W // 2, H // 2), logdet_accum

  def inverse(self, z: Tensor): 
    if len(z.size()) == 3: 
      C, W, H = z.size() 
      return z.reshape(C // 4, W * 2, H * 2) 
    elif len(z.size()) == 4: 
      B, C, W, H = z.size() 
      return z.reshape(B, C // 4, W * 2, H * 2) 

class AffineCouplingLayer2d(nn.Module): 
  """
  Affine Coupling Layer that contains neural networks s, t that act as a scaling and translation factor. 
  Designed for 2d inputs, using fully convolutional layers. 
  For 2d inputs, use AffineCouplingLayer2d. 
  """

  class CNN(nn.Module): 

    def __init__(self, idim: Size, depth: int): 
      super().__init__()
      self.idim = idim 
      C, H, W = idim
      layers = []
      for i in range(1, depth): 
        layers.append(nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)) 
        if i % 3 == 0: 
          ...
      self.layers = nn.Sequential(*layers)
    
    def forward(self, x: Tensor): 
      return self.layers(x) 
  
  def __init__(self, idim: Size, neural_net_layers: int, _: Size, mask_first: bool): 
    super().__init__()
    assert len(idim) >= 2 
    self.idim = idim 
    self.register_buffer("mask", self._init_mask(mask_first)) 
    

    self.s = self.CNN(idim, depth=neural_net_layers) 
    self.t = self.CNN(idim, depth=neural_net_layers) 

  def _init_mask(self, mask_first): 
    """
    Checkerboard mask 
    """
    mask = torch.arange(self.idim.numel()) % 2 == mask_first 
    return mask.reshape(self.idim).float()

  def forward(self, x, logdet_accum):
    """
    Return output f(x), and be careful with the masks
    Determinant calclation in section 3.3 of paper
    """
    x1, x2 = self.mask * x, (1 - self.mask) * x
    y1 = x1
    logscale = self.s(x1)
    # Clamp log-scale to prevent numerical overflow/underflow
    logscale = torch.clamp(logscale, min=-10, max=10)
    translation = self.t(x1)
    y2 = (x2 * torch.exp(logscale) + translation) * (1 - self.mask)
    # make sure to sum over only the samples! not the batch dimension! Also mask it
    nonbatch_dims = list(range(1, len(logscale.size())))
    return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=nonbatch_dims)

  def inverse(self, y):
    """
    The inverse is easy to calculate
    """
    y1, y2 = self.mask * y, (1 - self.mask) * y 
    x1 = y1 
    # just masking at the end should suffice? 
    x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
    return x1 + x2

class Glow(nn.Module): 

  def __init__(self, idim: Size, n_levels: int, flow_depth: int, neural_net_layers: int, hdim: Size, device): 
    self.idim = idim

    coupling_layers = [] 
    for l in range(n_levels - 1): # n_levels is L in the paper
      coupling_layers.append(Squeeze())
      for k in range(flow_depth): # referred to as K in the paper
        # One step of flow
        coupling_layers.append(ActNorm(idim))
        coupling_layers.append(Invertible1x1Conv(idim))
        mask_first = bool(k % 2)
        coupling_layers.append(AffineCouplingLayer2d(idim, neural_net_layers, hdim, mask_first))

      coupling_layers.append(Split())

    #  add final squeeze and step of flow. 
    coupling_layers.append(Squeeze())
    for k in range(flow_depth): # referred to as K in the paper
      # One step of flow
      coupling_layers.append(ActNorm(idim))
      coupling_layers.append(Invertible1x1Conv(idim))
      mask_first = bool(k % 2)
      coupling_layers.append(AffineCouplingLayer2d(idim, neural_net_layers, hdim, mask_first))

    self.coupling_layers = nn.ModuleList(coupling_layers)

    self.latent_prior = LogisticDistribution()
    self.device = device 
    self.to(device)

  def forward(self, x: Tensor): 
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    nonbatch_dims = list(range(1, len(x.size()))) 
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=nonbatch_dims) + logdet_accum
    return x, log_likelihood

  def inverse(self, z: Tensor): 
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, *self.idim]).to(self.device)
    return self.inverse(z)


