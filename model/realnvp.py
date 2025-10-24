import torch 
import torch.nn as nn
from .distribution import LogisticDistribution 
from torch import Tensor, Size

class AffineCouplingLayer1d(nn.Module): 
  """
  Affine Coupling Layer that contains neural networks s, t that act as a scaling and translation factor. 
  Designed only for 1d inputs, e.g. flattened images. 
  For 2d inputs, use AffineCouplingLayer2d. 
  """

  class MLP(nn.Module):
    """
    ResNet-style MLP with residual connections and layer normalization. 
    Only supports 1d inputs. 
    """

    class ResidualBlock(nn.Module):

      def __init__(self, hdim: Size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hdim[0])
        self.linear1 = nn.Linear(hdim[0], hdim[0])
        self.activation = nn.LeakyReLU(0.2)
        self.norm2 = nn.LayerNorm(hdim[0])
        self.linear2 = nn.Linear(hdim[0], hdim[0])

      def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = out + residual  # Skip connection
        return self.activation(out)

    def __init__(self, idim: Size, hdim: Size, depth: int):
      super().__init__()
      assert depth >= 2

      # Input projection
      self.input_layer = nn.Sequential(
        nn.Linear(idim[0], hdim[0]),
        nn.LayerNorm(hdim[0]),
        nn.LeakyReLU(0.2)
      )

      # Residual blocks
      self.res_blocks = nn.ModuleList([
        self.ResidualBlock(hdim) for _ in range(depth - 2)
      ])

      # Output projection
      self.output_layer = nn.Sequential(
        nn.LayerNorm(hdim[0]),
        nn.Linear(hdim[0], idim[0])
      )

    def forward(self, x):
      x = self.input_layer(x)
      for block in self.res_blocks:
        x = block(x)
      return self.output_layer(x)

  def __init__(self, idim: Size, neural_net_layers: int, hdim: Size, mask_first: bool): 
    super().__init__()

    self.idim = idim
    # register this as buffer so it's automatically moved to devices
    self.register_buffer("mask", self._init_mask(mask_first))

    self.s = self.MLP(idim, hdim, depth=neural_net_layers)
    self.t = self.MLP(idim, hdim, depth=neural_net_layers)

  def _init_mask(self, mask_first: bool): 
    mask = torch.zeros(self.idim) 
    if mask_first: 
      mask[::2] += 1 
    else: 
      mask[1::2] += 1 
    return mask

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
    return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=-1)

  def inverse(self, y): 
    """
    The inverse is easy to calculate
    """
    y1, y2 = self.mask * y, (1 - self.mask) * y 
    x1 = y1 
    x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
    return x1 + x2

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

class RealNVP(nn.Module): 
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE by implementing resdiual connections, 
  affine coupling layer, and convolutions to process 2d images directly. 
  """

  def __init__(self, idim: Size, n_coupling_layers: int, neural_net_layers: int, hdim: Size, device): 
    super().__init__()
    self.idim = idim
    
    AffineCouplingLayer = AffineCouplingLayer1d if len(self.idim) == 1 else AffineCouplingLayer2d 
    coupling_layers = []
    for i in range(n_coupling_layers): 
      mask_first = bool(i % 2)
      coupling_layers.append(AffineCouplingLayer(idim, neural_net_layers, hdim, mask_first))
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

