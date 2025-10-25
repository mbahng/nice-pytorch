import torch 
import torch.nn as nn
import torch.nn.functional as F
from .distribution import LogisticDistribution, Normal
import torch.nn.utils.weight_norm as weight_norm
from torch import Tensor, Size
from warnings import deprecated 
from enum import Enum
import numpy as np

@deprecated("""This was used so that we can compare RealNVP with NICE on 1D distributions.
            However, the squeeze operations can't be implemented here.""")
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

@deprecated("Used for comparison with NICE")
class RealNVP1d(nn.Module): 
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE by implementing resdiual connections, 
  affine coupling layer, and convolutions to process 2d images directly. 
  """

  def __init__(self, idim: Size, n_coupling_layers: int, neural_net_layers: int, hdim: Size, device): 
    super().__init__()
    self.idim = idim
    
    coupling_layers = []
    for i in range(n_coupling_layers): 
      mask_first = bool(i % 2)
      coupling_layers.append(AffineCouplingLayer1d(idim, neural_net_layers, hdim, mask_first))
    self.coupling_layers = nn.ModuleList(coupling_layers)
    from torch.distributions import Normal 

    self.latent_prior = Normal()
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


class Mask(Enum): 
  Checkerboard0 = 1 
  Checkerboard1 = 2 
  Channel0 = 3 
  Channel1 = 4

class LogitTransform(nn.Module):
  """
  Logit transform with dequantization as described in RealNVP paper Section 4.1
  Maps [0, 1] to (-inf, inf) for better modeling with Gaussian prior
  """
  def __init__(self, alpha=0.05):
    super().__init__()
    self.alpha = alpha  # Small constant to avoid log(0)
  
  def forward(self, x, logdet_accum, z):
    """x in [0, 1] -> y in (-inf, inf)"""

    # restrict data
    x *= 2.                   # [0, 2]
    x -= 1.                   # [-1, 1]
    x *= (1 - 2 * self.alpha) # [-0.9, 0.9]
    x += 1.                   # [0.1, 1.9]
    x /= 2.                   # [0.05, 0.95]

    
    # Apply logit with numerical stability
    # Map [0, 1] -> [alpha, 1-alpha] to avoid log(0)
    x_scaled = self.alpha + (1 - 2 * self.alpha) * x
    y = torch.log(x_scaled) - torch.log(1 - x_scaled)
    

    # logit data
    logit_x = torch.log(x) - torch.log(1. - x)

    # log-determinant of Jacobian from the transform
    pre_logit_scale = torch.tensor(
        np.log(1 - 2 * self.alpha) - np.log(2 * self.alpha))
    log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
        - F.softplus(-pre_logit_scale)

    logdet = torch.sum(log_diag_J, dim=(1, 2, 3))
    
    return y, logdet_accum + logdet, z
  
  def inverse(self, y, z):
    """y in (-inf, inf) -> x in [0, 1]"""
    # Apply sigmoid to map back to [alpha, 1-alpha]
    x_scaled = torch.sigmoid(y)
    
    # Invert the linear scaling: x = (x_scaled - alpha) / (1 - 2*alpha)
    x = (x_scaled - self.alpha) / (1 - 2 * self.alpha)
    
    # Clamp to ensure bounds (handles numerical errors)
    x = torch.clamp(x, 0, 1)
    return x, z

class AffineCouplingLayer2d(nn.Module): 
  """
  Affine Coupling Layer that contains neural networks s, t that act as a scaling and translation factor. 
  Designed for 2d inputs, using fully convolutional layers. 
  For 2d inputs, use AffineCouplingLayer2d. 
  """

  class ResidualCNN(nn.Module): 
    """
    CNN with residual connections and skip architecture similar to the reference implementation.
    Uses pre-activation residual blocks (BN -> ReLU -> Conv).
    """
    
    class ResidualBlock(nn.Module):
        """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv"""
        def __init__(self, channels: int):
            super().__init__()
            # Pre-activation: BN and ReLU come BEFORE convolutions
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu1 = nn.ReLU()
            self.conv1 = weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu2 = nn.ReLU()
            self.conv2 = weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        
        def forward(self, x: Tensor) -> Tensor:
            # Pre-activation pattern
            identity = x
            
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            
            # Add residual connection
            return out + identity
    
    def __init__(self, idim: Size, depth: int, hdim: Size, use_skip: bool = True):
      super().__init__()
      self.idim = idim
      self.hdim = hdim
      self.use_skip = use_skip
      C, _, _ = idim  # C is most likely 3 
      
      # Input projection
      self.bn1 = nn.BatchNorm2d(C)
      self.conv_input = weight_norm(nn.Conv2d(C, hdim[0], kernel_size=3, stride=1, padding=1))
      self.bn2 = nn.BatchNorm2d(hdim[0])
      self.relu = nn.ReLU()
      
      # Create residual blocks
      self.blocks = nn.ModuleList([
          self.ResidualBlock(hdim[0]) for _ in range(depth)
      ])
      
      # Skip architecture: 1x1 convolutions to accumulate features from all blocks
      if use_skip:
          # Initial skip connection
          self.skip_input = weight_norm(nn.Conv2d(hdim[0], hdim[0], kernel_size=1))
          
          # Skip connection for each residual block
          self.skip_connections = nn.ModuleList([
              weight_norm(nn.Conv2d(hdim[0], hdim[0], kernel_size=1)) 
              for _ in range(depth)
          ])
      
      # Output projection
      self.bn_out = nn.BatchNorm2d(hdim[0])
      self.relu_out = nn.ReLU()
      self.conv_output = weight_norm(nn.Conv2d(hdim[0], C, kernel_size=1))
      
      # For numerical stability, apply tanh and then scale  
      self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: Tensor): 
      # Input processing
      x = self.bn1(x) 
      x = self.conv_input(x) 
      x = self.bn2(x) 
      x = self.relu(x)
      
      if self.use_skip:
          # Initialize skip with processed input
          skip = self.skip_input(x)
          
          # Process through residual blocks with skip accumulation
          for i, block in enumerate(self.blocks):
              x = block(x)
              # Accumulate skip connections from all blocks
              skip = skip + self.skip_connections[i](x)
          
          # Use accumulated skip as the main path
          x = skip
      else:
          # Simple sequential residual connections (your original approach)
          for block in self.blocks:
              x = block(x)
      
      # Output processing with pre-activation
      x = self.bn_out(x)
      x = self.relu_out(x)
      x = self.conv_output(x)
      
      # Numerical stability
      x = torch.tanh(x) * self.scale
      return x

  def __init__(self, idim: Size, neural_net_layers: int, hdim: Size, mask: Mask):
    super().__init__()
    assert len(idim) >= 2
    self.idim = idim
    self.register_buffer("mask", self._init_mask(mask))

    self.s = self.ResidualCNN(idim, depth=neural_net_layers, hdim=hdim)
    self.t = self.ResidualCNN(idim, depth=neural_net_layers, hdim=hdim) 

  def _init_mask(self, mask: Mask): 
    """
    Alternating checkerboard or channel wise masking
    """
    match mask: 
      case Mask.Checkerboard0: 
        C, H, W = self.idim
        checker = torch.zeros((1, H, W))
        checker[:, ::2, ::2] = 1  # Even rows, even cols
        checker[:, 1::2, 1::2] = 1  # Odd rows, odd cols
        return checker.repeat(C, 1, 1)
      case Mask.Checkerboard1: 
        C, H, W = self.idim
        checker = torch.ones((1, H, W))
        checker[:, ::2, ::2] = 0  # Even rows, even cols
        checker[:, 1::2, 1::2] = 0  # Odd rows, odd cols
        return checker.repeat(C, 1, 1)
      case Mask.Channel0: 
        mask = torch.zeros(self.idim) 
        mask[:self.idim[0] // 2] += 1 
        return mask
      case Mask.Channel1: 
        mask = torch.zeros(self.idim) 
        mask[self.idim[0] // 2:] += 1 
        return mask

  def forward(self, x, logdet_accum, z):
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
    nonbatch_dims = list(range(1, len(logscale.size())))
    return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=nonbatch_dims), z

  def inverse(self, y, z): 
    """
    The inverse is easy to calculate
    """
    y1, y2 = self.mask * y, (1 - self.mask) * y 
    x1 = y1 
    # just masking at the end should suffice? 
    x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
    return x1 + x2, z 


class Squeeze(nn.Module): 
  """
  Squeezing operation used in each level to convert shape (C, H, W) to 
  shape (4C, H/2, W/2). See 2016 Dinh Section 3.6 or Figure 3. 
  Note that you can't just simply reshape!
  """

  def __init__(self, idim: Size): 
    super().__init__()
    self.idim = idim
    assert self.idim[-1] % 2 == 0 and self.idim[-2] % 2 == 0 
    self.odim = list(idim)
    self.odim[-1] = self.odim[-1] // 2 
    self.odim[-2] = self.odim[-2] // 2 
    self.odim[-3] = 4 * self.odim[-3] 
    self.odim = torch.Size(self.odim) 

  def _squeeze(self, x: Tensor): 
    B, C, H, W = x.size()
    x = x.reshape(B, C, H//2, 2, W//2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*4, H//2, W//2)
    return x 

  def _unsqueeze(self, z: Tensor): 
    B, C, H, W = z.size() 
    z = z.reshape(B, C//4, 2, 2, H, W)
    z = z.permute(0, 1, 4, 2, 5, 3)
    z = z.reshape(B, C//4, H*2, W*2)
    return z

  def forward(self, x: Tensor, logdet_accum: Tensor, z: Tensor):  
    x = self._squeeze(x) 
    z = self._squeeze(z) if z is not None else z
    return x, logdet_accum, z

  def inverse(self, x: Tensor, z: Tensor): 
    x = self._unsqueeze(x) 
    z = self._unsqueeze(z) if z is not None else z
    return x, z

class FactorOut(nn.Module): 
  
  def __init__(self, idim: Size, scale: int):  
    super().__init__()
    self.idim = idim 
    self.scale = scale # keeps track of how much has been factored out in powers of two
    self.odim = list(idim)
    self.odim[-1] = self.odim[-1]
    self.odim[-2] = self.odim[-2]
    self.odim[-3] = self.odim[-3] // 2
    self.odim = torch.Size(self.odim) 

  def forward(self, x: Tensor, logdet_accum: Tensor, z: Tensor): 
    B, C, H, W = x.size() 
    split = C // 2 
    new_z = x[:,:split,:,:] 
    x = x[:,split:,:,:] 
    if z is not None: 
      z = torch.concat([z, new_z], dim=1)
    else: 
      z = new_z 
    return x, logdet_accum, z

  def inverse(self, y: Tensor, z: Tensor):  # 3, 9 -> 6, 6 
    split = y.size(1) # channels 3

    # you want to take back from the right end of z and add it back in
    new_y = z[:,-split:,:,:] 
    z = z[:,:-split,:,:] 

    x = torch.concat([new_y, y], dim=1) 
    return x, z

class RealNVP(nn.Module):
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE by implementing resdiual connections,
  affine coupling layer, and convolutions to process 2d images directly.

  The 2nd paragraph in Section 3.6 explains how the layers should be made.
  """

  def __init__(self, idim: Size, n_scales: int, neural_net_layers: int, hdim: Size, device):
    super().__init__()
    self.idim = idim

    coupling_layers = []
    coupling_layers.append(LogitTransform())
    squeezed_idim = idim
    # Dinh 2015 refers to each of these steps as a "scale," hence the name
    for scale in range(n_scales - 1):
      # first apply 3 coupling layers with alternating checkboard masks
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard0))
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard1))
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard0))
    
      # apply squeeze operation.
      coupling_layers.append(s := Squeeze(squeezed_idim))
    
      squeezed_idim = s.odim
    
      # Then 3 more layers with alternating channel-wise masking
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Channel0))
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Channel1))
      coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Channel0))

      # Finally a factor out to split latent space 
      coupling_layers.append(f := FactorOut(squeezed_idim, scale)) 
      squeezed_idim = f.odim
    
    # final scale consists of 4 coupling layers with alternating checkerboard masks.
    coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard0))
    coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard1))
    coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard0))
    coupling_layers.append(AffineCouplingLayer2d(squeezed_idim, neural_net_layers, hdim, Mask.Checkerboard1))
    coupling_layers.append(FactorOut(squeezed_idim, n_scales - 1))


    self.coupling_layers = nn.ModuleList(coupling_layers)

    self.latent_prior = Normal(torch.tensor(0.), torch.tensor(1.))

    odim_x, _, odim_z = self.forward(torch.rand(1, *idim))
    self.odim_x = odim_x.squeeze(0).size() 
    self.odim_z = odim_z.squeeze(0).size()

    self.device = device
    self.to(device)

  def forward(self, x: Tensor):
    logdet_accum = torch.zeros(x.size(0), device=x.device) 
    z = None
    for i, coupling_layer in enumerate(self.coupling_layers):
      x, logdet_accum, z = coupling_layer.forward(x, logdet_accum, z)
    # log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=nonbatch_dims) + logdet_accum

    # Compute log probability
    nonbatch_dims = list(range(1, len(x.size())))
    K = np.prod(x.shape[1:])  # Total dimensions

    log_density_z = torch.sum(self.latent_prior.log_prob(torch.concat((x, z), dim=1)), dim=nonbatch_dims)
    log_density_x = log_density_z + logdet_accum
    
    # Add discretization correction for 8-bit images
    log_prob_x = log_density_x - K * np.log(256.0)
    
    return x, log_prob_x, z

  def inverse(self, y, z): 
    for coupling_layer in reversed(self.coupling_layers): 
      y, z = coupling_layer.inverse(y, z)  # type: ignore 
    assert z.size(1) == 0 # channels in latent dim should all be gone by now. 
    return y

  def sample(self, n_samples: int): 
    y = self.latent_prior.sample([n_samples, *self.odim_x]).to(self.device)
    z = self.latent_prior.sample([n_samples, *self.odim_z]).to(self.device)
    return self.inverse(y, z)

