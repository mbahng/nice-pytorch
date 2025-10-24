from torch.distributions import Distribution, Uniform
import torch.nn.functional as F
import torch

class LogisticDistribution(Distribution):
  """
  Simple distribution that samples logistic distribution
  """
  arg_constraints = {} 

  def __init__(self):
    super().__init__()

  def log_prob(self, x):
    return -(F.softplus(x) + F.softplus(-x))

  def sample(self, size):
    with torch.no_grad(): 
      z = Uniform(0., 1.).sample(size)
      return torch.log(z) - torch.log(1. - z)

