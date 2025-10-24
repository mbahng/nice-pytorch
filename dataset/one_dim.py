import torch
import math
from torch.utils.data import TensorDataset
from sklearn.datasets import make_s_curve, make_moons, make_circles
from torchvision import transforms, datasets 

def two_gaussians(x = 0.8, N=100):
  gaussian1 = 4 + math.sqrt(x) * torch.randn(N, 2)
  gaussian2 = -4 + (1/math.sqrt(x)) * torch.randn(N, 2)
  mask = (torch.rand(N) > 0.5).unsqueeze(1).expand(N, 2)
  Y = mask * gaussian1 + (~mask) * gaussian2
  ds = TensorDataset(Y)
  return ds

def four_gaussians(N=100):
  """
  Toy dataset with 4 Gaussian components arranged in a ring pattern
  """
  n_per_mode = N // 5

  # Create 4 Gaussians in a ring pattern
  angles = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2])
  radius = 5.0
  ring_data = []

  for angle in angles:
    center_x = radius * math.cos(angle)
    center_y = radius * math.sin(angle)
    center = torch.tensor([center_x, center_y])
    samples = center + 0.5 * torch.randn(n_per_mode, 2)
    ring_data.append(samples)

  # Combine all samples
  Y = torch.cat(ring_data, dim=0)

  # Shuffle the data
  perm = torch.randperm(Y.size(0))
  Y = Y[perm]

  ds = TensorDataset(Y)
  return ds

def moon(N=100, noise=0.05): 
  """
  Toy dataset consisting of two cresent moons
  """
  Y, _ = make_moons(N, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float)
  return TensorDataset(Y)

def circles(N=100, scale=0.5, noise=0.05): 
  """
  Toy dataset consisting of two concentric circles. 
  Factor = scale between inner vs outer circle. 
  """
  Y, _ = make_circles(N, factor=scale, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float) 
  return TensorDataset(Y)

def s_curve(N = 100, noise=0.05): 
  Y, _ = make_s_curve(N, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float) 
  return TensorDataset(Y)

def swiss_roll(N = 100, noise = 0.05): 
  Y, _ = swiss_roll(N, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float) 
  return TensorDataset(Y)

def mnist_flat(): 
  transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: torch.flatten(x))
  ])
  return datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)

def cifar10_flat(): 
  transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: torch.flatten(x))
  ])
  return datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)

def svhn_flat(): 
  transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: torch.flatten(x))
  ])
  return datasets.SVHN(root='./dataset', train=True, transform=transform, download=True)

