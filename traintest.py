import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import *

def train_nice(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in dataloader: 
    x = batch[0]
    x = x.view(x.size(0), *model.idim)
    x += torch.rand_like(x) / 256
    x += torch.clamp(x, 0, 1)

    z, log_likelihood = model(x) 

    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    loss.backward()
    optimizer.step()
    model.zero_grad() 

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)

def train_realnvp(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in dataloader: 
    x = batch[0]
    # x = x.view(x.size(0), *model.idim)

    # This should be included for discrete data
    x += torch.rand_like(x) / 256
    x += torch.clamp(x, 0, 1)

    z, log_likelihood = model(x) 

    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    model.zero_grad() 
    loss.backward()
    optimizer.step()

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)


def train(model, dataloader: DataLoader, optimizer: Optimizer): 
  if isinstance(model, NICE):
    return train_nice(model, dataloader, optimizer) 
  elif isinstance(model, RealNVP):
    return train_realnvp(model, dataloader, optimizer) 
  else: 
    raise NotImplementedError()

