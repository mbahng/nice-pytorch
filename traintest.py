import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import NICE, RealNVP, NormalizingFlow

def train_nice(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in dataloader: 
    x = batch[0]
    x = x.view(x.size(0), -1)

    z, log_likelihood = model(x) 

    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    model.zero_grad() 
    loss.backward()
    optimizer.step()

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)

def train_realnvp(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in dataloader: 
    x = batch[0]
    x = x.view(x.size(0), -1)

    z, log_likelihood = model(x) 

    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    model.zero_grad() 
    loss.backward()
    optimizer.step()

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)

def train(model: NormalizingFlow, dataloader: DataLoader, optimizer: Optimizer): 
  if isinstance(model, NICE):
    return train_nice(model, dataloader, optimizer) 
  elif isinstance(model, RealNVP):
    return train_nice(model, dataloader, optimizer) 
  else: 
    raise NotImplementedError()

