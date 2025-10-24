import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import *
from tqdm import tqdm

def train_nice(model, dataloader, optimizer, device): 
  model.train()
  total_loss = 0.0

  for batch in tqdm(dataloader): 
    x = batch[0].to(device)
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

def train_realnvp(model, dataloader, optimizer, device): 
  model.train()
  total_loss = 0.0

  for batch in tqdm(dataloader): 
    x = batch[0].to(device)

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


def train(model, dataloader: DataLoader, optimizer: Optimizer, device): 
  if isinstance(model, NICE):
    return train_nice(model, dataloader, optimizer, device) 
  elif isinstance(model, RealNVP):
    return train_realnvp(model, dataloader, optimizer, device) 
  else: 
    raise NotImplementedError()

