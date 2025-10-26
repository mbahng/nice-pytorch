import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import *
from tqdm import tqdm

def train_nice(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in tqdm(dataloader): 
    x = batch[0].to(model.device)

    # This should be included for discrete data
    x += torch.rand_like(x) / 256
    x = torch.clamp(x, 0, 1)

    _, log_likelihood = model(x) 

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

  for batch in tqdm(dataloader):
    x = batch[0].to(model.device)

    # This should be included for discrete data
    noise = torch.rand_like(x)
    x = torch.clamp((x * 255. + noise) / 256., 0, 1)

    _, log_likelihood, _ = model(x)

    model.zero_grad()
    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)

def train_glow(model, dataloader, optimizer): 
  model.train()
  total_loss = 0.0

  for batch in tqdm(dataloader):
    x = batch[0].to(model.device)

    # This should be included for discrete data
    noise = torch.rand_like(x)
    x = torch.clamp((x * 255. + noise) / 256., 0, 1)

    _, log_likelihood, _ = model(x)

    model.zero_grad()
    loss = -torch.mean(log_likelihood)
    total_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

  print(f'{total_loss / len(dataloader)}')
  return total_loss / len(dataloader)


def train(model, dataloader: DataLoader, optimizer: Optimizer): 
  if isinstance(model, NICE):
    return train_nice(model, dataloader, optimizer) 
  elif isinstance(model, RealNVP):
    return train_realnvp(model, dataloader, optimizer) 
  elif isinstance(model, Glow):
    return train_glow(model, dataloader, optimizer) 
  else: 
    raise NotImplementedError()

