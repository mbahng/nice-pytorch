import torch
from model import NICE

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

def train(model, dataloader, optimizer): 
  if isinstance(model, NICE):
    return train_nice(model, dataloader, optimizer) 
  else: 
    raise NotImplementedError()

