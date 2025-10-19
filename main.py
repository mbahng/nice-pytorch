import torch 
import math

from dataset import *
from model import NICE, RealNVP
import traintest as tnt
import viz

FlowModel = RealNVP

ds = toy_2gaussians(N=1000)
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True) 

input_dim = ds[0][0].size() # change if you are using different dataset

model = FlowModel(
  input_dim=math.prod(input_dim), 
  n_coupling_layers=4, 
  neural_net_layers=3,
  hidden_dim=50
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

best_loss = float('inf')

# visualizations may not make sense when you change
viz.plot_distribution(model, ds) 

for epoch in range(25): 
  loss = tnt.train(model, dl, optimizer)

  if loss < best_loss: 
    torch.save(model.state_dict(), f"saved/state-dict.pt")

  viz.plot_distribution(model, ds)

viz.plot_distribution(model, ds, savefig="fig/teaser.png")

