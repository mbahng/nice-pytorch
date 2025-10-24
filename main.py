import torch 

from dataset import *
from model import *
import traintest as tnt
import viz

torch.manual_seed(42)

ds = mnist_flat()
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True) 

model = RealNVP(
  idim=ds[0][0].size(), 
  n_coupling_layers=5, 
  neural_net_layers=6,
  hdim=torch.Size([50])
)

model.load_state_dict(torch.load("saved/state-dict.pt", weights_only=True))

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

best_loss = float('inf')

# visualizations may not make sense when you change
# viz.plot_distribution(model, ds) 
viz.plot_samples(model, savefig="fig/0.png")

for epoch in range(50): 
  loss = tnt.train(model, dl, optimizer)

  if loss < best_loss: 
    torch.save(model.state_dict(), f"saved/state-dict.pt")

  # viz.plot_distribution(model, ds)
  viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

# viz.plot_distribution(model, ds, savefig="fig/readme/moons.png")
viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

