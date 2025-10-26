import torch 

from dataset import *
from model import *
import traintest as tnt
import viz

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = cifar10()
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True) 

# model = RealNVP(
#   idim=ds[0][0].size(), 
#   n_scales=2, # should always be 2 for cifar10
#   neural_net_layers=8,
#   hdim=torch.Size([64]), 
#   device=device
# )

model = Glow(
  idim=ds[0][0].size(), 
  n_levels=2, # should always be 2 for cifar10 
  flow_depth=3, 
  neural_net_layers=8,
  hdim=torch.Size([64]), 
  device=device
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

best_loss = float('inf')

# viz.plot_distribution(model, ds) 
viz.plot_samples(model, savefig="fig/0.png")

for epoch in range(50): 
  loss = tnt.train(model, dl, optimizer)

  # if loss < best_loss: 
  #   torch.save(model.state_dict(), f"saved/state-dict.pt")

  # viz.plot_distribution(model, ds)
  viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

# viz.plot_distribution(model, ds, savefig="fig/readme/moons.png")
# viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

