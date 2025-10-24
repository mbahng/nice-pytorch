import torch 

from dataset import *
from model import *
import traintest as tnt
import viz

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = mnist_flat()
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True) 

model = NICE(
  idim=ds[0][0].size(), 
  n_coupling_layers=4, 
  neural_net_layers=6,
  hdim=torch.Size([1000]), 
  device=device
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

best_loss = float('inf')

# viz.plot_distribution(model, ds) 
viz.plot_samples(model, savefig="fig/0.png")

for epoch in range(50): 
  loss = tnt.train(model, dl, optimizer, device)

  if loss < best_loss: 
    torch.save(model.state_dict(), f"saved/state-dict.pt")

  # viz.plot_distribution(model, ds)
  viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

# viz.plot_distribution(model, ds, savefig="fig/readme/moons.png")
# viz.plot_samples(model, savefig=f"fig/{epoch+1}.png")

