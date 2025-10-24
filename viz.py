import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from model import *
from typing import Union

def plot_distribution(model, ds: TensorDataset, savefig = None): 
  """
  Only meaningful for 2-dimensional datasets. 
  Used to see how it evolves. 
  """
  if model.idim != torch.Size([2]): 
    raise AttributeError("The dimension must be 2. ")

  model.eval()
  with torch.no_grad():
    # Create a grid of points
    x_range = torch.linspace(-10, 10, 200)
    y_range = torch.linspace(-10, 10, 200)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Compute log-likelihood for each point
    _, log_likelihood = model(grid_points)
    log_likelihood = log_likelihood.reshape(200, 200)

    # Convert to probability density
    prob_density = torch.exp(log_likelihood)

    # Plot the heatmap and scatter 
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_density.T, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis', aspect='auto') # type: ignore
    plt.scatter(ds.tensors[0][:,0], ds.tensors[0][:,1], c="r", s=3)
    plt.colorbar(label='Probability Density')
    plt.title('Learned Probability Distribution (Flow Model)')
    if savefig:
      plt.savefig(savefig)
      plt.close()
    else:
      plt.show(block=False)
      plt.pause(0.1)
      plt.close()

def plot_samples(model, savefig=None): 
  """
  Randomly sample 25 and visualize them. 
  """ 
  # This is only meaningful for high dimensional datasets like MNIST. 
  if len(model.idim) != 3 and model.idim.numel() != 784:
    raise AttributeError("The dimension is wrong.")

  model.eval()
  with torch.no_grad(): 
    # make sure to permute so that channel dimension is last
    img = model.sample(25).reshape(-1, 1, 28, 28).permute(0, 2, 3, 1).detach().cpu().numpy()
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img[i], cmap='gray'); ax.axis('off')
    plt.tight_layout()
    if savefig:
      plt.savefig(savefig)
      plt.close(fig)
    else:
      plt.show(block=False)
      plt.pause(0.1)
      plt.close()

