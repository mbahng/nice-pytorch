from torchvision import transforms, datasets 

def mnist(): 
  return datasets.MNIST(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

def cifar10(): 
  return datasets.CIFAR10(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

def svhn(): 
  return datasets.SVHN(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

def celebA(): 
  return datasets.CelebA(
    root='./dataset', 
    split='all',
    transform=transforms.Compose([
      transforms.CenterCrop((216, 176)),  # Crop to desired size
      transforms.ToTensor()
    ]), 
    download=True
  )

def flowers102():
  return datasets.Flowers102(
    root='./dataset',
    transform=transforms.Compose([
      transforms.Resize(512),           # Resize shorter side to 512
      transforms.CenterCrop(512),       # Crop to 512x512
      transforms.ToTensor()
    ]),
    download=True
  )
