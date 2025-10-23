from torchvision import transforms, datasets 

def mnist(): 
  return datasets.MNIST(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

def cifar10(): 
  return datasets.CIFAR10(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

def svhn(): 
  return datasets.SVHN(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

