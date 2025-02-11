import torchvision as tv
import torch
import matplotlib.pyplot as plt
import random


def generate_train_tuple(dataset):
  random_numbers = []
  random_images = []

  for _ in range(4):
    rand = random.randint(1, len(dataset))-1
    random_numbers.append(dataset.__getitem__(rand)[1])
    random_images.append(dataset.__getitem__(rand)[0][0])
  
  top = torch.cat([random_images[0], random_images[1]], dim=1)
  bottom = torch.cat([random_images[2], random_images[3]], dim=1)
  random_images = torch.cat([top, bottom], dim=0)  
  
  return random_images, random_numbers

if __name__ == '__main__':
  
    train_dataset = tv.datasets.MNIST(root='src/train_data', train=True, download=True, transform=tv.transforms.ToTensor())
    test_dataset = tv.datasets.MNIST(root='src/test_data', train=False, download=True, transform=tv.transforms.ToTensor())

    test = generate_train_tuple(train_dataset)
    print(test[1])
    plt.imshow(test[0], cmap='grey')
    plt.show()
    print(test[0].shape)