import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter




def calc_mean_std():
    from torch.utils.data import ConcatDataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)

    # stack all train images together into a tensor of shape
    # (50000, 3, 32, 32)
    x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

    # get the mean of each channel
    mean = torch.mean(x, dim=(0,2,3)) # tensor([0.4914, 0.4822, 0.4465])
    std = torch.std(x, dim=(0,2,3)) # tensor([0.2470, 0.2435, 0.2616])



# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.close()


def data_viewer(training_loader, classes):
    dataiter = iter(training_loader)
    images, labels = next(dataiter)
    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print(' '.join(classes[labels[j]] for j in range(4)))


def criterion_check():
    criterion = nn.CrossEntropyLoss()
    # NB: Loss functions expect data in batches, so we're creating batches of 4
    # Represents the model's confidence in each of the 10 classes for a given input
    dummy_outputs = torch.rand(4, 10)
    # Represents the correct class among the 10 being tested
    dummy_labels = torch.tensor([1, 5, 3, 7])
    print(dummy_outputs)
    print(dummy_labels)
    loss = criterion(dummy_outputs, dummy_labels)
    print('Total loss for this batch: {}'.format(loss.item()))
    
