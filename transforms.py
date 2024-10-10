import torch
from torchvision import datasets, transforms
from torch import nn

import numpy as np
import cv2

import pdb

def image_convert(img):
    image = img.cpu().clone().detach().numpy()
    image = image.transpose(1,2,0)
    print(image.shape)
    image = image*(np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5)))
    image = image.clip(0,1)
    return image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#data_transforms = {
#'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#}

data_transforms = {
'train': transforms.Compose([transforms.Resize((32,32)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomAffine(0, shear = 10, scale = (0.8, 1.2)), transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
'val': transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
}

#transform_train = data_transforms['train']
#transform_valid = data_transforms['val']
print(data_transforms['train'])
print(data_transforms['val'])

train_dataset = datasets.CIFAR10(root = './test_data', train = True, download = True, transform = data_transforms['train'])
valid_dataset = datasets.CIFAR10(root = './test_data', train = False, download = True, transform = data_transforms['val'])

train_loader = iter(torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True))
valid_loader = iter(torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 100, shuffle = False))

#classes = { 'plane': 1, 'car':2, 'bird': 3, 'cat': 4, 'dear': 5, 'dog': 6, 'frog': 7, 'horse':8, 'ship': 8, 'truck': 9}
classes = ['plane', 'car', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']
images, labels = train_loader.next()



for idx in np.arange(20):
    cv2.imshow(classes[labels[idx].item()], image_convert(images))

#x = torch.rand(8, 1, 2, 2)
#print(x)

#dataset = transform_dataset(x, transform)

#for item in dataset:
#    print(item)
