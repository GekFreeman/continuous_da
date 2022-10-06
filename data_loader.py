# -*- coding:utf-8 -*-
from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import Sampler
import pdb
import numpy as np

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader

def load_training_index(root_path, dir, batch_size, indices, kwargs, target=False, pseudo=None):
    """
    target: 用于判断是否要生成带伪标签的数据集，是则不用sampler
    """
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    sampler = SubsetSampler(indices)  
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    if target:
        data = CustomDataset(data, indices, pseudo)
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True, **kwargs)
    return train_loader
    
def load_testing_index(root_path, dir, batch_size, indices, kwargs, target=False, pseudo=None):
    """
    target: 用于判断是否要生成带伪标签的数据集，是则不用sampler
    """
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    sampler = SubsetSampler(indices)  
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    if target:
        data = CustomDataset(data, indices, pseudo)
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=False, **kwargs)
    return train_loader
    
class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
 
    def __init__(self, indices):
        self.indices = indices
 
    def __iter__(self):
        return iter(self.indices)
 
    def __len__(self):
        return len(self.indices)
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices, pseudo):
        self.data = data
        self.indices = indices
        self.pseudo = pseudo
    def __getitem__(self, index):
        
        return self.data[self.indices[index]][0], self.pseudo[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.indices)
    
    
def load_training_source_debug(root_path, source_name, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + source_name, transform=transform);
    
    transform2 = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data2 = datasets.ImageFolder(root=os.path.join(root_path, source_name), transform=transform2)
    
    total_num = len(data.imgs)
    total_indices = list(range(total_num))
    data_cls1=list(np.random.choice(total_indices,size=total_num * 4 // 5,replace=False))
    data_cls2=[x for x in total_indices if x not in data_cls1]
    
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(data_cls1)                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(data_cls2)
    
    
    
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data2, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2