import os
import nibabel as nib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class BaselineDataset(Dataset):
    def __init__(self,annotations_file,img_dir, img_type='midsagittal', train=True, transform=None, target_transform=None):

        # TODO : add training and validation datasets usiing train function argument
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_type = img_type

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        # print(self.img_labels[index])
        img_path = os.path.join(self.img_dir,'MALPEM-' + self.img_labels.iloc[index,0])
        image = nib.load(img_path).get_fdata()

        # midsagittal image
        label = self.img_labels.iloc[index,1]

        if self.img_type == 'midsagittal':
            image = image[:,:,(image.shape[2]//2)]
        elif self.img_type == 'parasagittal':
            image = image[:,:,(image.shape[2]//2) + 1]

        # TODO : add more img_types [midaxial, midcoronal, paraaxial, paracoronal]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def load_dataset():
    img_dir = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir)) + '/data/features_CrossSect-5074_LongBLM12-802_LongBLM24-532')
    annotations_file = '/'.join(img_dir.split('/')[:-1]) + '/ADNI_MALPEM_baseline_1069.csv'
        
    data = BaselineDataset(annotations_file=annotations_file,img_dir=img_dir)

    return data

if __name__ == "__main__":
    data = load_dataset()