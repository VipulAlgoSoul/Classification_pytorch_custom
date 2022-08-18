import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import os

class LoadDataCsv(Dataset):
    '''This class is used to load data from csv files'''
    
    def __init__(self, csv_file, img_dir,csv_label_dict, transform=None,
                 target_transform=None, visualize=False):
        '''Initializing variables'''
        self.csv_file=pd.read_csv(csv_file).dropna()
        self.img_dir=img_dir
        self.label_dict=csv_label_dict
        self.transform=transform
        self.target_transform=target_transform
        self.visualize=visualize
        self.iml=os.listdir(self.img_dir)
        
        if self.visualize==True:
            print("Visualizing Data")
            self.visualize_data()
        
    def __len__(self):
        return len(self.csv_file)
        
    def __getitem__(self,idx):
        '''Get items'''
        img_name=self.csv_file.iloc[idx,0]       
        img_file=self.iml[[i for i,elem in enumerate(self.iml) if elem.split(".")[0]==img_name][0]]        
        img_path=os.path.join(self.img_dir,img_file)
        img=torch.permute(torch.tensor(cv2.imread(img_path)/255),(2,0,1))
        
        #print(img.shape, img.dtype)
        label_str=self.csv_file.iloc[idx,1]
        label_int=int(self.label_dict[label_str])
        
        label_one_hot=np.zeros((len(self.label_dict.keys())))
        label_one_hot[label_int]=1.0
        
        if self.visualize==True:
            print(self.label_dict)
            print(label_str)
            print(label_one_hot)
            
#        if self.transform:
#            img = self.transform(img)
#         if self.target_transform:
#             label = self.target_transform(label)
        
        return img.float(),torch.Tensor(label_one_hot).float() 
    
    def getitemname(self,idx):
        '''Get items'''
        img_name=self.csv_file.iloc[idx,0]       
        img_file=self.iml[[i for i,elem in enumerate(self.iml) if elem.split(".")[0]==img_name][0]]        
        img_path=os.path.join(self.img_dir,img_file)
        img=torch.tensor(cv2.imread(img_path))
        
        #print(img.shape, img.dtype)
        label_str=self.csv_file.iloc[idx,1]
        
        return img,label_str 
        
    def visualize_data(self):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.csv_file), size=(1,)).item()
            img, label = self.getitemname(sample_idx)
            figure.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

        
#iloc generality
        
