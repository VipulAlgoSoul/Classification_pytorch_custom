import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


class myConvnet(nn.Module):
	
    
	def __init__(self, num_channels, output_dim):
		
		super(myConvnet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=num_channels,out_channels=16,stride=1,kernel_size=(3,3))
		self.relu1 = nn.ReLU()
		self.mp1   = nn.MaxPool2d(kernel_size=(2,2))
		self.fc1   = nn.Linear(16*59*59,output_dim)

	def forward(self,x):
		x=self.relu1(self.conv1(x))
		x=self.mp1(x)
		x=x.view(-1,16*59*59)
		x=self.fc1(x)
		
		return x
