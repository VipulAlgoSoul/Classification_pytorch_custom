import torch 
import numpy as np

class data_creator():
	def __init__(self,shape,batch_size,initializer):
		self.shape=shape
		self.batch_size=batch_size
		self.initializer=initializer

		self.creatable_shape()

	
	def create(self):
		if self.initializer=="random":
			self.result=torch.rand(self.shape_cr)
			return self.result
	
	def creatable_shape(self):
		self.shape_cr=[self.batch_size]
		self.shape_cr.extend([i for i in self.shape])

x=data_creator((1,30,28),32,"random")
xx=x.create()

print(xx.shape)
