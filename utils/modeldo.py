import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import wandb

class modeldo():
	def __init__(self,model_dir,total_data, split, model, optimizer,
			lossfunc, int_lr,batch_size, epochs, patience,delta, num_ckpts):
		
		print("Initializing modeldo ....................................................................................")

		self.model_dir=model_dir
		self.ckpt_dir=os.path.join(self.model_dir,"checkpoints")
		self.res_dir=os.path.join(self.model_dir,"results")

		self.total_data=total_data
		self.split=split
		self.INIT_LR = int_lr
		self.model=model

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model=model.to(self.device)
		self.optimizer=self.init_optimizer(optimizer)
		self.lossfunc=self.get_loss_func(lossfunc)
		self.batch_size=batch_size
		self.train_dataloader, self.val_dataloader = self.prep_data()
		self.epochs=epochs
		print("The length of trainable is {} and validation data is {}".format(len(self.train_dataloader.dataset), len(self.val_dataloader.dataset)))
		self.history ={"Training Loss":[], "Training Accuracy":[],"Validation Loss":[], "Validation Accuracy":[],"epoch":[]}


		self.patience = patience
		self.delta = delta
		self.es_score=np.Inf
		self.es_counter = 0
		self.es_status=False
		self.es_param=0
		self.ckpt_paths=[]
		self.num_ckpts=num_ckpts

		self.wandbdict = {k: 0 for k in self.history.keys() if k != "epoch"}

		wandb.init(
			# Set the project where this run will be logged
			project="mytroch_classification",
			# We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
			name=f"experiment_{os.path.basename(self.model_dir)}",
			# Track hyperparameters and run metadata
			config={
				"learning_rate": self.INIT_LR,
				"architecture": "CNN",
				"dataset": "Pokemon",
				"epochs": self.epochs,
			})
		wandb.watch(self.model)
	def early_stop(self,parameter,name,t):
		#start initial score with infinity
		#append new scores
		#check if new score less than prvs score
		
		###########################################################################################################################################################
		#craete a better early stopping mechnaism
		#if self.history["Training Loss"][-1] > self.history["Validation Loss"][-1]:
		#	print("Early stopping due to over fitting /////////////////////////////////////////////////////////")
		#	ckpt_name = "NON_OVERFITTED"+"_"+str(parameter)+".pt"
		#	torch.save({'epoch':t, 'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(),'loss':self.lossfunc,}, os.path.join(self.ckpt_dir,ckpt_name))
		#	self.es_status =True

		############################################################################################################################################################

		if parameter < self.es_score-self.delta:
			#print("creating checkpoints,........................")
			#save ckpt
			ckpt_name = name+"_"+str(parameter)+".pt"
			torch.save({'epoch':t, 'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(),'loss':self.lossfunc,}, os.path.join(self.ckpt_dir,ckpt_name))
			self.counter = 0
			self.ckpt_paths.append(os.path.join(self.ckpt_dir,ckpt_name))

			if len(self.ckpt_paths) > self.num_ckpts:
				#removing first ckpt
				to_rem = self.ckpt_paths.pop(0)
				os.remove(to_rem)

		else:
			self.es_counter+=1
			if self.counter >= self.patience:
				print("Breaking at early stop ..........................................................................")
				self.es_status =True
				self.es_param = parameter
			
			
		
		
	def prep_data(self):
		total=self.total_data.__len__()
		train_num=int(total*self.split)
		val_num=total-train_num

		
		print("The total number of data points are {}, \n the split is {}\n, the train num is {}, test num is {}".format(total,self.split,train_num,val_num))
		train_set, val_set = torch.utils.data.random_split(self.total_data, [val_num, train_num]) 
		train_dataloader=DataLoader(train_set,batch_size=self.batch_size, shuffle=True)
		val_dataloader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
		print("The shape of train set and val set is ...................................................................{}, {}".format(len(train_dataloader), len(val_dataloader)))
		self.verbose_train=len(val_dataloader)
		return train_dataloader, val_dataloader
	
	def get_loss_func(self,lossfunc):
		if lossfunc.lower()=="crossentropyloss":
			print("The loss function is  ...............................................................................nn.CrossEntropyLoss")
			return nn.CrossEntropyLoss()
		
	def init_optimizer(self,optimizer):
		if optimizer.lower()=="adam":
			print("The optimizer is ....................................................................................adam")
			return Adam(self.model.parameters(),lr=self.INIT_LR)
		
	def train(self):
		size = len(self.train_dataloader.dataset) 
		#print(len(self.train_dataloader.dataset))
		train_loss, correct =0,0
		for batch, (X,y) in enumerate(self.train_dataloader):
			
			X=X.to(self.device)
			y=y.to(self.device)
			#print(X.shape,y.shape,X.dtype,">>>>>>>>>>>>>>>>>>>>>>>>>>")
			ypred =  self.model(X)
			loss  = self.lossfunc(ypred,y)
			train_loss+=loss.item()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			#print("prv correct", correct)
			correct += self.accuracy_classification(ypred,y)
			#print("The correct is >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",correct)
			
			if (batch%self.verbose_train) == 0:
				loss, current =loss.item(), (batch+1)*len(X)
				print("The loss is {}, after seeing {} datapoints".format(loss,current))

		train_loss/=(batch+1)
		correct/=size
		#print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",train_loss)
		self.history["Training Loss"].append(train_loss)
		self.history["Training Accuracy"].append(correct)

	def validate(self):
		size=len(self.val_dataloader.dataset)
		num_batches = len(self.val_dataloader)
		test_loss, correct =0,0

		with torch.no_grad():
			for X,y in self.val_dataloader:
				X=X.to(self.device)
				y=y.to(self.device)
				pred=self.model(X)

				test_loss+=self.lossfunc(pred,y).item()
				correct += self.accuracy_classification(pred,y)
				
	
		test_loss /= num_batches
		correct /= size
		self.history["Validation Loss"].append(test_loss)
		self.history["Validation Accuracy"].append(correct)
		print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

	def accuracy_classification(self,pred,ylabel):
    		argmax_indices_pred= torch.argmax(pred,dim=-1)
    		argmax_indices_y=torch.argmax(ylabel,dim=-1)
    
    		difft = argmax_indices_pred-argmax_indices_y    	
    		return torch.numel(difft)-torch.count_nonzero(difft).item()


	def do_training(self):

		for t in range(self.epochs):
			print("The number of epochs are  ---------------------------------------------------------------------------",t)
			self.train()
			self.validate()
			self.history["epoch"].append(t)
			print("Finished validating")

			self.early_stop(self.history["Validation Loss"][-1],"val_loss",t)

			if self.es_status == True: #checking early stopping flag
				break


		#torch.save(self.model.state_dict(), os.path.join(self.model_dir,"modelc.pt"))
		torch.save(self.model, os.path.join(self.model_dir, "model.pt"))
		# wandb.log(self.history)
		his_no_epoch={k:self.history[k] for k in self.history.keys() if k !="epoch"}
		wandb.log({"Accuracy Metrics": wandb.plot.line_series(
			xs=np.array(self.history["epoch"]),
			ys=[np.array(i) for i in {k:his_no_epoch[k] for k in his_no_epoch.keys() if k.endswith("Accuracy")}.values()],
			keys=[i for i in his_no_epoch.keys() if i.endswith("Accuracy")],
			title="Accuracy",
			xname="epochs")})

		wandb.log({"Loss Metrics": wandb.plot.line_series(
			xs=np.array(self.history["epoch"]),
			ys=[np.array(i) for i in
				{k: his_no_epoch[k] for k in his_no_epoch.keys() if k.endswith("Loss")}.values()],
			keys=[i for i in his_no_epoch.keys() if i.endswith("Loss")],
			title="Loss",
			xname="epochs")})



		wandb.finish()
		self.show_result()
	
	def show_result(self):

		plt.subplot(1, 2, 1)
		#print("the length of history point are ", len(self.history["Training Loss"]), len(self.history["Validation Loss"]))
		plt.plot( self.history["epoch"],self.history["Validation Loss"],self.history["epoch"],self.history["Training Loss"])
		plt.legend(["Validation Loss", "Training Loss"], loc ="lower right")

		plt.subplot(1, 2, 2)
		plt.plot( self.history["epoch"],self.history["Validation Accuracy"],self.history["epoch"],self.history["Training Accuracy"])
		plt.legend(["Validation Accuracy", "Training Accuracy"], loc ="lower right")
		plt.savefig(os.path.join(self.res_dir,"Loss-Accuracy.png"), dpi=1000)

		#plt.show()
		#plt.close()

		



			

		