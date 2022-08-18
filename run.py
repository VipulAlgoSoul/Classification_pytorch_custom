import os
import torch
import argparse 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils.labelmapcsv import LabelMapCsv
from utils.loaddatasetcsv import LoadDataCsv
from utils.fromyaml import FromYaml
from utils.myconvnet import myConvnet
from utils.modeldo import modeldo
from utils.makedir import create
import shutil
import wandb
wandb.login()

base_dir=os.getcwd()
parser = argparse.ArgumentParser()

##For dtaset yaml
parser.add_argument("-dy","--dataset_yaml",help="path of dataset yaml path",
	default=os.path.join(base_dir,"dataset_csv.yaml"))

#for config yaml
parser.add_argument("-cy","--config_yaml",help="path of config yaml path",
	default=os.path.join(base_dir,"config.yaml"))

parser.add_argument("-tl", "--target_csv_label",
	help="Select the Target column in dataframeto one-hot-encode",
	default=None)
parser.add_argument("-v","--visualize",type= bool, 
	help="visulaize the image and datapoints",default=False)
parser.add_argument("-s","--train_validation_split",
	help="percent of total data considered for training, rest for validation",
	type = float, default=None)
parser.add_argument("-e","--epochs",
	help="number of epochs",
	type = int, default=None)
parser.add_argument("-bs","--batchsize",
	help="the batch size",
	type = int, default=None)

args=parser.parse_args()

try:
	# creating folders
	model_dir = create(base_dir)
	print("The base dirctory is", base_dir)

	# Read Yaml
	#data_yaml = FromYaml(args.dataset_yaml)
	data_info = FromYaml.collect(args.dataset_yaml)
	print(data_info)


	############################################################### DATA YAML ##########################################
	# Label Mapping
	if args.target_csv_label:
		print("Collecting Label header from Commandline")
		data_info["label_header"] = args.target_csv_label
		# labelizer = LabelMapCsv(data_info["train_label_csv"], args.target_csv_label)
		labelizer = LabelMapCsv(data_info["train_label_csv"], data_info["label_header"])
	elif "label_header" in data_info.keys():
		print("Collecting Label header from ",args.dataset_yaml)
		labelizer = LabelMapCsv(data_info["train_label_csv"], data_info["label_header"])
	else:
		print("Collect label header, not present in commandline argumnet or dataset_yaml")
		labelizer = LabelMapCsv(data_info["train_label_csv"])

	print("The new data info is ",data_info)
	FromYaml.create_yaml(model_dir,data_info,"dataset_csv.yaml")
	print("Saving the new dataset yaml to {}".format(model_dir))


	label_map_dict=labelizer.create_map()
	print("The mapped label to integers are \n",label_map_dict)
	print("Saving the label map dict as yaml in {}".format(model_dir))
	FromYaml.create_yaml(model_dir, label_map_dict, "labelmap.yaml")
	# Label Encoding

	####################################################################################################################

	print("Colleecting Data")
	total_data = LoadDataCsv(csv_file = data_info["train_label_csv"],
						  img_dir=data_info["train_img_dir"],
						  csv_label_dict=label_map_dict,visualize=args.visualize, transform=ToTensor())
	print("The Total data Extracted ....................................................................................")

	print("Collecting data from config yaml")
	config_info = FromYaml.collect(args.config_yaml)
	# print("Printing original config data",config_info)
	# train validation split
	if args.train_validation_split:
		split=args.train_validation_split
		config_info["SPLIT"]=split
	else:
		split = config_info["SPLIT"]

	print("Train Validation split is ...................................................................................",split)

	if args.epochs:
		epochs=args.epochs
		config_info["EPOCHS"]=epochs
	else:
		epochs = config_info["EPOCHS"]


	print("The number of epochs is  ....................................................................................",epochs)

	if args.batchsize:
		batchsize=args.batchsize
		config_info["BATCH_SIZE"]=batchsize
	else:
		batchsize = config_info["BATCH_SIZE"]

	print("the batch size is  ..........................................................................................",batchsize)

	#test_set = LoadDataCsv(csv_file = data_info["test_label_csv"],
	#                      img_dir=data_info["test_img_dir"],
	#                      csv_label_dict=label_map_dict,visualize=args.visualize)

	print("The new data config info is ", config_info)
	print("Saving config info to {}".format(model_dir))
	FromYaml.create_yaml(model_dir,config_info,"config.yaml")

	my_conv_net = myConvnet(num_channels=3, output_dim=len(label_map_dict.keys()))
	print(my_conv_net)

	model_do = modeldo(model_dir= model_dir,total_data=total_data, split=split, model=my_conv_net, optimizer=config_info["OPTIMISER"],
			lossfunc=config_info["LOSS"], int_lr=config_info["INIT_LR"],batch_size=batchsize,epochs=epochs,patience=config_info["PATIENCE"], delta = config_info["DELTA"], num_ckpts=config_info["NUM_CKPTS"])
	model_do.do_training()

except Exception as e:
	print(e)
	shutil.rmtree(model_dir, ignore_errors=True)
