import os
import shutil

def copy_file(base_path,model_dir,file):
	shutil.copy(os.path.join(base_path,file), os.path.join(model_dir,file))
def create(base_path):
	print(base_path)
	#create Runs folder
	run_path=os.path.join(base_path,"RUNS")
	if not os.path.exists(run_path):
		os.mkdir(run_path)
	#check model folders
	
	#check if atleast one folder exist
	models=os.listdir(run_path)
	os.mkdir(os.path.join(run_path, str(len(models)+1)))
	model_dir = os.path.join(run_path, str(len(models)+1))

	###
	#copy_file(base_path,model_dir,"config.yaml")
	#copy_file(base_path,model_dir,"dataset_csv.yaml")
	copy_file(os.path.join(base_path,"utils"),model_dir,"myconvnet.py")

	#create
	ckpt_dir = os.path.join(model_dir,"checkpoints")
	os.mkdir(ckpt_dir)

	#create result folder
	reslt=os.path.join(model_dir,"results")
	os.mkdir(reslt)

	return model_dir
	
