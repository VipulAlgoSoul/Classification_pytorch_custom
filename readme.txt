readme
python version 3.10.4

## Creating dataset
1) create folder with trainable images
2) save csv file

There will be a yaml file dataset_csv.yaml in root directory, add changes there
#sample given in data folder

train_img_dir : "path to images"
train_label_csv : "path to csv"
label_header : Which header you want to consider for one hot encoding, please go through sample for more clarity

##Model creation
utils/myconvnet make changes only to the architecture and save 

## configuring pipeline
#sample config.yaml in root dir
INIT_LR : 0.003
EPOCHS : 10
BATCH_SIZE : 64
SPLIT : 0.2
OPTIMISER : ADAM
LOSS : CROSSENTROPYLOSS
PATIENCE : 7 #wait for 7 epochs for early stop
DELTA : 5 #minimum change req in loss for consider model is improving
NUM_CKPTS : 5 #the toal number of checkpoints to be saved in chekpoint directory
 
##run
python run.py 

##result will be in a new folder in run

## you will need a wandb account to visualize
##model original created in machine having GPU

##infer
python infer.py -im "path to image"