import os
import argparse
from utils.fromyaml import FromYaml
import cv2
import torch

base_dir=os.getcwd()
parser = argparse.ArgumentParser()

#Collect path of latest model
def collect_latest(base_dir):
    rp = os.path.join(base_dir,"RUNS")
    mn = max([int(i) for i in os.listdir(rp)])
    print("The latset model number is ..................................................................................",mn)
    mp = os.path.join(rp,str(mn))

    return os.path.join(mp,"model.pt"),os.path.join(mp,"labelmap.yaml")

def predict(img_p, model_p, label_map):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = torch.permute(torch.tensor(cv2.imread(img_p)/255), (2, 0, 1))
    img = img.float()
    img = img.unsqueeze(0)
    img=img.to(device)
    #print(img.shape)

    model = torch.load(model_p)
    model.eval()
    model= model.to(device)

    out = model(img)
    indx=torch.argmax(out,dim=-1).item()

    label = [key for key in label_map.keys() if label_map[key]==indx]

    print("The detected label is .......................................................................................",label[0])



def_model_path, def_label_map = collect_latest(base_dir)


##For dtaset yaml
parser.add_argument("-m","--model_path",help="path of pytorch path",
	default=def_model_path)

#for config yaml
parser.add_argument("-lm","--label_map",help="path of label map yaml",
	default=def_label_map)

parser.add_argument("-im","--image_input",help="path to image",
	default=def_label_map)

args=parser.parse_args()

print("The model being tested is .......................................................................................", args.model_path)
print("The Label map is ................................................................................................",args.label_map)
label_info = FromYaml.collect(args.label_map)
print("The labels are ", label_info)

predict(args.image_input, args.model_path, label_info)