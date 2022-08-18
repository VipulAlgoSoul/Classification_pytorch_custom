import yaml
from yaml.loader import SafeLoader
import os

class FromYaml():
    '''Getting info from .yaml file'''
    
    # def __init__(self,file_path):
    #
    #     '''yaml_type is 1) dataset :points to datset yaml
    #     typ2 is network config'''
    #
    #     self.yaml_path = file_path
    #
    # @property
    @staticmethod
    def collect(yaml_path):
        
        with open(yaml_path) as f:
            return yaml.load(f, Loader=SafeLoader)
    @staticmethod
    def create_yaml(model_path, data, name):
        #set isinstance
        yml_name = name
        yml_pth=os.path.join(model_path,yml_name)
        with open(yml_pth, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
