import pandas as pd

#create a classification label map
class LabelMapCsv():
    '''create a label map to integer'''
    
    def __init__(self,csv_path,label_header=None):
        self.csv_path=pd.read_csv(csv_path)
        self.label_header=label_header
        if self.label_header==None:
            print("The headers are\n",[col for col in self.csv_path])
            self.label_header=input("Type the Header Column which has to be mapped")
        
    def create_map(self):
        return {elem:i for i,elem in enumerate(self.csv_path[self.label_header].unique())}
