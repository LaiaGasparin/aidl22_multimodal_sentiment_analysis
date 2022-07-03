import torch
import torch.nn as nn
import os
import numpy as np
import json
from train import train_model
from data_loading import generate_train_test_val_split

with open("src/settings.json") as config_file:
    settings = json.load(config_file)

# Initialize device either with CUDA or CPU. For this session it does not
# matter if you run the training with your CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fix seed to be able to reproduce experiments
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dir = "."
#todo: in settings.json
video_path = os.path.join(dir, 'data')
csv_file = os.path.join(dir, 'data/train.csv')

# Creation of csv files
_, _, _ = generate_train_test_val_split(dir, train_size = 0.8, test_size = 0.2)

csv_file_train = os.path.join(dir, 'data/train.csv')
csv_file_val = os.path.join(dir, 'data/val.csv')
csv_file_test = os.path.join(dir, 'data/test.csv')

criterion = nn.functional.nll_loss 
params = settings["params"]
trans_conf = settings["transformer_conf"]
my_model, yreal_train, ypred_train, ypred_test, yreal_test  = train_model(params, 
                                                                          trans_conf,
                                                                          device, 
                                                                          criterion, 
                                                                          video_path = video_path, 
                                                                          csv_file_train=csv_file_train,
                                                                          csv_file_val=csv_file_val) 


    

