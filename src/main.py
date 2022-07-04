import torch
import torch.nn as nn
import os
import numpy as np
import json
from train import train_model
from data_loading import generate_train_test_val_split


if __name__ == "__main__":

    print('settings loaded1')
    print('settings loaded2')
    print('settings loaded3')

    # Fix seed to be able to reproduce experiments
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    params = {
    "ruta_AIDL_dataloader" : "/content/drive/MyDrive/Postgrau_AIDL",
    'batch_size': 3,
    'shuffle': False,
    'num_workers': 1,
    "max_epochs" : 100,
    'epochs': 15,
    'num_classes': 8,
    'lr': 1e-3,
    'useTrans': True
    }
    trans_conf = {
        'nhead': 2,
        'd_model': 128, # fer mes petit -> 128/256/512
        'num_layers': 2, #
        'dim_feedforward': 512  # era 2048 31/05: jugar amb el learning rate + scheduler (scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)) --> mes petit!!!
    }
    emotion_mapping = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    PROJECT_NAME = "multimodal_sentiment_analysis"
    ENTITY = "alexabos"

    dir = os.getcwd()
    video_path = os.path.join(dir, 'data')
    csv_file = os.path.join(dir, 'data/train.csv')

    # Creation of csv files
    print('Generating train/val/test split1')
    print('Generating train/val/test split2')
    print('Generating train/val/test split3')

    _, _, _ = generate_train_test_val_split(dir, train_size=0.8, test_size=0.2)

    csv_file_train = os.path.join(dir, 'data', 'train.csv')
    csv_file_val = os.path.join(dir, 'data', 'val.csv')
    csv_file_test = os.path.join(dir, 'data', 'test.csv')

    criterion = nn.functional.nll_loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    my_model, yreal_train, ypred_train, ypred_test, yreal_test = train_model(params,
                                                                             trans_conf,
                                                                             device,
                                                                             criterion,
                                                                             emotion_mapping=emotion_mapping,
                                                                             video_path=video_path,
                                                                             csv_file_train=csv_file_train,
                                                                             csv_file_val=csv_file_val)




