import torch
import os
import numpy as np
from train import train_model
import time
from data_loading import generate_train_test_val_split

if __name__ == "__main__":
    print('0. Instantiating... ')
    start = time.time()

    # Fix seed to be able to reproduce experiments
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    params = {
        'ruta_AIDL_dataloader': "/content/drive/MyDrive/Postgrau_AIDL",
        'batch_size': 3,
        'shuffle': False,
        'num_workers': 1,
        "max_epochs": 100,
        'epochs': 3,
        'num_classes': 8,
        'lr': 1e-3,
        'useTrans': True,   # Use transformer in Video processing?
        'hop_length': 512,  # in num. of samples
        'n_fft': 2048,      # window in num. of samples'
        'sample_rate': 48000,
        'number_mfcc': 13,
        'n_mels': 100,
        'win_mac': 'mac',   # win for windows / mac for mac
        'mode': 'audio',    # video/audio/all
        'testing': True     # to speedup testing with fewer videos
    }

    trans_conf = {
        'nhead': 2,
        'd_model': 128,     # 128/256/512
        'num_layers': 2,
        'dim_feedforward': 512
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
    ENTITY = "aidl22_project"

    if params['win_mac'] == 'win':
        os.chdir("../")

    dir = os.getcwd() # aidl22_multimodal_sentiment_analysis dir

    # Creation of csv files
    print('1. Generating train/val/test split... ')
    _, _, _ = generate_train_test_val_split(dir, testing=params['testing'], train_size=0.8, test_size=0.2)

    if params['testing'] == True: # to speedup testing with fewer videos folder data2
        video_path = os.path.join(dir, 'data2')
        csv_file_train = os.path.join(dir, 'data2', 'train.csv')
        csv_file_val = os.path.join(dir, 'data2', 'val.csv')
        csv_file_test = os.path.join(dir, 'data2', 'test.csv')

    else:                          # all videos folder data
        video_path = os.path.join(dir, 'data')
        csv_file_train = os.path.join(dir, 'data', 'train.csv')
        csv_file_val = os.path.join(dir, 'data', 'val.csv')
        csv_file_test = os.path.join(dir, 'data', 'test.csv')

    print('2. Training model... ')
    my_model, yreal_train, ypred_train, ypred_test, yreal_test = train_model(params,
                                                                             trans_conf,
                                                                             emotion_mapping=emotion_mapping,
                                                                             video_path=video_path,
                                                                             csv_file_train=csv_file_train,
                                                                             csv_file_val=csv_file_val,
                                                                             mode=params['mode'])

    print('3. End... Elapsed time: ', time.time()-start)