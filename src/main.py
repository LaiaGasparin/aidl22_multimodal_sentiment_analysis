import torch
import torch.nn as nn
import os
import numpy as np
import json
from train_video import train_model_video
from train_audio import train_model_audio
from model_audio import AudioNN, MyVGG
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig, Wav2Vec2Processor


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

#DATALOADER
csv_file_train = os.path.join(dir, 'data/train.csv')
csv_file_val = os.path.join(dir, 'data/val.csv')
csv_file_test = os.path.join(dir, 'data/test.csv')

criterion = nn.functional.nll_loss 
params = settings["params"]
trans_conf = settings["transformer_conf"]
emotion_mapping = settings["emotion_mapping"]

# Audio models
num_labels = 8
model_name_or_path = "facebook/wav2vec2-large-xlsr-53"
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(emotion_mapping)},
    id2label={i: label for i, label in enumerate(emotion_mapping)},
    finetuning_task="wav2vec2_clf"
model_audio = AudioNN(num_classes=8, d_model=512, device=device, dim_feedforward=2048, kernel_1 = 3, kernel_2 = 3, stride_1 = 2, stride_2 = 2).to(device)
model_audio_pretrained = model = MyVGG(device = device)
model_audio_pretrained_wav2vec = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
model_audio_pretrained_wav2vec.freeze_feature_extractor()


# TODO: CHOOSE BY INPUT ARGUMENT IF AUDIO/VIDEO, AND WHICH MODEL
model = model_audio

if torch.cuda.is_available():
        model.cuda()
        

my_model, yreal_train, ypred_train, ypred_test, yreal_test  = train_model_audio(params, model) 

my_model, yreal_train, ypred_train, ypred_test, yreal_test  = train_model_video(params, 
                                                                          trans_conf,
                                                                          device, 
                                                                          criterion, 
                                                                          video_path = video_path, 
                                                                          csv_file_train=csv_file_train,
                                                                          csv_file_val=csv_file_val, 
                                                                          emotion_mapping = emotion_mapping) 


    

