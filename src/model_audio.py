import torch
import torch.nn as nn
import torchvision.models as models
from model import PositionalEncoding
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig, Wav2Vec2Processor

class AudioNN(nn.Module):
    def __init__(self, num_classes, d_model, device, dim_feedforward=2048, kernel_1 = 3, kernel_2 = 3, stride_1 = 2, stride_2 = 2):
        super().__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv1d(100, 64, kernel_1, stride = stride_1),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 512, kernel_2, stride = stride_2),
            nn.ReLU()
        )
        #self.proj = nn.Linear(64*11*20, 512)
        self.pe = PositionalEncoding(d_model, max_len=500)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, dim_feedforward),
            num_layers=3
        )
        self.classifier = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x, src_key_padding_mask, kernel_1 = 3, kernel_2 = 3, stride_1 = 2, stride_2 = 2):
        # print('input size: ', x.shape)
        bsz, l, ch = x.shape
        x = x.permute(0, 2, 1)
        # print('input permute size: ', x.shape)
        x = self.conv1(x)
        # print('After conv1:', x.shape)
        x = self.conv2(x)
        # print('After conv2:', x.shape)
        # print('src_key_padding_mask:', src_key_padding_mask.shape)      

        #padding mask to length
        
        in_lengths = (~src_key_padding_mask).sum(1)
        # print(in_lengths)
        out_lengths = torch.floor(((in_lengths-kernel_1)/stride_1)+1)
        #print('After conv_1 length:', out_lengths)

        out_lengths_2 = torch.floor(((out_lengths-kernel_2)/stride_2)+1)
        #print('After conv_2 length:', out_lengths_2)

        mask = ~(torch.arange(max(out_lengths_2),device= self.device).repeat(len(out_lengths_2), 1) < torch.tensor(out_lengths_2,  device= self.device).unsqueeze(1))
        # print('Mask shape', mask.shape)

        # l, ch, n = x.shape
        x = x.permute(2, 0, 1)
        # print("Shape x: ", x.shape) 
        x = self.pe(x)
        # print('After pe:', x.shape)
        x = self.transformer(x,src_key_padding_mask=mask)
        # print('After trf:', x.shape)

        x = x.mean(0)
        
        x = self.classifier(x)
        # print('After linear:', x.shape)
        x = self.log_softmax(x)
        # print('After log_softmax:', x.shape)
        return x

class MyVGG(nn.Module):
    def __init__(self, device, num_classes=8):
        super().__init__()
        # Step 0: instantiate the pretrained model.
        self.pretrained_VGG = models.vgg11(pretrained=True)
        self.pretrained_VGG.to(device)   
        
        # This freezes all model parameters but last conv_layer from VGG11 and the classifier layers
        # ct = 0
        # for param in self.pretrained_VGG.features.parameters():
        #     if ct < 18:
        #         param.requires_grad = False
        #     ct += 1

        for param in self.pretrained_VGG.parameters():
              param.requires_grad = False
                
        # Replace the last layer, considering our number of classes.
        in_features = self.pretrained_VGG.classifier[6].in_features
        self.pretrained_VGG.classifier[6] = nn.Linear(in_features, 8)

        # Adapt the input layer to accept only one channel.
        conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pretrained_VGG.features[0] = conv1

          
    def forward(self, x):
        
        #print(x.shape)
        x = x.unsqueeze(0)      
        #print(x.shape)
        x = x.permute(1, 0, 2, 3)
        x = self.pretrained_VGG(x)
        
        return x


