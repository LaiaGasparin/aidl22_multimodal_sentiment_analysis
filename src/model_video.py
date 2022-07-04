import torchvision.models as models
import torch.nn as nn
from model import PositionalEncoding


    
class VideoNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(64*11*20, 512)
        self.pe = PositionalEncoding(512, max_len=500)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 4, 2048),
            3
        )
        self.classifier = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x):
        print(x.shape)
        x = self.convs(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.proj(x)
        print(x.shape)
        x = x.unsqueeze(1)
        print(x.shape)
        x = self.pe(x)
        x = self.transformer(x)
        print(x.shape)
        x = x.mean(0)
        print(x.shape)
        x = self.classifier(x)
        x = self.log_softmax(x)
        return x