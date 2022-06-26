import torchvision.models as models
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MyNet(nn.Module):
    def __init__(self, device, num_classes, d_model, nhead,
                 channels, height, weight, 
                 channel_out1, kernel_size1, pooling1,
                 channel_out2, kernel_size2, pooling2,
                 channel_out3, kernel_size3, pooling3,
                 dim_feedforward=2048
                 ):
        super().__init__()

        pretrained_model = models.resnet18(pretrained=True)
        pretrained_model.to(device)

        # Features extractor: CNN part of resnet18 - 8 first children
        features_resnet18 = nn.Sequential(*(list(pretrained_model.children())[0:8]))

        # This freezes layers 1-6 in the total 8 layers of Resnet18
        ct = 0
        for param in features_resnet18.parameters():
            ct += 1
            if ct < 7:
              param.requires_grad = False

        self.features = features_resnet18
        self.proj = nn.Linear(int(512*(math.ceil((height/(2*2*2*2*2))))*(math.ceil((weight/(2*2*2*2*2))))), d_model)
        self.pe = PositionalEncoding(d_model, max_len=500)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), 
            num_layers=3

        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x, src_key_padding_mask):
        #print('Començant fw model: ')
        x = x.to(torch.float32)
        #print("Shape x principi",x.shape)
        bsz, frames, ch, w, h = x.shape
        x = x.view(bsz*frames, ch, w, h)  
        #print("Shape x abans pretrained: ", x.shape)
        x = self.features(x)
        #print("Shape x despres pretrained: ", x.shape)
        
        _, ch, w, h = x.shape
        x = x.view(bsz, frames, ch, w, h) 
        #print("Shape x: ", x.shape) 
        x = x.view(bsz, frames, ch*w*h)
        #print("Shape abans Linear", x.shape)
        x = self.proj(x) # per convertir a la dimensio de l'embedding
        #print("Shape després Linear", x.shape)
        
        # print('Shape mask: ', src_key_padding_mask.shape)

        x = x.transpose(1, 0) # el transformer espera les dimensions (T, B, C)
        x = self.pe(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # print('1: ', x.shape)
        x = x.mean(0)
        # print('2: ', x.shape)
        
        x = self.classifier(x)
        x = self.log_softmax(x)
        return x