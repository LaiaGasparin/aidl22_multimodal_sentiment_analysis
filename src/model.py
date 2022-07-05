import torchvision.models as models
import torch
import torch.nn as nn
import math
from transformers import AutoProcessor, AutoModelForPreTraining

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
    def __init__(self, num_classes, d_model=512, nhead=2,
                 channels=3, height=112, weight=112,
                 useTrans=True,
                 dim_feedforward=2048,
                 mode='all'):
        super().__init__()

        self.mode=mode

        # Video --------------------------------------------------------------------------------------------------------
        pretrained_model_video = models.resnet18(pretrained=True)
        # Features extractor: CNN part of resnet18 - 8 first children
        features_resnet18 = nn.Sequential(*(list(pretrained_model_video.children())[0:8]))

        # This freezes layers 1-6 in the total 8 layers of Resnet18
        ct = 0
        for param in features_resnet18.parameters():
            ct += 1
            if ct < 9:
                param.requires_grad = False

        self.features = features_resnet18
        self.proj = nn.Linear(
            int(512 * (math.ceil((height / (2 * 2 * 2 * 2 * 2)))) * (math.ceil((weight / (2 * 2 * 2 * 2 * 2))))),
            d_model)
        self.pe = PositionalEncoding(d_model, max_len=500)
        self.transformer_vid = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=3
        )
        self.classifier_vid = nn.Linear(d_model, num_classes)

        # Audio --------------------------------------------------------------------------------------------------------
        self.transformer_aud = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")
        ct = 0
        for param in self.transformer_aud.parameters():
            if ct == 171 or ct == 185:
                param.requires_grad = False
            ct += 1

        self.classifier_aud = nn.Linear(256, num_classes)

        # Common -------------------------------------------------------------------------------------------------------
        #  05/07 aplicar layer norm abans combiner!!!
        self.combiner = nn.Linear(d_model+256, num_classes)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, vid, audio, useTrans, src_key_padding_mask_vid, src_key_padding_mask_aud):
        if self.mode == 'video':
            print('  2.3.1 Video Model FW: ')
            x = vid.to(torch.float32)

            # print("Shape x principi",x.shape)
            bsz, frames, ch, w, h = x.shape
            x = x.view(bsz * frames, ch, w, h)
            # print("Shape x abans pretrained: ", x.shape)
            x = self.features(x)
            # print("Shape x despres pretrained: ", x.shape)

            _, ch, w, h = x.shape
            x = x.view(bsz, frames, ch, w, h)
            # print("Shape x: ", x.shape)
            x = x.view(bsz, frames, ch * w * h)
            # print("Shape abans Linear", x.shape)
            x = self.proj(x)  # per convertir a la dimensio de l'embedding

            if (useTrans):
                x = x.transpose(1, 0)  # el transformer espera les dimensions (T, B, C)
                x = self.pe(x)
                x = self.transformer_vid(x, src_key_padding_mask=src_key_padding_mask_vid)
                x = x.mean(0)

            if (useTrans == False):
                # print('1: ', x.shape)
                x = x.mean(1)
                # bsz, w, h = x.shape
                # x = x.view(bsz*w, h)
                # print('2: ', x.shape)

            x = self.classifier_vid(x)
            x = self.log_softmax(x)
            return x

        elif self.mode == 'audio':
            print('  2.3.1 Audio Model FW: ')
            x = audio.squeeze(-1)
            x = self.transformer_aud(x)
            x = x.projected_states.mean(1)
            x = self.classifier_aud(x)

            x = self.log_softmax(x)

            return x

        else: # run video with transformer + audio + combine
            print('  2.3.1 Combined Model FW: ')
            x = vid.to(torch.float32)

            # print("Shape x principi",x.shape)
            bsz, frames, ch, w, h = x.shape
            x = x.view(bsz * frames, ch, w, h)
            # print("Shape x abans pretrained: ", x.shape)
            x = self.features(x)
            # print("Shape x despres pretrained: ", x.shape)

            _, ch, w, h = x.shape
            x = x.view(bsz, frames, ch, w, h)
            # print("Shape x: ", x.shape)
            x = x.view(bsz, frames, ch * w * h)
            # print("Shape abans Linear", x.shape)
            x = self.proj(x)  # per convertir a la dimensio de l'embedding

            if (useTrans):
                x = x.transpose(1, 0)  # el transformer espera les dimensions (T, B, C)
                x = self.pe(x)
                x = self.transformer_vid(x, src_key_padding_mask=src_key_padding_mask_vid)
                x = x.mean(0)

            if (useTrans == False):
                # print('1: ', x.shape)
                x = x.mean(1)
                # bsz, w, h = x.shape
                # x = x.view(bsz*w, h)
                # print('2: ', x.shape)

            print('(combined) Video shape: ', x.shape)

            xx = audio.squeeze(-1)
            xx = self.transformer_aud(xx)
            xx = xx.projected_states.mean(1)

            print('(combined) Audio shape: ', xx.shape)

            res=torch.cat([x, xx],dim=1)
            print('(combined) Final shape: ', res.shape)

            res = self.combiner(res)
            res = self.log_softmax(res)
            return res








