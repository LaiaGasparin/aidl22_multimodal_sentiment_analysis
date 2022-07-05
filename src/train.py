import wandb
from utils import *
import numpy as np
import torch
import torch.optim as optim
from dataset_video import MyVideo
from model import MyNet
import torchvision.transforms as transforms
import torch.nn as nn
from functools import partial

PROJECT_NAME = "multimodal_sentiment_analysis"
ENTITY = "aidl22_project"

# wandb.init(project=PROJECT_NAME, entity=ENTITY)


def train_single_epoch(model, train_loader, optimizer, device, criterion, useTrans, mode='all'):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more
    # print('Starting to watch train')
    wandb.watch(model, criterion, log="all", log_freq=10)
    log_interval = 5
    model.train()

    accs, losses, ypred, yreal = [], [], [], []
    print('........ Start train loop...')

    for batch_idx, (vid, audio, emo, mask_vid, mask_aud) in enumerate(train_loader):
        print('........ Inside loop...')
        optimizer.zero_grad()
        print('........ Loading data...')

        if mode == 'video':
            vid, emo, mask_vid = vid.to(device), emo.to(device), mask_vid.to(device)
        elif mode == 'audio':
            audio, emo, mask_aud = audio.to(device), emo.to(device), mask_aud.to(device)
        else:
            vid, audio, emo, mask_vid, mask_aud = vid.to(device), audio.to(device), emo.to(device),\
                                                  mask_vid.to(device), mask_aud.to(device)

        print('........  running model ...')
        emo_ = model(vid=vid, audio=audio, useTrans=useTrans, src_key_padding_mask_vid=mask_vid,
                     src_key_padding_mask_aud=mask_aud)

        loss = criterion(emo_, emo)
        loss.backward()

        optimizer.step()
        acc = accuracy(emo, emo_)
        losses.append(loss.item())
        accs.append(acc.item())
        ypred.extend(torch.argmax(emo_, -1).cpu().detach().numpy().tolist())
        yreal.extend(emo.cpu().detach().numpy().tolist())

        if batch_idx % log_interval == 0:
            wandb.log({"train inner loss": np.mean(losses), "train inner acc:": np.mean(accs)})

    return np.mean(losses), np.mean(accs), np.array(ypred), np.array(yreal)


def eval_single_epoch(model, val_loader, device, criterion, useTrans, mode='all'):
    model.eval()
    accs, losses, ypred, yreal = [], [], [], []
    with torch.no_grad():
        for batch_idx, (vid, audio, emo, mask_vid, mask_aud) in enumerate(val_loader):
            if mode == 'video':
                vid = vid.to(torch.float32)
                vid = vid.to(device)
                emo = emo.to(device)
                mask_vid = mask_vid.to(torch.float32)
                mask_vid = torch.Tensor(mask_vid).to(device)

            elif mode == 'audio':
                audio, emo, mask_aud = audio.to(device), emo.to(device), mask_aud.to(device)

            else:
                vid, audio, emo, mask_vid, mask_aud = vid.to(device), audio.to(device), emo.to(device), \
                                                      mask_vid.to(device), mask_aud.to(device)

            emo_ = model(vid=vid, audio=audio, useTrans=useTrans, src_key_padding_mask_vid=mask_vid,
                         src_key_padding_mask_aud=mask_aud)

            loss = criterion(emo_, emo)
            acc = accuracy(emo, emo_)
            losses.append(loss.item())
            accs.append(acc.item())
            ypred.extend(torch.argmax(emo_, -1).cpu().detach().numpy().tolist())
            yreal.extend(emo.cpu().detach().numpy().tolist())

    return np.mean(losses), np.mean(accs), np.array(ypred), np.array(yreal)


def train_model(params, trans_conf, emotion_mapping, video_path, csv_file_train, csv_file_val, mode):

    criterion = nn.functional.nll_loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tell wandb to get started
    with wandb.init(project=PROJECT_NAME, config=params, entity=ENTITY):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        if mode == 'video':
            print(" 2.1 Loading video... ")
            train_transforms = transforms.Compose([transforms.CenterCrop(400), transforms.Resize(size=(112, 112))])
            myVideo_train = MyVideo(video_path=video_path, csv_file=csv_file_train, emotion_mapping=emotion_mapping,
                                    transform_vid=train_transforms, mode=mode)
            myVideo_val = MyVideo(video_path=video_path, csv_file=csv_file_val, emotion_mapping=emotion_mapping,
                                  transform_vid=train_transforms, mode=mode)

            fr, channels, height, weight = myVideo_train[0][0].shape # get C/H/W of a video

            print(" 2.2 Loading video loader... ")
            train_loader = torch.utils.data.DataLoader(myVideo_train, batch_size=params['batch_size'],
                                                       shuffle=params['shuffle'], num_workers=params['num_workers'],
                                                       collate_fn=partial(my_collate, mode=mode))
            val_loader = torch.utils.data.DataLoader(myVideo_val, batch_size=params['batch_size'],
                                                     shuffle=params['shuffle'], num_workers=params['num_workers'],
                                                     collate_fn=partial(my_collate, mode=mode))
            print(" 2.3 Instantiating model... ")
            model = MyNet(num_classes=8, d_model=trans_conf['d_model'], nhead=trans_conf['nhead'],
                          channels=channels, height=height, weight=weight,
                          useTrans=params['useTrans'],
                          dim_feedforward=trans_conf['dim_feedforward'],
                          mode=mode).to(device)

        elif mode == 'audio':
            print(" 2.1 Loading audio... ")
            myAudio_train = MyVideo(video_path=video_path, csv_file=csv_file_train, emotion_mapping=emotion_mapping,
                                    transform_aud=transform_to_16khz, mode=mode)
            myAudio_val = MyVideo(video_path=video_path, csv_file=csv_file_val, emotion_mapping=emotion_mapping,
                                  transform_aud=transform_to_16khz, mode=mode)

            print(" 2.2 Loading video loader... ")
            train_loader = torch.utils.data.DataLoader(myAudio_train, batch_size=params['batch_size'],
                                                       shuffle=params['shuffle'],
                                                       num_workers=params['num_workers'],
                                                       collate_fn=partial(my_collate, mode=mode))
            val_loader = torch.utils.data.DataLoader(myAudio_val, batch_size=params['batch_size'],
                                                     shuffle=params['shuffle'], num_workers=params['num_workers'],
                                                     collate_fn=partial(my_collate, mode=mode))
            print(" 2.3 Instantiating model... ")
            model = MyNet(num_classes=8, mode='audio').to(device)


        else:
            print(" 2.1 Loading combined data... ")
            train_transforms = transforms.Compose([transforms.CenterCrop(400), transforms.Resize(size=(112, 112))])
            myVideo_train = MyVideo(video_path=video_path, csv_file=csv_file_train,
                                    emotion_mapping=emotion_mapping,
                                    transform_vid=train_transforms, transform_aud=transform_to_16khz,
                                    mode=mode)
            myVideo_val = MyVideo(video_path=video_path, csv_file=csv_file_train,
                                  emotion_mapping=emotion_mapping,
                                  transform_vid=train_transforms, transform_aud=transform_to_16khz,
                                  mode=mode)

            print(" 2.2 Loading combined loader... ")
            train_loader = torch.utils.data.DataLoader(myVideo_train, batch_size=params['batch_size'],
                                                       shuffle=params['shuffle'],
                                                       num_workers=params['num_workers'],
                                                       collate_fn=partial(my_collate, mode=mode))
            val_loader = torch.utils.data.DataLoader(myVideo_val, batch_size=params['batch_size'],
                                                     shuffle=params['shuffle'], num_workers=params['num_workers'],
                                                     collate_fn=partial(my_collate, mode=mode))

            print(" 2.3 Instantiating model... ")
            fr, channels, height, weight = myVideo_train[0][0].shape
            model = MyNet(num_classes=8, d_model=trans_conf['d_model'], nhead=trans_conf['nhead'],
                          channels=channels, height=height, weight=weight,
                          useTrans=params['useTrans'],
                          dim_feedforward=trans_conf['dim_feedforward'],
                          mode=mode).to(device)


        optimizer = optim.Adam(model.parameters(), params["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)

        for epoch in range(params["epochs"]):
            print('Training single epoch...')
            loss, acc, ypred_train, yreal_train = train_single_epoch(model=model, train_loader=train_loader, optimizer=optimizer,
                                                                     device=device, criterion=criterion, useTrans=params['useTrans'],mode=mode)

            wandb.log({"train epoch": epoch, "train loss": loss, "train acc:": acc, "current_lr:": scheduler.get_last_lr()[0]})
            wandb.log({"conf_mat_train": wandb.plot.confusion_matrix(preds=ypred_train,y_true=yreal_train,
                                                                     class_names=["neutral", "calm", "happy", "sad",
                                                                                  "angry", "fearful", "disgust",
                                                                                  "surprised"])})
            scheduler.step()

            print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f} current_lr={scheduler.get_last_lr()[0]:.6f}")

            print('Eval single epoch...')
            loss, acc, ypred_test, yreal_test = eval_single_epoch(model=model, val_loader=val_loader, device=device,
                                                                  criterion=criterion, useTrans=params['useTrans'], mode=mode)

            wandb.log({"conf_mat_test": wandb.plot.confusion_matrix(preds=ypred_test,y_true=yreal_test,
                                                                    class_names=["neutral", "calm", "happy", "sad",
                                                                                 "angry", "fearful", "disgust",
                                                                                 "surprised"])})
            wandb.log({"Eval epoch": epoch, "Eval loss": loss, "Eval acc:": acc})

            print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")

    return model, yreal_train, ypred_train, ypred_test, yreal_test