import wandb
from utils import accuracy, my_collate, transform_toMelSpec_db
import numpy as np
import torch
import torch.optim as optim
from dataset_audio import AV_Dataset
from model_audio import AudioNN

PROJECT_NAME = "temp"
ENTITY = "aidl"


def train_single_epoch_audio(model, train_loader, optimizer, device, criterion):
    
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    model.train()
    
    accs, losses, ypred, yreal = [], [], [], []
    
    for batch_idx, (aud, emo, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        aud, emo, mask = aud.to(device), emo.to(device), mask.to(device)
        emo_ = model(aud, mask)
        emo = emo-1
        
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=emo_.cpu().detach().numpy(), 
                                                        y_true=emo.cpu().detach().numpy(), 
                                                        preds=None, class_names=["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])})
            
        loss = criterion(emo_, emo)
        loss.backward()
        optimizer.step()
        acc = accuracy(emo, emo_)
        losses.append(loss.item())
        accs.append(acc.item())
        ypred.append(emo_.cpu().detach().numpy())
        yreal.append(emo.cpu().detach().numpy())
        
        return np.mean(losses), np.mean(accs), ypred, yreal

def eval_single_epoch_audio(model, val_loader, device, criterion):
    
    accs, losses, ypred, yreal = [], [], [], []
    with torch.no_grad():
        model.eval()
        for batch_idx, (aud, emo, mask) in enumerate(val_loader):
            aud = aud.to(torch.float32)
            aud = aud.to(device)
            emo = emo.to(device)
            mask = mask.to(torch.bool)
            mask = torch.tensor(mask).to(device)
            emo_ = model(aud, mask)
            # print('emo_: ', emo_)
            # print('emo: ', emo)
            emo = emo-1
            
            wandb.log({"conf_mat_ev" : wandb.plot.confusion_matrix(probs=emo_.cpu().detach().numpy(), 
                                                        y_true=emo.cpu().detach().numpy(), 
                                                        preds=None, class_names=["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])})
            
            loss = criterion(emo_, emo)
            acc = accuracy(emo, emo_)
            losses.append(loss.item())
            accs.append(acc.item())
            ypred.append(emo_.cpu().detach().numpy())
            yreal.append(emo.cpu().detach().numpy())
            
        return np.mean(losses), np.mean(accs), ypred, yreal

def train_model_audio(params, model, trans_conf, device, criterion, video_path, csv_file_train, csv_file_val, emotion_mapping):

    with wandb.init(project=PROJECT_NAME, config=params, entity=ENTITY):
        config = wandb.config
    
        myAudio_train= AV_Dataset(video_path=video_path, csv_file=csv_file_train, emotion_mapping = emotion_mapping, transform=transform_toMelSpec_db, mode="audio")
        myAudio_val= AV_Dataset(video_path=video_path, csv_file=csv_file_val,  transform=transform_toMelSpec_db, mode="audio")

        train_loader = torch.utils.data.DataLoader(myAudio_train, batch_size=params['batch_size'], shuffle=params['shuffle'], num_workers=params['num_workers'], collate_fn=my_collate)
        val_loader = torch.utils.data.DataLoader(myAudio_val, batch_size=params['batch_size'], shuffle=params['shuffle'], num_workers=params['num_workers'], collate_fn=my_collate)
        # test_loader = torch.utils.data.DataLoader(myVideo_test, batch_size=params['batch_size'], shuffle=params['shuffle'], num_workers=params['num_workers'], collate_fn=my_collate)


        optimizer = optim.Adam(model.parameters(), params["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)

        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        for epoch in range(params["epochs"]):
            scheduler.step()
            loss, acc, ypred_train, yreal_train = train_single_epoch_audio(model, train_loader, optimizer, device, criterion)
            wandb.log({"train epoch": epoch, "train loss": loss, "train acc:": acc, "current_lr:": scheduler.get_last_lr()[0]})
            scheduler.step()
            print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f} current_lr={scheduler.get_last_lr()[0]:.6f}")
            loss, acc, ypred_test, yreal_test = eval_single_epoch_audio(model, val_loader, device, criterion)
            wandb.log({"Eval epoch": epoch, "Eval loss": loss, "Eval acc:": acc})
            print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    

    return model, train_loss, train_acc, valid_loss, valid_acc