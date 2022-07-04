from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from torchvision.io import read_video

class MyVideo(Dataset):
    def __init__(self, video_path, csv_file, emotion_mapping, transform=None):
      super().__init__()      
      self.root_dir = video_path              # directory where the videos are stored
      self.vid_files = pd.read_csv(csv_file)  # csv_file with videos info             
      self.transform = transform
      self.EMOTION_MAPPING = emotion_mapping

    def __len__(self):
      return len(self.vid_files)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
            idx = idx.tolist()

      vid_name = os.path.join(self.root_dir, self.vid_files.iloc[idx, 1])
      # print('Nom video: ', vid_name)
      emotion = self.vid_files.iloc[idx, 0]
      # print('Nom emoció: ', emotion)

      MAPPING = {"neutral": 0, "calm": 1, "happy": 2, "sad": 3, "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7}

      num_emo = MAPPING[emotion]
      # print('Nom emoció: ', num_emo)
      frames, audio, metadata = read_video(str(vid_name)) # (N, H, W, C)
      frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W) / N: numero de frames, C: numero de channels

      if self.transform:
        img_batch = torch.stack([self.transform(frames[i]) for i in range(0, len(frames))]) 
      else:
        img_batch = torch.stack([frames[i] for i in range(0, len(frames))])  
    
      return img_batch, num_emo