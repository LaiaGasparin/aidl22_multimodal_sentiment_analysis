from torch.utils.data import Dataset
import os
import pandas as pd
from utils import *
from torchvision.io import read_video

class MyVideo(Dataset):
    def __init__(self, video_path, csv_file, emotion_mapping, transform_vid=None, transform_aud=None, mode='all'):
      super().__init__()      
      self.root_dir = video_path              # directory where the videos are stored
      self.vid_files = pd.read_csv(csv_file)  # csv_file with videos info             
      self.transform_vid = transform_vid
      self.transform_aud = transform_aud
      self.EMOTION_MAPPING = emotion_mapping
      self.mode = mode

    def __len__(self):
      return len(self.vid_files)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
            idx = idx.tolist()

      vid_name = os.path.join(self.root_dir, self.vid_files.iloc[idx, 1])
      emotion = self.vid_files.iloc[idx, 0]

      MAPPING = {"neutral": 0, "calm": 1, "happy": 2, "sad": 3, "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7}

      num_emo = MAPPING[emotion]
      img_frames, audio_frames, metadata = read_video(str(vid_name)) # (N, H, W, C)
      img_frames = img_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W) / N: numero de frames, C: numero de channels

      # stereo to mono
      audio_frames = audio_frames.float().mean(0, keepdim=True)

      # if video mode:
      if self.mode == 'video':
        audio_batch = None
        if self.transform_vid:
          img_batch = torch.stack([self.transform_vid(img_frames[i]) for i in range(0, len(img_frames))])
        else:
          img_batch = torch.stack([img_frames[i] for i in range(0, len(img_frames))])
        # return img_batch, num_emo

      # if audio mode:
      elif self.mode == 'audio':
        img_batch = None
        if self.transform_aud == transform_toMelSpec_db:
          audio_batch = torch.stack([self.transform_aud(audio_frames[i]) for i in range(0, len(audio_frames))])
          audio_batch = audio_batch.squeeze(0).transpose(0, 1)

        elif self.transform_aud == transform_to_16khz:
          audio_batch = torch.stack([self.transform_aud(audio_frames[i], 48000, 16000) for i in range(0, len(audio_frames))])
          audio_batch = audio_batch.squeeze(1).transpose(0, 1)

        # return audio_batch, num_emo

      else: # Combined mode
        if self.transform_vid:
          img_batch = torch.stack([self.transform_vid(img_frames[i]) for i in range(0, len(img_frames))])
        else:
          img_batch = torch.stack([img_frames[i] for i in range(0, len(img_frames))])

        if self.transform_aud == transform_toMelSpec_db:
          audio_batch = torch.stack([self.transform_aud(audio_frames[i]) for i in range(0, len(audio_frames))])
          audio_batch = audio_batch.squeeze(0).transpose(0, 1)

        elif self.transform_aud == transform_to_16khz:
          audio_batch = torch.stack([self.transform_aud(audio_frames[i], 48000, 16000) for i in range(0, len(audio_frames))])
          audio_batch = audio_batch.squeeze(1).transpose(0, 1)

      return img_batch, audio_batch, num_emo










