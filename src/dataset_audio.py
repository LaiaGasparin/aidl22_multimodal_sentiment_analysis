from torch.utils.data import Dataset
import torch
from utils import transform_toMelSpec_db, plot_spectrogram, transform_to_16khz
from torchvision.io import read_video

class AV_Dataset(Dataset):
    def __init__(self, video_path, csv_file, emotion_mapping, transform=None, mode="video"):
        super().__init__()
        self.root_dir = video_path              # directory where the videos are stored
        self.vid_files = pd.read_csv(csv_file)  # csv_file with videos info          
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.vid_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vid_name = self.vid_files.iloc[idx, 1] # idx is the instance we want and 1 is the column id
        emotion = self.vid_files.iloc[idx, 0]

        num_emo = emotion_mapping[emotion]
        img_frames, audio_frames, metadata = read_video(str(vid_name)) # (N, H, W, C)
        # stereo to mono
        #print(audio_frames.shape)
        audio_frames = audio_frames.float().mean(0, keepdim = True)
      
        if self.mode=="video":
            img_frames = img_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W) / N: numero de frames, C: numero de channels
            img_batch = torch.stack([img_frames[i] for i in range(0, len(img_frames))])

            return img_batch, num_emo
        
        else: 
            if self.transform == transform_toMelSpec_db:
                audio_batch = torch.stack([self.transform(audio_frames[i]) for i in range(0, len(audio_frames))]) 
            elif self.transform == transform_to_16khz:
                audio_batch = torch.stack([self.transform(audio_frames[i], 48000, 16000) for i in range(0, len(audio_frames))]) 

        Debug = False
        
        if Debug:
            print(audio_batch.shape)
            if self.transform == transform_toMelSpec_db:
                plot_spectrogram(audio_batch[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
        
        if self.transform == transform_toMelSpec_db:
            audio_batch = audio_batch.squeeze(0).transpose(0,1)
        
        elif self.transform == transform_to_16khz:
            audio_batch = audio_batch.squeeze(1).transpose(0,1) 
            #audio_batch = audio_batch.transpose(0,1)

        if Debug:
            print(audio_batch.shape)
            if self.transform == transform_toMelSpec_db:
                print(max(audio_batch[1]))
                print(min(audio_batch[1]))
            else: 
                print(max(audio_batch[0]))
                print(min(audio_batch[0]))
        return audio_batch, num_emo
