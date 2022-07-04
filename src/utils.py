import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchaudio import transforms as T

def confusion_matrix(preds, labels, batch_size, n_classes):

    preds = torch.argmax(preds, 1)
    conf_matrix = torch.zeros(n_classes, n_classes)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1

    print(conf_matrix)
    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        print('Sensitivity = {}'.format(sensitivity))
        print('Specificity = {}'.format(specificity))
        
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def my_collate(batch):
    data_org = [item[0] for item in batch]
    
    length = [item[0].shape[0] for item in batch] 
    data_final = torch.nn.utils.rnn.pad_sequence(data_org, batch_first=True)
    mask = ~(torch.arange(max(length)).repeat(len(length), 1) < torch.Tensor(length).unsqueeze(1))

    targets = torch.LongTensor([item[1] for item in batch])
    return [data_final, targets, mask]

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().cpu().numpy().mean()
    return acc


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    # im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def transform_toMelSpec_db(waveform_frame, audio_params):
    
    spectrogram = T.MelSpectrogram(sample_rate = audio_params['sample_rate'], n_fft = audio_params['n_fft'],
                                 hop_length = audio_params['hop_length'], n_mels = audio_params['n_mels']) #, normalized=True)
    melspectogram_db_transform = T.AmplitudeToDB(stype= "power", top_db = 80)

    melspec = spectrogram(waveform_frame)
    melspec_db = melspectogram_db_transform(melspec)
  
    #place normalization
    melspec_db = (melspec_db - melspec_db.mean())/melspec_db.std()
    return torch.FloatTensor(melspec_db)

def transform_to_16khz(waveform_frame,sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=torch.float32)
    waveform = resampler(waveform_frame)

    waveform = (waveform - waveform.mean())/waveform.std() #Normalized
    return torch.FloatTensor(waveform)