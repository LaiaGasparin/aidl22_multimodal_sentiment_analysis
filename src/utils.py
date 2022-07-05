import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

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