import pandas as pd
import os
import random
import sklearn.model_selection
from itertools import compress

def generate_train_test_val_split(dir, testing, train_size = 0.8, test_size = 0.2):
    os.chdir(dir)
    if testing == True:
        data_dir = dir + '/data2'
        video_path = os.path.join(dir, 'data2')
        csv_file = os.path.join(dir, 'data2/train.csv')
    else:
        data_dir = dir + '/data'
        video_path = os.path.join(dir, 'data')
        csv_file = os.path.join(dir, 'data/train.csv')

    if os.path.exists(data_dir + '/train.csv'):
        os.remove(data_dir + '/train.csv')
    if os.path.exists(data_dir + '/val.csv'): 
        os.remove(data_dir + '/val.csv')
    if os.path.exists(data_dir + '/test.csv'):
        os.remove(data_dir + '/test.csv')

    files = os.listdir(os.path.abspath(data_dir))
    files = list(compress(files, ['mp4' in i for i in files]))

    random.shuffle(files)

    train, test = sklearn.model_selection.train_test_split(files, train_size=0.8, test_size=0.2, random_state=42, stratify=[z[6:8] for z in files])
    train, val = sklearn.model_selection.train_test_split(train, train_size=0.8, test_size=0.2, random_state=42, stratify=[z[6:8] for z in train])

    # mapping with target
    emotion = {"01": "neutral", "02": "calm","03": "happy","04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

    train = pd.DataFrame({'id': train, 'split': 'train'})
    train['aux'] = train['id'].str.slice(6, 8)
    train['label'] = train['aux'].map(emotion)
    train = train[['label', 'id', 'split']]
    train = train[train["id"].str.contains('checkpoints') == False]
    train.to_csv(data_dir + '/train.csv', index=False)

    val = pd.DataFrame({'id': val, 'split': 'val'})
    val['aux'] = val['id'].str.slice(6, 8)
    val['label'] = val['aux'].map(emotion)
    val = val[['label', 'id', 'split']]
    val = val[val["id"].str.contains('checkpoints') == False]
    val.to_csv(data_dir + '/val.csv', index=False) 

    test = pd.DataFrame({'id': test, 'split': 'test'})
    test['aux'] = test['id'].str.slice(6, 8)
    test['label'] = test['aux'].map(emotion)
    test = test[['label', 'id', 'split']]
    test = test[test["id"].str.contains('checkpoints') == False]
    test.to_csv(data_dir + '/test.csv', index=False)    
    
    return train, test, val
