import pandas as pd
import os
import random
import sklearn

def generate_train_test_val_split(dir, train_size = 0.8, test_size = 0.2):
    os.chdir(dir)
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
    random.shuffle(files)
    # fer l'slice abans de les emocions x passar-li al stratify!!!!!! / incloure random_stat
    train, test = sklearn.model_selection.train_test_split(files, train_size=train_size, test_size=test_size, random_state=42) # mantenir test constant 
    train, val = sklearn.model_selection.train_test_split(train, train_size=train_size, test_size=test_size, random_state=42)

    # mapping with target 
    #add to settings.json
    emotion = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
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
