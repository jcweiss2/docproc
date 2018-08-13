import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle, gzip

def saveModel(named_objects, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with gzip.GzipFile(directory+'/'+filename+'.gz', 'w') as f:
        pickle.dump(named_objects, f)
        print('Save complete: '+directory+'/'+filename+'.gz')
        f.close()
    return

def loadModel(path):
    # jcwList = pickle.load(os.open(path,'rb'))
    # os.close(path)
    with gzip.open(path, 'rb') as f:
        named_objects = pickle.load(f)
        return named_objects


class MIMICDataset(Dataset):
    def __init__(self, tokenfile, labelfile, datadir, dataprefix):
        self.tokenfile, self.labelfile, self.datadir, self.dataprefix = \
            tokenfile, labelfile, datadir, dataprefix
        self._files = [datadir + f for f in os.listdir(datadir) if f.startswith(dataprefix)]
        self.tokens = loadModel(tokenfile)
        self.tokens = dict(zip(list(self.tokens),
                               np.arange(len(self.tokens))))
        self.labels = pd.read_csv(labelfile)
        self.labels['row'] = self.labels.index
        self.labels = self.labels[['row','DESCRIPTION']]
        # self.labels = dict(zip(self.labels.iloc[:,0].values,
        #                        self.labels.iloc[:,1].values))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        sparseData = loadModel(self._files[i])
        labels = [self.labels[self.labels.row==list(s.keys())[0]].iloc[:,1].values[0]
                  for s in sparseData]
        matrix = np.zeros((len(labels), len(self.tokens)))
        for si, s in enumerate(sparseData):
            bow = list(s.values())[0]
            if bow is None:
                continue
            for k,v in bow.items():
                matrix[si, self.tokens[k]] = v
        return torch.log(1 + torch.tensor(matrix, dtype=torch.float32)), labels
        

def mimicDS():
    tokenfile = '/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/extracts/notes/tokenspickled.gz.gz'
    labelfile = '/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/NOTEEVENTS.csv'
    datadir = '/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/extracts/notes/'
    dataprefix = 'RadiologyDict'
    md = MIMICDataset(tokenfile, labelfile, datadir, dataprefix)
    return md

# hellomatrix, hellolabels = md[0]
