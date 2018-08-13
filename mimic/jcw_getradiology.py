import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import gzip, pickle
from tqdm import tqdm

NUM_WORKERS = 0
step = 1000


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


class MimicDatasetForFiltering(Dataset):
    def __init__(self, directory, step=100):
        self._files = [s for s in os.listdir(directory) if s.startswith('note')]
        self._length = len(self._files)
        self._directory = directory

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        '''returns a batch, dataset is not responsible for getting the index of note'''
        return loadModel(self._directory+self._files[i])

md = MimicDatasetForFiltering('/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/extracts/notes/')
mdloader = md  # DataLoader(md, num_workers=NUM_WORKERS)
# rrows = pd.read_csv('/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/extracts/notes/row_raddescription.csv')
mn = pd.read_csv('/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/NOTEEVENTS.csv', quotechar='"')  # , nrows=100)
mn['row'] = mn.index

# rnotes = rrows.iloc[:,0].values
ri = 0
rlist = [None]*sum(mn.CATEGORY=='Radiology')
skipped = []

for bowi, bow in tqdm(enumerate(mdloader)):
    for singlebow in bow:
        row = list(singlebow.keys())[0]
        if mn.CATEGORY[row] != 'Radiology':
            continue
        rlist[ri] = singlebow
        ri += 1


def getLbubs(yourlist, step):
    lbubs = np.arange(0, len(yourlist), step=step)
    lbub = np.transpose(np.array([lbubs.tolist(),
                                  np.concatenate((lbubs[1:],np.array([len(yourlist)]))).tolist()]))
    return lbub
def saver(lbub):
    temp = [rlist[d] for d in range(lbub[0],lbub[1])]
    saveModel(temp,'/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/mimic3/extracts/notes/','RadiologyDict' + str(lbub[0]))

lbub = getLbubs(rlist, step)
with mp.Pool() as pool:
    pool.map(saver, lbub)
