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


class MimicDataset(Dataset):
    def __init__(self, directory, step=100):
        self._files = os.listdir(directory)
        self._length = len(self._files)
        self._directory = directory

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        '''returns a batch, dataset is not responsible for getting the index of note'''
        return loadModel(self._directory+self._files[i])

md = MimicDataset('/mim/counts/')
mdloader = md  # DataLoader(md, num_workers=NUM_WORKERS)
rnotes = pd.read_csv('/mim/extracts/RadiologyRows.txt',header=None).values[:,0]
ri, r = 0, rnotes[0]
rlist = [None]*len(rnotes)
skipped = []

for bowi, bow in tqdm(enumerate(mdloader)):
    if len(bow) != step:
        print('Warning: ', bowi, 'step has only', len(bow), 'elements.' )
    lbub = [bowi*step, bowi*step+step]
    if r > lbub[1]:
        continue
    while r < lbub[1] and ri < len(rnotes):
        if r%step >= len(bow):
            print('Skipping the notes:')
            while r < lbub[1] and ri < len(rnotes):
                print(r)
                skipped = skipped + [r]
                ri += 1
                if ri == len(rnotes):
                    continue
                r = rnotes[ri]
            continue
        rlist[ri] = bow[r%step]
        ri += 1
        if ri == len(rnotes):
            continue
        r = rnotes[ri]

# saveModel(rlist, '/mim/extracts', 'RadiologyBagOfWords')
skipped = np.array(skipped)

def getLbubs(yourlist, step):
    lbubs = np.arange(0, len(yourlist), step=step)
    lbub = np.transpose(np.array([lbubs.tolist(),
                                  np.concatenate((lbubs[1:],np.array([len(yourlist)]))).tolist()]))
    return lbub
def saver(lbub):
    temp = [d for d in rlist[lbub[0]:lbub[1]]]
    saveModel(temp,'/mim/extracts','RadiologyDict' + str(lbub[0]))

lbub = getLbubs(rlist, step)
with mp.Pool() as pool:
    pool.map(saver, lbub)

saveModel(skipped,'/mim/extracts','RadiologySkippedMessesWithIndexing')
