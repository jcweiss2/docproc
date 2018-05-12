import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import tqdm
import itertools


class DocProcDrugDataset(Dataset):
    ''' Merged data in parts '''

    def __init__(self, root_dir, prefix, suffix='.txt'):
        self.root_dir = root_dir
        self.prefix = prefix
        self.suffix = suffix
        self.metacsv = pd.read_csv(os.path.join(root_dir,prefix+str('_meta.csv')))

    def __len__(self):
        return len(self.metacsv)

    def __getitem__(self, idx, delimiter=',', skip=3, cuda=True, zero_column_long=False):
        ex_name = os.path.join(self.root_dir,
                               self.prefix + str(self.metacsv.iloc[idx,0]) +
                               self.suffix)
        ex_datum = torch.from_numpy(pd.read_csv(ex_name,header=None).as_matrix())
        if zero_column_long:
            return ex_datum[:,skip]
        # if cuda:
        #     ex_datum = ex_datum.cuda()
        ex_datum = torch.log(1+ex_datum.float())
        # ex_datum = np.genfromtxt(ex_name, delimiter=delimiter)
        return ex_datum[:,skip:]  # current version has [uid, uid, npi, ...data]


def split_csv(data, out_dir, out_prefix, suffix='.txt', batch_size=64):
    # data.insert(loc=0, column=out_prefix, value=np.arange(len(data)))  # First row is an index, not data
    # data[out_prefix].to_csv(os.path.join(out_dir,out_prefix+'_meta.csv'))
    df = pd.DataFrame({out_prefix: np.arange(((len(data)-1) // batch_size)+1)})
    df.to_csv(os.path.join(out_dir,out_prefix+'_meta.csv'))
    for i in np.arange(((len(data)-1) // batch_size)+1):
        lb, ub = i*batch_size, np.minimum((i+1)*batch_size, len(data))
        np.savetxt(os.path.join(out_dir,
                                out_prefix+str(i)+suffix),
                   data.iloc[lb:ub],
                   delimiter=',', fmt='%d')  # integers only
    return


if __name__ == "__main__":

    ### Maker
    # data_name = 'medicare/docprocdrug2.csv'
    # data = pd.read_csv(data_name)  # , nrows=100)
    # split_csv(data, 'medicare/parts_docprocdrug/','dpd')
    
    # ### User
    data_dir = 'medicare/parts_docprocdrug'
    dpdd = DocProcDrugDataset('medicare/parts_docprocdrug',
                              'dpd')
    # ### Iteratively
    # # for i in np.arange(len(dpdd)):
    # #     print(dpdd[i])
    # ### With a data loader
    # dataloader = DataLoader(dpdd, batch_size=128, shuffle=True, num_workers=16)
    # for i, s in enumerate(dataloader):
    #     print(i, s.shape)
    # ### With a data loader mixing
    dataloader = DataLoader(dpdd, batch_size=32, shuffle=True, num_workers=30)
    for i, s in enumerate(dataloader):
        # print(i, s.shape)
        print(i, s.cuda().  # .permute(2,0,1).contiguous().
              view(32*64, -1).float().mean(1)[:10])
        

