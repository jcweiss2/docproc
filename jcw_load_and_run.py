import torch
import gzip, pickle
from jcw_cluster_classes import *
import pandas as pd
from collections import Iterable
import matplotlib.pyplot as plt
import numpy as np
from jcw_make_parts import DocProcDrugDataset

def loadJcwModel(path, legacy=False):
    # jcwList = pickle.load(os.open(path,'rb'))
    # os.close(path)
    if legacy:
        with gzip.open(path, 'rb') as f:
            vals = []
            # objs = pickle.load(f)
            for _ in range(pickle.load(f)):
                vals.append(pickle.load(f))
        return vals
    else:
        with gzip.open(path, 'rb') as f:
            named_objects = pickle.load(f)
        return named_objects

### Load model
# infile = 'outputs/docprocModel.gz'
infile = 'outputs/docprocModel2018-11-08 17:47:59.623423.gz'
model = loadJcwModel(infile)
Gen, Out, Clu, side_channel_size, c_output_size = \
    [model[s] for s in ['Gen', 'Out', 'Clu', 'side_channel_size', 'c_output_size']]

# # doc-meds
# # mydatadf = pd.read_csv('medicare/docprocwide.csv', nrows=100)
# # mydata = torch.log(torch.FloatTensor(mydatadf.drop(['npi'], axis=1).as_matrix()+1))
# # mydata = mydata.t()
# mydatadf = pd.read_csv('medicare/PartD_Prescriber_PUF_NPI_Drug_15_total_claims_wide.csv', nrows=100)
# mydata = torch.log(torch.FloatTensor(mydatadf.drop(['npi'], axis=1).as_matrix()+1))

# doc-meds-proc
data_dir = 'medicare/parts_docprocdrug'
dpdd = DocProcDrugDataset('medicare/parts_docprocdrug',
                          'dpd')
mdata = dpdd


Gen, Out, Clu = [s.cpu() for s in [Gen, Out, Clu]]
Gen.hd, Gen.permuteTensor = Gen.hd.cpu(), Gen.permuteTensor.cpu()

### Apply model to get cluster probability signature
ex = 9
# datum = mydata[ex].unsqueeze(0)
datum = mdata[ex][0]
testing = Clu(Gen(datum)[:,:-side_channel_size])

if isinstance(c_output_size, Iterable):
    plt.subplot(len(c_output_size),1,1)
    cumc = 0
    for ci, c in enumerate(c_output_size):
        plt.subplot(len(c_output_size),1,ci+1)
        plt.bar(np.arange(1, c+1), testing[0,cumc:(cumc+c)].detach().numpy())
        cumc += c
else:
    plt.bar(np.arange(1,testing.shape[1]+1), testing[0].detach().numpy())

plt.show()
