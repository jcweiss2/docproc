import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import os
import gzip, pickle

# mn = pd.read_csv('../mimic/extracts/miniNOTES.csv', quotechar='"')
mn = pd.read_csv('NOTES.csv', quotechar='"')  # , nrows=100)

print('Tokenizing')
with mp.Pool() as pool:
    docs = pool.map(word_tokenize, mn.TEXT)
# docs = [word_tokenize(m) for m in tqdm(mn.TEXT)]
mn = None

stops = stopwords.words('english')
porter = PorterStemmer()

def stemmed(doc):
    result = []
    for t in doc:
        if t.isalpha() and t not in stops:
            result = result + [porter.stem(t)]
    return result
print('Stemming')
with mp.Pool() as pool:
    docs = pool.map(stemmed, docs)
# docs = [[porter.stem(t) for t in d if t.isalpha() and t not in stops] for d in tqdm(docs)]

def reremovestops(doc):
    result = []
    for t in doc:
        if t not in stops:
            result = result + [t]
    return result
print('Stop word removing')
with mp.Pool() as pool:
    docs = pool.map(reremovestops, docs)
# docs = [[t for t in d if t not in stops] for d in tqdm(docs)]

# [t[:5] for t in tokens]

### Detect all words
print('Making sets')
tokens = set()
with mp.Pool() as pool:
    docsets = pool.map(set, docs)
tokens = set.union(*docsets)
# docsets = [set(d) for d in docs]
# for d in tqdm(docs):
#     tokens = tokens.union(set(d))

counter = Counter(dict(zip(tokens, [0]*len(tokens))))

### Create matrix for bag of words

def saveModel(named_objects, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with gzip.GzipFile(directory+'/'+filename+'.gz', 'w') as f:
        pickle.dump(named_objects, f)
        print('Save complete: '+directory+'/'+filename+'.gz')
        f.close()
    return
# saveModel(docs, '/mim', 'docspickled.gz')
saveModel(tokens, '/mim', 'tokenspickled.gz')

def loadModel(path):
    # jcwList = pickle.load(os.open(path,'rb'))
    # os.close(path)
    with gzip.open(path, 'rb') as f:
        named_objects = pickle.load(f)
        return named_objects

### Note a dataframe for 2 million counts for 200k tokens is perhaps too large. So we save as pickle dicts (sparse)
print('Saving')
# df = pd.DataFrame(0, columns=list(tokens), index=range(len(docs)))
piece = 100
df = pd.DataFrame(0, columns=list(tokens), index=range(piece))
# dfi = pd.DataFrame(columns=list(tokens))
if os.path.isfile('notecounts.csv'):
    os.remove('notecounts.csv')


lbubs = np.arange(0, len(docs), step=piece)
lbub = np.transpose(np.array([lbubs.tolist(),
                              np.concatenate((lbubs[1:],np.array([len(docs)]))).tolist()]))
tokenslist = list(tokens)

def saver(lbub):
    # df = pd.DataFrame(0, columns=tokenslist, index=range(lbub[1]-lbub[0]))
    # for i in np.arange(lbub[0],lbub[1]):
    #     d = docs[i]
        # row = counter.copy()
        # row.update(Counter(d))
        # df.loc[i%piece] = row
    ### Save bag of words matrix
    # df.to_csv('notecounts'+lbub[0]+'.csv.gz', index=False, header=True, mode='w')
    temp = [dict(Counter(d)) for d in docs[lbub[0]:lbub[1]]]
    saveModel(temp,'/mim/counts','notecountsdict' + str(lbub[0]))
with mp.Pool() as pool:
    pool.map(saver, lbub)


# Next, e.g.
hellotokens = loadModel('/mim/tokenspickled.gz.gz')
hello = loadModel('/mim/counts/notecountsdict0.gz')
