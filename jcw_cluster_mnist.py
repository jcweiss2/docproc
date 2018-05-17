# Generative Adversarial Networks (GAN) example in PyTorch.
import torch.multiprocessing as mp
mp.set_start_method('spawn')
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import datetime as dt
import jcw_pywavelets as jpw
import pdb
import tqdm
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
import pandas as pd
import datetime as dt
from jcw_utils import logandwide

# Data params
# data_mean = 4
# data_stddev = 1.25

# Model params
# g_input_size = 32
# Random noise dimension coming into generator, per output vector
# g_hidden_size = 32  # Generator complexity
# g_output_size = 128    # size of generated output vector
# o_input_size = g_output_size   # Minibatch size - cardinality of distributions
# o_hidden_size = 8   # Discriminator complexity
# o_output_size = 1    # Single dimension for 'real' vs. 'fake'
# i_hidden_size = 8
c_size = 8
minibatch_size = 128
# data_size = 1000

num_blocks = 100
max_block_width = 1000
max_block_height = 100

subsample_stride = 1

o_learning_rate = 1e-4  # 2e-4
g_learning_rate = 1e-5
c_learning_rate = 1e-3
# optim_betas = (0.9, 0.999)
num_epochs = 500
burn_in = 0
print_interval = 1
mbi_print_interval = 10000
image_interval = 16

# anchor_only=False
# alpha = 0  # 1e-1  # penalty for minibatch deviation from training data marginal distributions over features
# beta = 1e-1  # hyperparameter for importance of Information loss

suffix = ''

# gpu-mode
gpu_mode = True
device = 'cuda' if gpu_mode else 'cpu'
    
### My data not theirs.
# mydata = torch.Tensor(np.zeros((data_size,g_output_size)))
# for i in range(num_blocks):
#     xlb =  np.random.randint(mydata.shape[0])
#     xub = np.minimum(xlb + 1 + np.random.randint(max_block_height), mydata.shape[0])
#     ylb =  np.random.randint(mydata.shape[1])
#     yub = np.minimum(ylb + 1 + np.random.randint(max_block_width), mydata.shape[1])
    
#     mydata[xlb:xub,ylb:yub] = torch.Tensor(np.random.poisson(np.random.randint(10), size=(xub-xlb, yub-ylb)))
# mydata = mydata.clone()

desired_centroids = 10
noise_sd = 1./100
explode_factor = 10000

### Synthetic data
if 'docproc_random_seed' in globals():
    torch.manual_seed(docproc_random_seed)
else:
    torch.manual_seed(42)


law = logandwide()
# mydatasize = torch.Size((1000, 1000))
# centroidsize = torch.Size((desired_centroids, mydatasize[1]))
# centroids = F.normalize(torch.FloatTensor(centroidsize).normal_(),2,1)
# mydata = torch.cat([torch.FloatTensor(torch.Size((int(mydatasize[0]/centroidsize[0]),
#                                                   mydatasize[1]))).normal_(std=noise_sd) +
#                     c for c in centroids])
# mydata = mydata * torch.FloatTensor(torch.Size([mydatasize[0]])).\
#     random_(1,explode_factor).unsqueeze(1)
# mydata = mydata / torch.min(mydata.norm(2,1),torch.ones_like(mydata[:,0])).unsqueeze(1)
# true_assignments = np.repeat(np.arange(centroidsize[0]),int(mydatasize[0]/centroidsize[0]))
# mydata = F.normalize(Variable(mydata),2,0)
# mydata = Variable(mydata)
# mydata = Variable((1-2*(mydata > 0).float())*torch.log(1+torch.abs(mydata)))
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
mdata = MNIST('mnist/', train=True, download=True, transform=Compose([ToTensor(), law]))
mloader = DataLoader(mdata, batch_size=32, shuffle=True, num_workers=30)

# mydatadf = pd.DataFrame(mydata.data.numpy())
# mydatadf['npi'] = true_assignments

# mydata = mydata.t() # comment out if you want to cluster over physicians instead.

# clustercsv = 'medicare/small_wide.csv'
# mypd = pd.read_csv(clustercsv)
# mypd = mypd.drop(columns='npi')
# mydata = Variable(torch.log(torch.FloatTensor(mypd.as_matrix())+1))

data_size = len(mdata)
g_input_size = 1024  # noise input size
hidden_size = 128  # latent space size
g_output_size = hidden_size
side_channel_size = 1
c_input_size = hidden_size - side_channel_size
c_hidden_size = 128
c_output_size = int(desired_centroids*2)  # # clusters
o_input_size = hidden_size
o_hidden_size = hidden_size
o_output_size = mdata[0][0].size()[1]


# ### Comparisons ###

# from sklearn.cluster import KMeans, MiniBatchKMeans
# import datetime as dt
# mydatanumpy = mydata.cpu().data.numpy()
# kmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(mydatanumpy)
# assignments = kmeans.labels_
# npidf = pd.DataFrame({'npi':mydatadf['npi'],
#                       'cluster':assignments})
# npidf.to_csv('' + str(dt.datetime.now()) + 'clusters_kmeans.csv')
# ac = AgglomerativeClustering(n_clusters=desired_centroids).fit(mydatanumpy)  # too slow
# assignments = ac.labels_
# npidf = pd.DataFrame({'npi':mydatadf['npi'],
#                       'cluster':assignments})
# npidf.to_csv('' + dt.now() + 'clusters_agg.csv')


# ### Uncomment only one of these
(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % (name))


# ##### DATA: Target data and generator input data
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


hd = [w.float() for w in jpw.create_haar_dictionary(10, vectorType=Variable).values()]


def MakePermuteMatrix(ofsize):
    return Variable(torch.eye(ofsize)[torch.LongTensor(np.random.permutation(ofsize)),:])


def SignMatrix(ofsize):
    return Variable(torch.diag((torch.randn(ofsize) > 0.5).long()*2-1).float())


def subsample(matrix, subsample_stride=1, flip_long=True):
    sx = subsample_stride
    sy = subsample_stride
    if type(subsample_stride) == tuple:
        sx = subsample_stride[0]
        sy = subsample_stride[1]
    if(matrix.shape[0] < matrix.shape[1]):
        matrix = matrix.transpose(1, 0)
        temp = sx
        sx = sy
        sy = temp
    return matrix[::sx, ::sy]


# ##### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hd):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.hd = hd[int(np.log2(hidden_size))-1].t()  # value (input) to freq (output) space, to pool
        self.permute_number = 4
        self.permuteTensor = torch.cat(tuple([SignMatrix(hidden_size).matmul(MakePermuteMatrix(hidden_size)).unsqueeze(0) for p in range(self.permute_number)]),0)  # PN x H x H
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.pool = nn.MaxPool1d(3,stride=2)

    def forward(self, x):
        x0 = 0
        x = F.leaky_relu(self.map1(x))
        x = x.matmul(self.hd)  # x.matmul(self.hd)
        xs = x.matmul(self.permuteTensor)  # PN x B x H
        # pdb.set_trace()
        # x = torch.max(x,2)[0]
        x = xs.permute((1,2,0))  # B x H x PN
        x = self.batchnorm2(self.pool(x)).sum(2) + x0
        x = F.leaky_relu(self.map2(x))
        x = self.batchnorm3(self.map3(x))
        return F.leaky_relu(self.map4(F.leaky_relu(x)),1e-1)

    def lastlayer(self, x):
        return F.leaky_relu(self.map4(F.leaky_relu(x)),1e-1)

    
class Clusterer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Clusterer, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
       
    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.softmax(self.map3(x))
        return x

    
class Outputter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Outputter, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.leaky_relu(self.map3(x))
        return x


class Enforcer(nn.Module):
    def __init__(self, loss, similarity):
        super(Enforcer, self).__init__()
        self.loss = loss
        self.similarity = similarity
        
    def forward(self, x, y):
        return self.loss(self.similarity(x), self.similarity(y))
    

# class Info(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, hd):
#         super(Info, self).__init__()
#         # self.hd = hd[int(np.log2(hidden_size))-1]
#         self.map1 = nn.Linear(input_size, hidden_size)
#         self.map2 = nn.Linear(hidden_size, output_size)
#         self.batchnorm1 = nn.BatchNorm1d(hidden_size)
#         # self.permute_number = 4
#         # self.permuteTensor = torch.cat(tuple([MakePermuteMatrix(hidden_size).unsqueeze(0) for p in range(self.permute_number)]),0)  # PN x H x H
#         # self.pool = nn.MaxPool1d(2,stride=2)
        
#     def forward(self, x):
#         x = F.tanh(self.batchnorm1(self.map1(x)))
#         # x = x.matmul(self.hd)
#         # xs = x.matmul(self.permuteTensor).permute((1,2,0))  # PN x B x H
#         # x = self.pool(xs).sum(2)
#         # pdb.set_trace()
#         # xfreq = x.matmul(self.hd)
#         return F.sigmoid(self.map2(x))

    
def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


def batch_cosine(data, normalize=True):
    if normalize:
        # temp = data / (1e-10 + data.matmul(data.t()).sum(1, keepdim=True).pow(0.5))
        temp = F.normalize(data)
    else:
        temp = data
    return temp.matmul(temp.t())


def arangeIntervals(stop, step):
    numbers = np.arange(stop, step=step)
    if np.any(numbers == stop):
        pass
    else:
        numbers = np.concatenate((numbers,[stop]))
    return zip(numbers[:-1], numbers[1:])


# mynoise = Variable(gi_sampler(data_size, g_input_size))
# mynoise = mydata  # autoencoder
# g_input_size = mynoise.shape[1]
g_input_size = mdata[0][0].size()[1]

# d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
Gen = Generator(input_size=g_input_size, hidden_size=g_output_size, output_size=g_output_size, hd=hd)
Out = Outputter(input_size=o_input_size, hidden_size=o_hidden_size, output_size=o_output_size)
Clu = Clusterer(input_size=c_input_size, hidden_size=c_hidden_size,output_size=c_output_size)
Enf = Enforcer(nn.MSELoss(), batch_cosine)

tzero = Variable(torch.zeros(1)).squeeze()
if gpu_mode:
    Gen = Gen.cuda()
    Gen.permuteTensor = Gen.permuteTensor.cuda()
    Gen.hd = Gen.hd.cuda()
    Out = Out.cuda()
    Clu = Clu.cuda()
    tzero = tzero.cuda()
    
# I = Info(input_size=d_hidden_size, hidden_size = d_hidden_size, output_size=c_size, hd=hd)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()
# d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas, weight_decay=1e-6)
g_optimizer = optim.Adam(Gen.parameters(), lr=g_learning_rate)
# o_optimizer = optim.RMSprop(itertools.chain(Out.parameters(),Gen.parameters()),
#                          lr=o_learning_rate)  # , weight_decay=1e-3)
o_optimizer = optim.Adam(itertools.chain(Out.parameters(),Gen.parameters()),
                         lr=o_learning_rate)  # , weight_decay=1e-3)
c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters()), lr=c_learning_rate, weight_decay=1e-10)
# c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters(), Out.parameters()), lr=c_learning_rate, weight_decay=1e-10)
# i_optimizer = optim.RMSprop(itertools.chain(G.parameters(), D.parameters()), lr=i_learning_rate)

alpha = 1e-6

# if gpu_mode:
#     mynoise = mynoise.cuda()
#     mydata = mydata.cuda()

# pr_g_update = 1
pr_g_update = 0  # For autoencoder withour norm or cluster constraints
g_lambda = 1e-4
g_o_ratio = 1e-1
pr_c_update = 1
c_only = False
c_lambda = 1e-0
# c_lambda = 0
c_l2_lambda = 1e-4
# c_l2_lambda = 0
c_e_lambda = 1e-4
# c_e_lambda = 0
o_lambda = 1e-1
e_lambda = 1e-0
s_lambda = 1e-4
# s_lambda = 0


# num_epochs = 1000
# burn_in = 0
for epoch in range(num_epochs):
    epoch_losses = Variable(torch.zeros(5))
    if gpu_mode:
        epoch_losses = epoch_losses.cuda()

    epoch_running_counter = 0
    gcounter, ccounter = 0, 0
    dcounter = 0

    print('Epoch ' + str(epoch))
    # g_epoch_indices = torch.LongTensor(np.random.choice(data_size, size=data_size, replace=False))
    for i0, load_batch_mydata in enumerate(mloader):
        # print('Loaded batch ' + str(i0))
        load_batch_mydata = load_batch_mydata[0]  # throw out the label
        if gpu_mode:
            load_batch_mydata = load_batch_mydata.cuda()
        load_batch_mydata = load_batch_mydata.view(load_batch_mydata.size()[0]*load_batch_mydata.size()[1],-1)
        load_batch_mydata = load_batch_mydata[torch.randperm(load_batch_mydata.size()[0]),:]
        # g_minibatch_epoch_indices = g_minibatch_epoch_indices.cuda()
        # for i0, i1 in arangeIntervals(data_size, minibatch_size):
        # g_minibatch_epoch_indices = \
        #     g_epoch_indices[i0:i1]
        #     # Variable(g_epoch_indices[i0:i1]).cuda()
        # torch.LongTensor(np.random.choice(data_size, size=minibatch_size))
        
        for mbi in np.arange(((load_batch_mydata.size()[0]-1) // minibatch_size)+1):
            mblb, mbub = mbi*minibatch_size, np.minimum((mbi+1)*minibatch_size, load_batch_mydata.size()[0])
            batch_mydata = load_batch_mydata[mblb:mbub,:]
            batch_noise = batch_mydata
            dcounter += batch_mydata.size()[0]
            
            Gen.zero_grad()
            Out.zero_grad()
            Clu.zero_grad()
            
            hidden = Gen(batch_noise)

            if np.random.uniform() < pr_g_update:
                g_norms = torch.norm(hidden, 2, 1)
                g_loss = g_lambda*torch.max(g_norms,1-torch.log(g_norms+1e-10)).mean()
                s_loss = s_lambda*torch.pow(batch_cosine(hidden).mean() - 0.5,2)  # spread out (batch)
                # s_loss = tzero
                # g_loss += s_loss
                # g_loss.backward(retain_graph=True)  # incorporate into o_optimizer instead.
                # g_optimizer.step()
                gcounter += 1
            else:
                g_loss, s_loss = tzero, tzero
                
            if not c_only:
                output = Out(hidden)
                o_loss = o_lambda * MSE(output, batch_mydata)
                o_alone = o_loss
                e_loss = e_lambda * Enf(hidden[:,side_channel_size:], batch_mydata)  # Enforce H and O similarity
                # e_loss = tzero
                o_loss += e_loss
                o_loss += g_loss + s_loss
                o_loss.backward()
                o_optimizer.step()
            else:
                o_loss, e_loss = tzero, tzero

            if np.random.uniform() < pr_c_update:
                chidden = Variable(hidden[:,side_channel_size:].data, requires_grad=False)  # separator
                # chidden = hidden[:,side_channel_size:]
                clusters = Clu(chidden)
                c_loss = c_l2_lambda*clusters.sum(0).pow(2).mean()
                if epoch < burn_in:
                    pass
                else:
                    c_loss += c_lambda * (
                        batch_cosine(torch.sqrt(clusters+1e-10), normalize=False) -
                        F.relu(batch_cosine(chidden))).pow(2).sum(1).mean()
                    # c_loss += c_lambda * (
                    #     batch_cosine(torch.sqrt(clusters+1e-10),
                    #                  normalize=False).pow(batch_cosine(chidden)
                    # c_e_loss = tzero
                c_e_loss = ((clusters*torch.log(clusters+1e-10)).mean() + 0.8/minibatch_size).pow(2)  # Match on desired entropy
                c_loss = c_loss + c_e_lambda*c_e_loss  # + [p for p in Clu.map3.parameters()][0].pow(2).mean()
                # c_loss += o_loss
                c_loss.backward()
                c_optimizer.step()
                ccounter += 1
            else:
                c_loss = tzero

            epoch_losses += torch.stack((g_loss, s_loss, o_alone, e_loss, c_loss))
            epoch_running_counter += batch_mydata.size()[0]
            
    if epoch % print_interval == 0:
        el = epoch_losses.cpu().data.numpy() / epoch_running_counter
        # el *= 1*minibatch_size/data_size*\
            #       np.array([1.*data_size/minibatch_size/np.maximum(gcounter,1),
        #                 1.*data_size/minibatch_size/np.maximum(gcounter,1),
        #                 1,
        #                 1,
        #                 1.*data_size/minibatch_size/ccounter])
        print("%s: [H: %8.6f;  s: %8.6f]; [O: %8.6f; e: %8.6f]; C: %8.6g" %
              (epoch, el[0], el[1], el[2], el[3], el[4]))
            
        
dataloader_fixed = DataLoader(mdata, batch_size=64, shuffle=False, num_workers=30)
Gen = Gen.eval()
assignments = np.zeros(data_size).astype(int)
i0 = 0
for i, mybatch in enumerate(dataloader_fixed):
    mynoise = mybatch[0].cuda().view(-1, mybatch[0].size()[2])
    i1 = i0+mynoise.size()[0]
    assignments[i0:i1] = np.argmax(
        Clu(Gen(mynoise)[:,side_channel_size:]).
        cpu().data.numpy(), axis=1)
    i0 = i1
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(assignments.astype(int)))


truths = np.zeros(data_size)
i0 = 0
for i, mybatch in enumerate(dataloader_fixed):
    nextset = mybatch[1]
    i1 = i0 + len(nextset)
    truths[i0:i1] = nextset.detach().numpy()
    i0 = i1
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(truths.astype(int)))


prefix = 'mnist_' + str(dt.datetime.now()) + \
         '_centroids' + str(desired_centroids) + '_'

# npidf = pd.DataFrame({'npi':mydatadf['npi'],
#                       'cluster':assignments})
# npidf.to_csv('180422clusters40.csv')



print('Post-process to get k clusters: ', len(np.unique(assignments)), ' -> ', desired_centroids)
# mydata = mydata.cuda()
ncs = c_output_size
merged_assignments = assignments.copy().astype(int)
while ncs > desired_centroids and len(np.unique(merged_assignments)) > desired_centroids:
    unique_mas = np.unique(merged_assignments)
    n_mas = len(unique_mas)
    numer, denom = torch.zeros(n_mas, n_mas), torch.zeros(n_mas, n_mas)
    i0 = 0
    for i, mybatch in enumerate(dataloader_fixed):
        mynoise = mybatch[0].to(device).view(-1, mybatch[0].size()[2]).contiguous()
        i1 = i0+mynoise.size()[0]
        clusters = Clu(Gen(mynoise)[:,side_channel_size:])
        cossims = batch_cosine(torch.sqrt(clusters+1e-10), normalize=False).cpu()
        mas = merged_assignments[i0:i1]
        uvals, uidx = np.unique(mas,return_inverse=True)
        um_mat = torch.sparse.FloatTensor(torch.LongTensor([uidx.tolist(),
                                                            np.arange(len(uidx)).tolist()]),
                                          torch.ones(len(uidx))).to_dense().t()
        isin = np.isin(unique_mas, uvals)
        expander = torch.sparse.FloatTensor(torch.LongTensor([np.where(isin == 1)[0].tolist(),
                                                              np.arange(sum(isin)).tolist()]),
                                            torch.ones(len(uvals)),
                                            torch.Size((n_mas, len(uvals)))).to_dense().t()
        
        numerBatch = (cossims-torch.diag(torch.ones(len(mas)))).matmul(um_mat).t().matmul(um_mat)
        numer = numer + numerBatch.matmul(expander).t().matmul(expander)
        denomBatch =       (1-torch.diag(torch.ones(len(mas)))).matmul(um_mat).t().matmul(um_mat)
        denom = denom + denomBatch.matmul(expander).t().matmul(expander)
        i0 = i1
        # pdb.set_trace()
    approx_similarity = numer/(denom + 1e-8) * (1-torch.diag(torch.ones(n_mas)))
    print(n_mas)
    # pdb.set_trace()
    merger = np.unravel_index(approx_similarity.data.cpu().numpy().argmax(),
                              approx_similarity.data.cpu().numpy().shape)
    merged_assignments[np.where(merged_assignments == unique_mas[merger[1]])[0]] = \
        unique_mas[merger[0]]
print(np.bincount(merged_assignments.astype(int)))


mydata = torch.zeros(len(mdata),o_output_size)
i0 = 0
for i, mybatch in enumerate(dataloader_fixed):
    i1 = i0 + len(mybatch[0])
    mydata[i0:i1] = mybatch[0].squeeze()
    i0 = i1

### Get hidden:
hiddenVectors = np.zeros((mydata.shape[0], hidden_size))
for i0, i1 in arangeIntervals(data_size, 100):
    hiddenVectors[i0:i1] = Gen(mydata[i0:i1].cuda()).cpu().data.numpy()
# npihiddendf = pd.DataFrame(hiddenVectors)
# TODO fix
npihiddendf = pd.DataFrame({'truth':truths})
# npihiddendf['truth'] = truths
npihiddendf['cluster'] = assignments
npihiddendf['cluster_merged'] = merged_assignments
npihiddendf.to_csv(prefix + 'hidden.csv')
summ = pd.DataFrame({'Measure':pd.Series(['ARI','NMI'])})
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
summ['Ours'] = [adjusted_rand_score(truths, assignments),
                adjusted_mutual_info_score(truths, assignments)]
summ['Ours merged'] = [adjusted_rand_score(truths, merged_assignments),
                adjusted_mutual_info_score(truths, merged_assignments)]
print(summ)
summ.to_csv(prefix + 'summary.csv')

    
# Get kmeans
from sklearn.cluster import MiniBatchKMeans
mydatanumpy = mydata.cpu().data.numpy()
kmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(mydatanumpy)
assignments_kmeans = kmeans.labels_
npidf = pd.DataFrame({'truth':truths,
                      'cluster':assignments_kmeans})
npidf.to_csv(prefix + 'clusters_kmeans.csv')


# Get hidden kmeans
hiddenkmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(hiddenVectors)
assignments_hidden_kmeans = hiddenkmeans.labels_
npidf = pd.DataFrame({'truth':truths,
                      'cluster':assignments_hidden_kmeans})
npidf.to_csv(prefix + 'clusters_hidden_kmeans.csv')

summ['Kmeans'] = [adjusted_rand_score(truths, assignments_kmeans),
                  adjusted_mutual_info_score(truths, assignments_kmeans)]
summ['Hidden Kmeans'] = [adjusted_rand_score(truths, assignments_hidden_kmeans),
                         adjusted_mutual_info_score(truths, assignments_hidden_kmeans)]
print(summ)
summ.to_csv(prefix + 'summary.csv')

# Get our method
# npidf = pd.DataFrame({'truth':truths,
#                       'cluster':assignments})
# npidf.to_csv(prefix + 'clusters_ours.csv')

# # Get our method merged  # Post process merge clusters
# print('Post-process to get k clusters: ', len(np.unique(assignments)), ' -> ', desired_centroids)
# mydata = mydata.cuda()
# ncs = c_output_size
# merged_assignments = assignments.copy()
# sim_dim = 100
# while ncs > desired_centroids and len(np.unique(merged_assignments)) > desired_centroids:
#     unique_mas = np.unique(merged_assignments)
#     n_mas = len(unique_mas)
#     approx_similarity = Variable(torch.FloatTensor(torch.Size((n_mas,n_mas))))
#     for i in np.arange(n_mas):
#         for j in np.arange(n_mas):
#             i_indices = np.random.choice(np.where(merged_assignments == unique_mas[i])[0], sim_dim)
#             j_indices = np.random.choice(np.where(merged_assignments == unique_mas[j])[0], sim_dim)
#             approx_similarity[i,j] = F.normalize(mydata[i_indices,:],2,1).matmul(
#                 F.normalize(mydata[j_indices,:],2,1).t()).mean()
#             approx_similarity[j,i] = 0
#     merger = np.unravel_index(approx_similarity.cpu().data.numpy().argmax(),approx_similarity.size())
#     merged_assignments[np.where(merged_assignments == unique_mas[merger[1]])[0]] = unique_mas[merger[0]]
#     ncs -= 1
# print(np.bincount(merged_assignments.astype(int)))
# npidf = pd.DataFrame({'truth':mydatadf['npi'],
#                       'cluster':merged_assignments})
# npidf.to_csv(prefix + 'clusters_ours_merged.csv')

# # Original data
# mydatadf.rename(columns={'npi':'truth'}).to_csv(prefix + 'original.csv')


# # Performance summary
# summ = pd.DataFrame({'Measure':pd.Series(['ARI','NMI'])})
# from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
# summ['Ours'] = [adjusted_rand_score(true_assignments, assignments),
#                 adjusted_mutual_info_score(true_assignments, assignments)]
# summ['Ours merged'] = [adjusted_rand_score(true_assignments, merged_assignments),
#                        adjusted_mutual_info_score(true_assignments, merged_assignments)]
# summ['Kmeans'] = [adjusted_rand_score(true_assignments, assignments_kmeans),
#                   adjusted_mutual_info_score(true_assignments, assignments_kmeans)]
# summ['Hidden Kmeans'] = [adjusted_rand_score(true_assignments, assignments_hidden_kmeans),
#                          adjusted_mutual_info_score(true_assignments, assignments_hidden_kmeans)]
# print(summ)
# summ.to_csv(prefix + 'summary.csv')


# # summ = pd.DataFrame({'Measure':pd.Series(['ARI','NMI','Silhouette'])})
# # from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
# # summ['Ours'] = [adjusted_rand_score(true_assignments, assignments),
# #                 adjusted_mutual_info_score(true_assignments, assignments),
# #                 silhouette_score(mydata, assignments, metric='cosine')]
# # summ['Ours merged'] = [adjusted_rand_score(true_assignments, merged_assignments),
# #                        adjusted_mutual_info_score(true_assignments, merged_assignments),
# #                        silhouette_score(mydata, merged_assignments, metric='cosine')]
# # summ['Kmeans'] = [adjusted_rand_score(true_assignments, assignments_kmeans),
# #                   adjusted_mutual_info_score(true_assignments, assignments_kmeans),
# #                   silhouette_score(mydata, assignments_kmeans, metric='cosine')]
# # summ['Hidden Kmeans'] = [adjusted_rand_score(true_assignments, assignments_hidden_kmeans),
# #                          adjusted_mutual_info_score(true_assignments, assignments_hidden_kmeans),
# #                          silhouette_score(mydata, assignments_hidden_kmeans, metric='cosine')]
# # print(summ)
# # summ.to_csv(prefix + 'summary.csv')
