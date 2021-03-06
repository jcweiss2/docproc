# Generative Adversarial Networks (GAN) example in PyTorch.
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
import pdb, os
import tqdm
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn import metrics
import pandas as pd
import datetime as dt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score


np.random.seed(int(1e8+1))
torch.manual_seed(1e8+1)

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
minibatch_size = 1000
# data_size = 1000

num_blocks = 100
max_block_width = 1000
max_block_height = 100

subsample_stride = 1

o_learning_rate = 1e-3  # 2e-4
g_learning_rate = 1e-5
c_learning_rate = 1e-3
# optim_betas = (0.9, 0.999)
num_epochs = 1000
c_only_epochs = 100 # 1000
burn_in = 10
print_interval = 1
image_interval = 16

# anchor_only=False
# alpha = 0  # 1e-1  # penalty for minibatch deviation from training data marginal distributions over features
# beta = 1e-1  # hyperparameter for importance of Information loss

suffix = ''

# gpu-mode
# gpu_mode = True
gpu_mode = torch.cuda.is_available()
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

### Synthetic data
desired_centroids = 25
noise_sd = 1./100
explode_factor = 10000

mydatasize = torch.Size((1000, 1000))
centroidsize = torch.Size((desired_centroids, mydatasize[1]))
centroids = F.normalize(torch.FloatTensor(centroidsize).normal_(),2,1)
mydata = torch.cat([torch.FloatTensor(torch.Size((int(mydatasize[0]/centroidsize[0]),
                                                  mydatasize[1]))).normal_(std=noise_sd) +
                    c for c in centroids])
# mydata = mydata + 0.25
# mydata[torch.exp(-2*torch.abs(mydata)) > torch.rand_like(mydata)] = 0
# mydata = mydata * torch.FloatTensor(torch.Size([mydatasize[0]])).\
#     random_(1,explode_factor).unsqueeze(1)
# mydata = (mydata*5).round()/5. * torch.FloatTensor(torch.Size([mydatasize[0]])).\
#     random_(1,explode_factor).unsqueeze(1)
mydata = mydata / torch.min(mydata.norm(2,1),torch.ones_like(mydata[:,0])).unsqueeze(1)
mydata[torch.isnan(mydata)] = 0
true_assignments = np.repeat(np.arange(centroidsize[0]),int(mydatasize[0]/centroidsize[0]))
mydata = F.normalize(Variable(mydata),2,0)
# mydata = Variable(mydata)
# mydata = Variable((1-2*(mydata > 0).float())*torch.log(1+torch.abs(mydata)))
mydatadf = pd.DataFrame(mydata.data.numpy())
mydatadf['npi'] = true_assignments

# tsne_dims = 3
# tsne_method = 'barnes_hut'  # 'exact'  # 'barnes_hut'
# # tsne_metric = 'euclidean'  # 'cosine'
# tsne_metric = 'cosine'
# print('Doing TSNE')
# X_embedded = TSNE(verbose=2, n_components=tsne_dims, n_iter=1000, metric=tsne_metric, method=tsne_method).\
#              fit_transform(mydata.cpu().detach().numpy())
# print('Doing Kmeans')
# km = KMeans(n_clusters=desired_centroids).fit(X_embedded)
# assignments = km.predict(X_embedded)
# print(np.unique(assignments, return_counts=True)[1])
# print('TSNE(' + str(tsne_dims) + ')-->Kmeans(' + str(desired_centroids) + '):', adjusted_rand_score(true_assignments, assignments))
# quit('Done')

# mydata = mydata.t() # comment out if you want to cluster over physicians instead.

# clustercsv = 'medicare/small_wide.csv'
# mypd = pd.read_csv(clustercsv)
# mypd = mypd.drop(columns='npi')
# mydata = Variable(torch.log(torch.FloatTensor(mypd.as_matrix())+1))


# # SENECA
# desired_centroids = 10
# mydatadf = pd.read_csv('/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/seneca/extracts/imputedTrain.csv')
# mydata2df = pd.read_csv('/media/jweiss2/c670a65a-dc35-4970-94f7-071e8b478104/seneca/extracts/imputedTest.csv')
# mydatadf = mydatadf.append(mydata2df)
# mydata = pd.get_dummies(mydatadf.drop('ids',1).drop('hosp',1).drop('year',1).
#                         drop('vp_d',1).drop('mv_d',1).drop('icu',1).drop('dead',1).drop('ids1',1))
# transform_record = {}
# for column in mydata.columns:
#     if min(mydata[column]) < 0 and max(mydata[column]) > 0:
#         continue
#     elif min(mydata[column]) < 0:  # negative values only
#         values = -mydata[column].values
#         transform_negate = True
#     else:
#         values = mydata[column].values
#         transform_negate = False
#     if isinstance(values[0], np.bool_):
#         values = values.astype(np.float_)

#     mv = min(values)
#     values = values - mv
#     if np.std(values)/np.mean(values) > 2.5 * np.std(np.log1p(values))/np.mean(np.log1p(values)):
#         print(column, ' is log transformed', np.round(np.std(values)/np.mean(values) / (2 * np.std(np.log1p(values))/np.mean(np.log1p(values))), 4))
#         mydata[column] = values
#         transform_record[column] = '-log, shift ' + str(mv) if transform_negate else 'log, shift ' + str(mv)
#     else:
#         pass

# mydata = Variable(torch.nn.functional.normalize(
#     torch.from_numpy(mydata.as_matrix().astype(float)).float(),
#     2,0))
    

# # MM@MFLD
# desired_centroids = 10
# mydatadf = pd.read_csv('~/workspace/marshfield/extracts/MM/cluster/C90_or_MGUS_selectedmeds_wide.csv')
# mydatadf = mydatadf.drop('STUDY_ID', 1)
# mydata = torch.log(1 + torch.from_numpy(mydatadf.as_matrix()).float())
# mydata = mydata.t()



# mydata = Variable(torch.nn.functional.normalize(
#     torch.log(1.+torch.from_numpy(mydata.as_matrix().astype(float)).float()),
#     2,0))


data_size = mydata.size()[0]
g_input_size = 1024  # noise input size
hidden_size = 128  # latent space size
g_output_size = hidden_size
side_channel_size = 1
c_input_size = hidden_size - side_channel_size
c_hidden_size = 128
# c_output_size = int(desired_centroids)  # # clusters
c_output_size = int(min(desired_centroids*2, desired_centroids+50))  # # clusters
o_input_size = hidden_size
o_hidden_size = hidden_size
o_output_size = mydata.size()[1]


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
    def __init__(self, input_size, hidden_size, output_size, injectivity_by_positivity=False):
        ''' Can get injectivity by strictly monotonicly increasing activations and positive definite matrix, and
        a positive-element matrix is pd.
        '''
        super(Outputter, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.ibp = injectivity_by_positivity

    def enforce_pd(self):
        if self.ibp:
            self.map1.weight.data.clamp_(min=1e-10)
            self.map2.weight.data.clamp_(min=1e-10)
            self.map3.weight.data.clamp_(min=1e-10)
        else:
            print('Error: only call enforce_pd when injectivity_by_positivity is activated')

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.leaky_relu(self.map3(x))
        return x


class Enforcer(nn.Module):
    def __init__(self, loss, similarity, similarity2=None):
        super(Enforcer, self).__init__()
        self.loss = loss
        self.similarity = similarity
        self.similarity2 = similarity2  # for t-SNE where similarities are normal- and t-
        
    def forward(self, x, y):
        if self.similarity2 is None:
            return self.loss(self.similarity(x), self.similarity(y))
        else:
            sim1 = self.similarity(x)
            sim2 = self.similarity2(y)
            # pdb.set_trace()
            return self.loss(sim1, sim2)
    

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


def batch_2norm(x):
    ''' Compute 2-norms of rows of x '''
    return (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(2)

def tsne_functional(sigma2, dist='normal'):
    '''
    Traditionally sigma2 is determined by preprocessing to find perplexity.
    '''
    def tsne_similarity(data):
        '''
        Input: batch of hidden representations. Computes the Gaussian similarity for p_{i|j} and then averages p_{i|j} and p_{j|i}.
        Returns: the log p_{ij} values
        Note the t-sne formulation calculates p_{ij} up front, but this is not feasible when the number of datapoints is too large.
        We approximate it batchwise instead
        '''
        # pdb.set_trace()
        if dist == 'normal':
            numer = torch.exp(-batch_2norm(data)/2/sigma2) - torch.eye(data.shape[0]).to(data.device.type)
        elif dist == 't':
            numer = torch.pow(1 + batch_2norm(data), -1) - torch.eye(data.shape[0]).to(data.device.type)
        elif dist == 'cos':
            # pdb.set_trace()
            numer = torch.clamp(batch_cosine(data),min=0) - torch.eye(data.shape[0]).to(data.device.type)
        else:
            print('Error in tsne_functional: error', dist, 'is not recognized')
        numer = (1-1e-16)*numer + 1e-16/(numer.shape[0]-1)
        denom = numer.sum(1, keepdim=True)
        numer = numer + torch.eye(data.shape[0]).to(data.device.type)  # avoid negative infinities; diagonal is ignored in tsne_kl so long as not inf or nan
        return torch.log(0.5 * (numer/denom + numer/denom.t()))
    return tsne_similarity


def tsne_kl(y, x):
    ''' x and y are in log probability space; this ignores the diagonal. Note y (q) is *first* argument '''
    if torch.isnan(torch.exp(x)).any():
        pdb.set_trace()
        x = x.clamp(max=10)
    return ((torch.ones(x.shape[0]).to(x.device.type) - torch.eye(x.shape[0]).to(x.device.type)) * (torch.exp(x) * (x - y))).sum()


def arangeIntervals(stop, step):
    numbers = np.arange(stop, step=step)
    if np.any(numbers == stop):
        pass
    else:
        numbers = np.concatenate((numbers,[stop]))
    return zip(numbers[:-1], numbers[1:])


# mynoise = Variable(gi_sampler(data_size, g_input_size))
mynoise = mydata  # autoencoder
g_input_size = mynoise.shape[1]

# outputter_enforce_pd = False
outputter_enforce_pd = True  # if you want the outputter to be injective up to rank |H|
outputter_enforce_pd_str = '' if not outputter_enforce_pd else '_pdon'

# d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
Gen = Generator(input_size=g_input_size, hidden_size=g_output_size, output_size=g_output_size, hd=hd)
Out = Outputter(input_size=o_input_size, hidden_size=o_hidden_size, output_size=o_output_size, injectivity_by_positivity=outputter_enforce_pd)
Clu = Clusterer(input_size=c_input_size, hidden_size=c_hidden_size,output_size=c_output_size)

# using_tsne = False
# Enf = Enforcer(nn.MSELoss(), batch_cosine)  # cosine sim
using_tsne = True
tsne_sigma2 = 0.01
# Enf = Enforcer(tsne_kl, tsne_functional(np.nan, 't'), tsne_functional(tsne_sigma2, 'normal'))  # t-SNE objective
Enf = Enforcer(tsne_kl, tsne_functional(np.nan, 't'), tsne_functional(np.nan, 'cos'))  # t-SNE objective  # cossim

using_tsne_str = '' if using_tsne is False else '_tsne' + str(tsne_sigma2)

tzero = Variable(torch.zeros(1))
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
                         lr=o_learning_rate, weight_decay=1e-10)  # , weight_decay=1e-3)
# c_optimizer = optim.Adam(itertools.chain(Clu.parameters()), lr=c_learning_rate, weight_decay=1e-10)
c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters()), lr=c_learning_rate, weight_decay=1e-10)
c_finish_optimizer = optim.Adam(itertools.chain(Clu.parameters()), lr=1e-3, weight_decay=1e-10)
# c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters(), Out.parameters()), lr=c_learning_rate, weight_decay=1e-10)
# i_optimizer = optim.RMSprop(itertools.chain(G.parameters(), D.parameters()), lr=i_learning_rate)

# alpha = 1e-6

# if gpu_mode:
#     mynoise = mynoise.cuda()
#     mydata = mydata.cuda()

pr_g_update = 1
g_lambda = 1e-4  # hidden is on hypersphere
pr_c_update = 1
c_only = False
c_lambda = 1e-0  # clusters sqrt(p) match hidden angle
c_l2_lambda = 0  # 1e-4  # cluster probabilities are l2 regularized
c_e_lambda = 1e-4  # clusters probabilities are entropic
o_lambda = 1e-1  # autoencoder
e_lambda = 1e-0  # H and O similarity
s_lambda = 0  # 1e-4  # spread out
c_only_end_phase = False

if using_tsne:
    # g_lambda = 0
    # c_e_lambda = 0
    s_lambda = 0

# num_epochs = 1000
# burn_in = 0
for epoch in range(num_epochs+c_only_epochs):
    epoch_losses = Variable(torch.zeros(5))
    if gpu_mode:
        epoch_losses = epoch_losses.cuda()

    if c_only_end_phase and epoch > num_epochs:
        c_only = True
        print('conly ', end='')

    gcounter, ccounter = 0, 0
    g_epoch_indices = torch.LongTensor(np.random.choice(data_size, size=data_size, replace=False))
    for i0, i1 in arangeIntervals(data_size, minibatch_size):
        g_minibatch_epoch_indices = \
            g_epoch_indices[i0:i1]
            # Variable(g_epoch_indices[i0:i1]).cuda()
        # torch.LongTensor(np.random.choice(data_size, size=minibatch_size))
        if outputter_enforce_pd:
            Out.enforce_pd()
    
        Gen.zero_grad()
        Out.zero_grad()
        Clu.zero_grad()

        # noise_data = mynoise[np.random.choice(data_size, size=minibatch_size),:]
        batch_noise = mynoise[g_minibatch_epoch_indices]
        batch_mydata = mydata[g_minibatch_epoch_indices]
        if gpu_mode:
            batch_noise = batch_noise.cuda()
            batch_mydata = batch_mydata.cuda()
            g_minibatch_epoch_indices = g_minibatch_epoch_indices.cuda()
            
        hidden = Gen(batch_noise)

        if np.random.uniform() < pr_g_update:
            g_norms = torch.norm(hidden, 2, 1)
            if using_tsne:
                g_loss = g_lambda*g_norms.var()
            else:
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
            e_loss = e_lambda * Enf(hidden[:,side_channel_size:], batch_mydata)  # Enforce O and H similarity
            # e_loss = tzero
            o_loss += e_loss
            o_loss += g_loss + s_loss
            o_loss.backward()
            [p.grad.clamp_(-1,1) for p in Gen.parameters() if p.grad is not None]
            o_optimizer.step()
        else:
            o_loss, e_loss = tzero, tzero[0]

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
            c_e_loss = ((clusters*torch.log(clusters+1e-10)).mean() + 1.0/minibatch_size).pow(2)  # Match on desired entropy
            c_loss = c_loss + c_e_lambda*c_e_loss  # + [p for p in Clu.map3.parameters()][0].pow(2).mean()
            # c_loss += o_loss
            c_loss.backward()
            if c_only:
                c_finish_optimizer.step()
            else:
                c_optimizer.step()
            ccounter += 1
        else:
            c_loss = tzero

        epoch_losses += torch.stack((g_loss, s_loss, o_alone, e_loss, c_loss))

    if epoch % print_interval == 0:
        el = epoch_losses.cpu().data.numpy()
        el *= 1*minibatch_size/data_size*\
              np.array([1.*data_size/minibatch_size/gcounter,
                        1.*data_size/minibatch_size/gcounter,
                        1,
                        1,
                        1.*data_size/minibatch_size/ccounter])
        print("%s: [H: %6.4f;  s: %6.4f]; [O: %6.4f; e: %6.4f]; C: %6.4f" %
              (epoch, el[0], el[1], el[2], el[3], el[4]))

Gen = Gen.eval()
assignments = np.zeros(mynoise.shape[0])
for i0, i1 in arangeIntervals(data_size, 100):
    assignments[i0:i1] = np.argmax(
        Clu( Gen( mynoise[i0:i1].to(device) )[:,side_channel_size:] ).\
        cpu().data.numpy(), axis=1)
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(assignments.astype(int)))

pd.DataFrame({'assignments':assignments}).to_csv('~/workspace/marshfield/extracts/MM/cluster/C90_or_MGUS_' + str(dt.datetime.now()) + '_t.csv', index=False)

logdir = 'simulation_output'
if not os.path.exists(logdir):
    os.makedirs(logdir)
prefix = logdir + '/' + \
         'centroids' + str(desired_centroids) + \
         '_samples' + str(mydatasize[0]) + \
         '_dims' + str(mydatasize[1]) + \
         '_sd' + str(noise_sd) + \
         '_explode' + str(explode_factor) + \
         using_tsne_str + \
         outputter_enforce_pd_str  # + \

# npidf = pd.DataFrame({'npi':mydatadf['npi'],
#                       'cluster':assignments})
# npidf.to_csv('180422clusters40.csv')

# Get hidden:
hiddenVectors = np.zeros((mynoise.shape[0], hidden_size))
for i0, i1 in arangeIntervals(data_size, 100):
    hiddenVectors[i0:i1] = Gen(mynoise[i0:i1].to(device)).cpu().data.numpy()
npihiddendf = pd.DataFrame(hiddenVectors)
npihiddendf['truth'] = mydatadf['npi']
npihiddendf.to_csv(prefix + 'hidden.csv')

# Get kmeans
from sklearn.cluster import MiniBatchKMeans
mydatanumpy = mydata.cpu().data.numpy()
kmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(mydatanumpy)
assignments_kmeans = kmeans.labels_
npidf = pd.DataFrame({'truth':mydatadf['npi'],
                      'cluster':assignments_kmeans})
npidf.to_csv(prefix + 'clusters_kmeans.csv')


# Get hidden kmeans
hiddenkmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(hiddenVectors)
assignments_hidden_kmeans = hiddenkmeans.labels_
npidf = pd.DataFrame({'truth':mydatadf['npi'],
                      'cluster':assignments_hidden_kmeans})
npidf.to_csv(prefix + 'clusters_hidden_kmeans.csv')

# Get our method
npidf = pd.DataFrame({'truth':mydatadf['npi'],
                      'cluster':assignments})
npidf.to_csv(prefix + 'clusters_ours.csv')

# Get our method merged  # Post process merge clusters
print('Post-process to get k clusters: ', len(np.unique(assignments)), ' -> ', desired_centroids)
mydata = mydata.to(device)
ncs = c_output_size
merged_assignments = assignments.copy()
sim_dim = 100
while ncs > desired_centroids and len(np.unique(merged_assignments)) > desired_centroids:
    unique_mas = np.unique(merged_assignments)
    n_mas = len(unique_mas)
    approx_similarity = Variable(torch.FloatTensor(torch.Size((n_mas,n_mas))))
    for i in np.arange(n_mas):
        for j in np.arange(n_mas):
            i_indices = np.random.choice(np.where(merged_assignments == unique_mas[i])[0], sim_dim)
            j_indices = np.random.choice(np.where(merged_assignments == unique_mas[j])[0], sim_dim)
            approx_similarity[i,j] = F.normalize(mydata[i_indices,:],2,1).matmul(
                F.normalize(mydata[j_indices,:],2,1).t()).mean()
            approx_similarity[j,i] = 0
    merger = np.unravel_index(approx_similarity.cpu().data.numpy().argmax(),approx_similarity.size())
    merged_assignments[np.where(merged_assignments == unique_mas[merger[1]])[0]] = unique_mas[merger[0]]
    ncs -= 1
print(np.bincount(merged_assignments.astype(int)))
npidf = pd.DataFrame({'truth':mydatadf['npi'],
                      'cluster':merged_assignments})
npidf.to_csv(prefix + 'clusters_ours_merged.csv')

# Original data
mydatadf.rename(columns={'npi':'truth'}).to_csv(prefix + 'original.csv')


# Performance summary
summ = pd.DataFrame({'Measure':pd.Series(['ARI','NMI'])})
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
summ['Ours'] = [adjusted_rand_score(true_assignments, assignments),
                adjusted_mutual_info_score(true_assignments, assignments)]
summ['Ours merged'] = [adjusted_rand_score(true_assignments, merged_assignments),
                       adjusted_mutual_info_score(true_assignments, merged_assignments)]
summ['Kmeans'] = [adjusted_rand_score(true_assignments, assignments_kmeans),
                  adjusted_mutual_info_score(true_assignments, assignments_kmeans)]
summ['Hidden Kmeans'] = [adjusted_rand_score(true_assignments, assignments_hidden_kmeans),
                         adjusted_mutual_info_score(true_assignments, assignments_hidden_kmeans)]
print(summ)
summ.to_csv(prefix + 'summary.csv')


# summ = pd.DataFrame({'Measure':pd.Series(['ARI','NMI','Silhouette'])})
# from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
# summ['Ours'] = [adjusted_rand_score(true_assignments, assignments),
#                 adjusted_mutual_info_score(true_assignments, assignments),
#                 silhouette_score(mydata, assignments, metric='cosine')]
# summ['Ours merged'] = [adjusted_rand_score(true_assignments, merged_assignments),
#                        adjusted_mutual_info_score(true_assignments, merged_assignments),
#                        silhouette_score(mydata, merged_assignments, metric='cosine')]
# summ['Kmeans'] = [adjusted_rand_score(true_assignments, assignments_kmeans),
#                   adjusted_mutual_info_score(true_assignments, assignments_kmeans),
#                   silhouette_score(mydata, assignments_kmeans, metric='cosine')]
# summ['Hidden Kmeans'] = [adjusted_rand_score(true_assignments, assignments_hidden_kmeans),
#                          adjusted_mutual_info_score(true_assignments, assignments_hidden_kmeans),
#                          silhouette_score(mydata, assignments_hidden_kmeans, metric='cosine')]
# print(summ)
# summ.to_csv(prefix + 'summary.csv')
