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
import pdb
import tqdm
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
import pandas as pd

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
minibatch_size = 256
# data_size = 1000

num_blocks = 100
max_block_width = 1000
max_block_height = 100

subsample_stride = 1

o_learning_rate = 1e-4  # 2e-4
g_learning_rate = 1e-5
c_learning_rate = 5e-4
# optim_betas = (0.9, 0.999)
num_epochs = 2000
burn_in = 40
print_interval = 1
image_interval = 16

# anchor_only=False
# alpha = 0  # 1e-1  # penalty for minibatch deviation from training data marginal distributions over features
# beta = 1e-1  # hyperparameter for importance of Information loss

suffix = ''

# gpu-mode
gpu_mode = True
    
### My data not theirs.
# mydata = torch.Tensor(np.zeros((data_size,g_output_size)))
# for i in range(num_blocks):
#     xlb =  np.random.randint(mydata.shape[0])
#     xub = np.minimum(xlb + 1 + np.random.randint(max_block_height), mydata.shape[0])
#     ylb =  np.random.randint(mydata.shape[1])
#     yub = np.minimum(ylb + 1 + np.random.randint(max_block_width), mydata.shape[1])
    
#     mydata[xlb:xub,ylb:yub] = torch.Tensor(np.random.poisson(np.random.randint(10), size=(xub-xlb, yub-ylb)))
# mydata = mydata.clone()

mydatadf = pd.read_csv('medicare/PartD_Prescriber_PUF_NPI_Drug_15_total_claims_wide.csv')
mydata = Variable(torch.log(torch.FloatTensor(mydatadf.drop(['npi'], axis=1).as_matrix()+1)))

desired_centroids = 20

### Synthetic data
# mydatasize = torch.Size((1000, 1000))
# centroidsize = torch.Size((desired_centroids, mydatasize[1]))
# centroids = F.normalize(torch.FloatTensor(centroidsize).normal_(),2,1)
# mydata = torch.cat([torch.FloatTensor(torch.Size((int(mydatasize[0]/centroidsize[0]),
#                                                   mydatasize[1]))).normal_(std=1/100) +
#                     c for c in centroids])
# mydata = mydata * torch.FloatTensor(torch.Size([mydatasize[0]])).random_(1,10000).unsqueeze(1)
# mydata = mydata / torch.min(mydata.norm(2,1),torch.ones_like(mydata[:,0])).unsqueeze(1)
# true_assignments = np.repeat(np.arange(centroidsize[0]),int(mydatasize[0]/centroidsize[0]))
# mydata = F.normalize(Variable(mydata),2,0)
# # mydata = Variable(mydata)
# # mydata = Variable((1-2*(mydata > 0).float())*torch.log(1+torch.abs(mydata)))
# mydatadf = pd.DataFrame(mydata.data.numpy())
# mydatadf['npi'] = true_assignments

# mydata = mydata.t() # comment out if you want to cluster over physicians instead.

# clustercsv = 'medicare/small_wide.csv'
# mypd = pd.read_csv(clustercsv)
# mypd = mypd.drop(columns='npi')
# mydata = Variable(torch.log(torch.FloatTensor(mypd.as_matrix())+1))

data_size = mydata.size()[0]
g_input_size = 1024  # noise input size
hidden_size = 128  # latent space size
g_output_size = hidden_size
side_channel_size = 1
c_input_size = hidden_size - side_channel_size
c_hidden_size = 128
c_output_size = int(desired_centroids*2)  # # clusters
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


mynoise = Variable(gi_sampler(data_size, g_input_size))
# mynoise = mydata  # autoencoder
g_input_size = mynoise.shape[1]

# d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
Gen = Generator(input_size=g_input_size, hidden_size=g_output_size, output_size=g_output_size, hd=hd)
Out = Outputter(input_size=o_input_size, hidden_size=o_hidden_size, output_size=o_output_size)
Clu = Clusterer(input_size=c_input_size, hidden_size=c_hidden_size,output_size=c_output_size)
Enf = Enforcer(nn.MSELoss(), batch_cosine)

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
                         lr=o_learning_rate)  # , weight_decay=1e-3)
c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters()), lr=c_learning_rate, weight_decay=1e-10)
# c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters(), Out.parameters()), lr=c_learning_rate, weight_decay=1e-10)
# i_optimizer = optim.RMSprop(itertools.chain(G.parameters(), D.parameters()), lr=i_learning_rate)

alpha = 1e-6

# if gpu_mode:
#     mynoise = mynoise.cuda()
#     mydata = mydata.cuda()

pr_g_update = 1
g_lambda = 1e-4
g_o_ratio = 1e-1
pr_c_update = 1
c_only = False
c_lambda = 1e-0
c_l2_lambda = 1e-4
c_e_lambda = 1e-4
o_lambda = 1e-1
e_lambda = 1e-0
s_lambda = 1e-4

# num_epochs=500    
for epoch in range(num_epochs):
    epoch_losses = Variable(torch.zeros(5))
    if gpu_mode:
        epoch_losses = epoch_losses.cuda()

    gcounter, ccounter = 0, 0
    g_epoch_indices = torch.LongTensor(np.random.choice(data_size, size=data_size, replace=False))
    for i0, i1 in arangeIntervals(data_size, minibatch_size):
        g_minibatch_epoch_indices = \
            g_epoch_indices[i0:i1]
            # Variable(g_epoch_indices[i0:i1]).cuda()
        # torch.LongTensor(np.random.choice(data_size, size=minibatch_size))
    
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

        epoch_losses += torch.cat((g_loss, s_loss, o_alone, e_loss, c_loss))

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
        Clu( Gen( mynoise[i0:i1].cuda() )[:,side_channel_size:] ).\
        cpu().data.numpy(), axis=1)
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(assignments.astype(int)))
print(batch_cosine(mydata[:100]))
print(batch_cosine(Out(Gen(mynoise[:100].cuda()))))

# mydatadf = pd.read_csv(clustercsv)
# pd.DataFrame({'drug': mydatadf.drop('npi',1).columns,
#               'cluster':assignments}).to_csv('180416drgclusters.csv')
npidf = pd.DataFrame({'npi':mydatadf['npi'],
                      'cluster':assignments})
npidf.to_csv('180421clusters40.csv')

# Get hidden:
hiddenVectors = np.zeros((mynoise.shape[0], hidden_size))
for i0, i1 in arangeIntervals(data_size, 100):
    hiddenVectors[i0:i1] = Gen(mynoise[i0:i1].cuda()).cpu().data.numpy()
# kmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(hiddenVectors)
# assignments = kmeans.labels_
# print('ARI (Hidden Kmeans):', adjusted_rand_score(true_assignments, kmeans.labels_))
# print('NMI (Hidden Kmeans):', adjusted_mutual_info_score(true_assignments, kmeans.labels_))
# print('SIL (Hidden Kmeans):', silhouette_score(hiddenVectors, kmeans.labels_, metric='cosine'))
npihiddendf = pd.DataFrame(hiddenVectors)
npihiddendf['npi'] = mydatadf['npi']
npihiddendf.to_csv('180421npihidden40.csv')


# Post process merge clusters
print('Post-process to get k clusters: ', len(np.unique(assignments)), ' -> ', desired_centroids)
mydata = mydata.cuda()
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

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
print('\nARI (NN method):', adjusted_rand_score(true_assignments, assignments))
print('ARI (NN merged):', adjusted_rand_score(true_assignments, merged_assignments))
print('ARI (   Kmeans):', adjusted_rand_score(true_assignments, kmeans.labels_))

print('\nNMI (NN method):', adjusted_mutual_info_score(true_assignments, assignments))
print('NMI (NN merged):', adjusted_mutual_info_score(true_assignments, merged_assignments))
print('NMI (   Kmeans):', adjusted_mutual_info_score(true_assignments, kmeans.labels_))

print('\nSIL (NN method):', silhouette_score(mydata.cpu().data.numpy(), assignments, metric='cosine'))
print('SIL (NN merged):', silhouette_score(mydata.cpu().data.numpy(), merged_assignments, metric='cosine'))
print('SIL (   Kmeans):', silhouette_score(mydata.cpu().data.numpy(), kmeans.labels_, metric='cosine'))

# assignments2 = np.argmax(np.log(Clu(Gen(mynoise.cuda())).cpu().data.numpy()) -
#                          np.log(np.mean(Clu(Gen(mynoise.cuda())).cpu().data.numpy(),0,
#                                         keepdims=True)), axis=1)
# print(np.bincount(assignments2))

# mydatadf = pd.read_csv(clustercsv)
# pd.DataFrame({'drug': mydatadf.drop('npi',1).columns,
#               'cluster':assignments}).to_csv('180416drgclusters.csv')
# npidf = pd.DataFrame({'npi':mydatadf['npi'],
#                       'cluster':assignments})
# npidf.to_csv('180420clusters25.csv')

# Get hidden:
hiddenVectors = np.zeros((mynoise.shape[0], hidden_size))
for i0, i1 in arangeIntervals(data_size, 100):
    hiddenVectors[i0:i1] = Gen(mynoise[i0:i1].cuda()).cpu().data.numpy()
kmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(hiddenVectors)
assignments = kmeans.labels_
print('ARI (Hidden Kmeans):', adjusted_rand_score(true_assignments, kmeans.labels_))
print('NMI (Hidden Kmeans):', adjusted_mutual_info_score(true_assignments, kmeans.labels_))
print('SIL (Hidden Kmeans):', silhouette_score(hiddenVectors, kmeans.labels_, metric='cosine'))

# npihiddendf = pd.DataFrame(hiddenVectors)
# npihiddendf['npi'] = mydatadf['npi']
# npihiddendf.to_csv('180420npihidden25.csv')

# outdf = mydatadf
# outdf['assignment'] = assignments
# ssg = outdf.groupby('assignment').mean()
# cdf = pd.DataFrame({'counts':pd.Series.value_counts(assignments)})
# cdf['assignment'] = cdf.index
# ssg.join(cdf).to_csv('outputs/180417seneca_groups.csv')

# npissaved = pd.read_csv('180415clusters.csv')
# image_w_cluster = mydata.cpu().data.numpy()[:,np.argsort(npissaved.cluster)]
# drugssaved = pd.read_csv('180416drgclusters.csv')
# image_w_cluster = mydata.cpu().data.numpy()[:,np.argsort(drugssaved.cluster)]
# image_w_cluster = mydata.cpu().data.numpy()[:,np.random.permutation(mydata.shape[1])]
# image_ordered = mydata.cpu().data.numpy()[np.argsort(assignments),:]
# plt.imshow(image_ordered, aspect='auto')
# plt.show()

# plt.subplot(121)
# plt.imshow(mydata[:1000].cpu().data.numpy())
# plt.clim(0,20)
# plt.subplot(122)
# plt.imshow(Out(Gen(mynoise[:1000].cuda())).cpu().data.numpy())
# plt.clim(0,20)
# # plt.colorbar()
# plt.show()


print('A few ', np.argmax(softs, axis=1)[:10])
print(np.bincount(np.argmax(softs, axis=1)))
K.eval(K.categorical_crossentropy(K.variable(softs[:5]), K.variable(softs[:5])))
print('h cossim\n', 
      np.round(K.eval((K.dot(K.variable(h[:5]), K.transpose(K.variable(h[:5]))))),2)) 
print('c cossim\n',
      np.round(K.eval((K.dot(K.sqrt(K.variable(softs[:5])), K.transpose(K.sqrt(K.variable(softs[:5])))))),2))
print('see how c is nonnegative which encourages h nonnegative through backprop')
print(K.eval((K.dot(K.variable(h[:5]), K.transpose(K.variable(h[:5]))))) - \
      K.eval((K.dot(K.sqrt(K.variable(softs[:5])), K.transpose(K.sqrt(K.variable(softs[:5])))))))
# How about by the ratio of p(_)/mean(p(_))?
assignments2 = np.argmax(np.log(softs) - np.log(np.mean(softs,0, keepdims=True)), axis=1)
print(np.bincount(assignments2))


#### OLD ####
print('hi')

#  1A: Train D on real
# d_real_data = Variable(d_sampler(d_input_size))

d_real_decision, _ = D(preprocess(d_real_data))
d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size)))  # ones = true
d_real_error.backward()  # compute/store gradients, but don't change params

#  1B: Train D on fake
d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
d_fake_data, _ = G(d_gen_input)
d_fake_data.detach()  # detach to avoid training G on these labels
d_fake_decision, d_fake_last_hidden = D(preprocess(d_fake_data))  # D(preprocess(d_fake_data.t()))
d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size)))  # zeros = fake
d_fake_error.backward(retain_graph=True)
d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

for g_index in range(g_anchor_steps):
    G.zero_grad()
    # 2B: Train G on anchors (random noise that maps to the data distribution)
    
    g_fake_anchor_output, _ = G(myanchors)
    # g_minibatch_indices = np.random.choice(data_size, size=minibatch_size)
    g_minibatch_indices = g_minibatch_epoch_indices[g_index,:]
    g_anchor_loss = MSE(g_fake_anchor_output[g_minibatch_indices,:],
                        Variable(mydata[g_minibatch_indices,:]))
    g_anchor_loss.backward(retain_graph=True)
    g_optimizer.step()  # Only optimizes G's parameters
    
    for g_index in range(g_steps):
        if anchor_only and epoch > 0:
            break
        # 2A: Train G on D's response (but DO NOT train D on these labels)
        
        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data, _ = G(gen_input)
        dg_fake_decision, _ = D(preprocess(g_fake_data))  # D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(minibatch_size)))  # we want to fool, so pretend it's all genuine
        g_loss = g_error + alpha*torch.pow(g_fake_data.mean(0)-Variable(mydata.mean(0)),2).sum()
        g_loss.backward(retain_graph=True)
        g_optimizer.step()


    # I.zero_grad()
    if not anchor_only or epoch == 0:
        i_prediction = I(d_fake_last_hidden)
        i_output = d_gen_input[:,:c_size]
        i_error = torch.pow((i_prediction - i_output),2).mean()
        i_loss = beta * i_error
        i_loss.backward()
        i_optimizer.step()
        I.zero_grad()
        
    if epoch % print_interval == 0:
        print("%s: D: %6.3f/%6.3f;  G: %6.3f; I: %6.3f;  (Real: %s, Fake: %s) " % (epoch,
                                                                                   extract(d_real_error)[0],
                                                                                   extract(d_fake_error)[0],
                                                                                   extract(g_error)[0],
                                                                                   extract(i_error)[0],
                                                                                   stats(extract(d_real_data)),
                                                                                   stats(extract(d_fake_data))))
    if epoch % print_interval == 0:
        print(epoch)
    if epoch % image_interval == 0:
        plt.figure(figsize=(10,8))
        plt.subplot(141)
        plt.imshow(subsample(G(Variable(gi_sampler(data_size, g_input_size)))[0].data.numpy(),subsample_stride=subsample_stride))
        plt.xlabel('Mime, permuted')
        plt.subplot(142)
        plt.imshow(subsample(mydata[np.random.choice(mydata.shape[0], size=mydata.shape[0], replace=True),:], subsample_stride=subsample_stride))
        plt.xlabel('Truth sampled, permuted')
        plt.subplot(143)
        plt.imshow(subsample(g_fake_anchor_output.data.numpy(), subsample_stride=subsample_stride))
        plt.xlabel('Mime, anchored')
        plt.subplot(144)
        plt.imshow(subsample(mydata, subsample_stride=subsample_stride))
        plt.xlabel('Truth')
        plt.colorbar()
        plt.subplots_adjust(wspace=0.00)
        plt.savefig('images/gan_' + suffix + '_' + str(dt.datetime.now()) + '.svg', format="svg"); plt.clf() 

        
plt.imshow(mydata)
plt.colorbar()
plt.savefig('images/gan_' + suffix + '_' + str(dt.datetime.now()) + '_truth.svg', format="svg"); plt.clf() 
plt.imshow(mydata[np.random.choice(mydata.shape[0], size=mydata.shape[0], replace=False),:])
plt.colorbar()
plt.savefig('images/gan_' + suffix + '_' + str(dt.datetime.now()) + '_jumbled_truth.svg', format="svg"); plt.clf() 


### Extract clusters based on noise process from the last layer
g_input = Variable(gi_sampler(data_size, g_input_size))
g_fake_data, g_fake_predata = G(g_input)

db = DBSCAN(min_samples=2,eps=10).fit_predict(g_fake_predata.data.numpy())
# ap = AffinityPropagation().fit(g_fake_predata.data.numpy(), )
ac = AgglomerativeClustering(n_clusters=100).fit(g_fake_predata.data.numpy())
# cluster_centers_indices = ap.cluster_centers_indices_
# labels = ap.labels_

# n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % len(np.unique(db)))
# print("Silhouette Coefficient: %0.3f"
#             % metrics.silhouette_score(g_fake_predata.data.numpy(), labels, metric='sqeuclidean'))

df = pd.DataFrame(g_fake_data.data.numpy())
df = pd.DataFrame(mydata[np.random.choice(mydata.shape[0], size=mydata.shape[0], replace=False),:].numpy())
ac = AgglomerativeClustering(n_clusters=100, linkage='average', affinity='l1').fit(mydata)
df['cluster'] = ac.labels_
df_sorted = df.sort_values('cluster').reset_index(drop=True)
plt.subplot(121)
plt.imshow(df_sorted)
plt.clim(0,20)
plt.subplot(122)
plt.imshow(df)
plt.clim(0,20)
plt.colorbar()
plt.savefig('images/gan_' + str(dt.datetime.now()) + '_jumbled_truth.svg', format="svg"); plt.clf() 


# manual k-means with distance function in output space:
K = 40
k_iterations = 100
anchors_mapped, anchors_predata = G(myanchors)  # used to relocate centers
gen_centers = Variable(gi_sampler(K, g_input_size))
centers_mapped, centers = G(gen_centers)  # K x o_width, K x h

# hard clustering
dists_to_centers = torch.abs(Variable(mydata).unsqueeze(0).expand(K, mydata.size()[0],mydata.size()[1]) -
                             centers_mapped.unsqueeze(1).expand(centers_mapped.size()[0],
                                                                mydata.size()[0],
                                                                centers_mapped.size()[1])).sum(2)  # K x o_rows
assignments = np.argmin(dists_to_centers.data.numpy(),0)  # hard assignments
for k_index in range(k_iterations):
    newcenters = []
    for assignment in np.unique(assignments):
        newcenters.append(anchors_predata[torch.LongTensor(np.where(assignment == assignments)[0]),:].mean(0).unsqueeze(0))
    if K - len(newcenters) > 0:
        # newcenters.append(anchors_predata[np.random.choice(anchors_predata.size()[0], size=K - len(newcenters)),:])  # insert based on data
        newcenters.append(Variable(gi_sampler(K-len(newcenters), g_input_size)))  # insert randomly according to GAN
        # TODO insert according to cluster impurity
    centers = torch.cat(tuple(newcenters),0)
    centers_mapped = G.lastlayer(centers)
    dists_to_centers = torch.abs(Variable(mydata).unsqueeze(0).expand(K, mydata.size()[0],mydata.size()[1]) -
                                 centers_mapped.unsqueeze(1).expand(centers_mapped.size()[0],
                                                                    mydata.size()[0],
                                                                    centers_mapped.size()[1])).sum(2)  # K x o_rows
    assignments = np.argmin(dists_to_centers.data.numpy(),0)  # hard assignments

# soft clustering # not working effectively
# for k_index in range(k_iterations):
#     dists_to_centers = torch.abs(Variable(mydata).unsqueeze(0).expand(K, mydata.size()[0],mydata.size()[1]) -
#                                  centers_mapped.unsqueeze(1).expand(centers_mapped.size()[0],
#                                                                     mydata.size()[0],
#                                                                     centers_mapped.size()[1])).sum(2)  # K x o_rows
#     min_dists = dists_to_centers.min(1)[0]
#     avg_min_dist = min_dists.mean()
#     min_dist_multiples = - torch.pow(dists_to_centers/min_dists.unsqueeze(1).expand(K, dists_to_centers.size()[1]),2) - 1e-2  # add a small epsilon (but average means it should be OK)
#     # min_dist_multiples = - torch.pow(dists_to_centers/avg_min_dist,2) - 1e-2  # add a small epsilon (but average means it should be OK)
#     # if k_index == 0:
#     #     min_dist_multiples = - torch.pow(dists_to_centers/avg_min_dist,2) - 1e-2  # add a small epsilon (but average means it should be OK)
#     # else:
#     #     min_dist_multiples = - torch.pow(dists_to_centers/avg_min_dist,2)*torch.log(1+assignments.sum(1,keepdim=True).expand(K,dists_to_centers.size()[1])) - 1e-2
#     assignments = F.softmax(min_dist_multiples,0)  # K x o_rows
#     centers = assignments.matmul(anchors_predata)
#     centers_mapped = G.lastlayer(centers)
# assignments = assignments.max(0)
    
# Plot the cluster centers, mapped to output space
plt.imshow(subsample(centers_mapped.data.numpy(), subsample_stride=(1,subsample_stride)))
plt.xlabel('K medoids')
plt.colorbar()
plt.savefig('images/gan_' + str(dt.datetime.now()) + '_kmedoids.svg', format="svg"); plt.clf() 
    
# Plot the data but color nonzero by cluster membership


# Plot the true data by cluster membership
image_w_cluster = subsample(torch.cat((mydata, torch.Tensor(assignments).float().unsqueeze(1).expand(-1,int(mydata.size()[0]*10/50))),1),
                                      subsample_stride=(10,subsample_stride))
image_permuted = image_w_cluster[:,np.random.permutation(image_w_cluster.shape[1])]
image_ordered = np.argsort(image_permuted[-1,:])
image_permuted[-int(mydata.size()[0]/50):,:] = (image_permuted[-int(mydata.size()[0]/50):,:] % 2) * 18
plt.imshow(image_permuted[:,image_ordered].numpy(), aspect='auto')
plt.xlabel('Cluster membership')
plt.colorbar()
plt.savefig('images/gan_' + str(dt.datetime.now()) + '.svg', format="svg"); plt.clf() 



core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# plt.imshow(G(Variable(gi_sampler(data_size, g_input_size))).data.numpy())
# plt.colorbar()
# plt.show()
