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
minibatch_size = 128
# data_size = 1000

num_blocks = 100
max_block_width = 1000
max_block_height = 100

subsample_stride = 1

o_learning_rate = 1e-3  # 2e-4
g_learning_rate = 1e-5
c_learning_rate = 1e-3
# optim_betas = (0.9, 0.999)
num_epochs = 1000000
print_interval = 4
image_interval = 16

# anchor_only=False
# alpha = 0  # 1e-1  # penalty for minibatch deviation from training data marginal distributions over features
# beta = 1e-1  # hyperparameter for importance of Information loss

suffix = ''

# gpu-mode
gpu_mode = True

if gpu_mode:
    t = Variable(torch.FloatTensor(np.array([0,0.5,1]))).cuda()
    ts = t + t
    
### My data not theirs.
# mydata = torch.Tensor(np.zeros((data_size,g_output_size)))
# for i in range(num_blocks):
#     xlb =  np.random.randint(mydata.shape[0])
#     xub = np.minimum(xlb + 1 + np.random.randint(max_block_height), mydata.shape[0])
#     ylb =  np.random.randint(mydata.shape[1])
#     yub = np.minimum(ylb + 1 + np.random.randint(max_block_width), mydata.shape[1])
    
#     mydata[xlb:xub,ylb:yub] = torch.Tensor(np.random.poisson(np.random.randint(10), size=(xub-xlb, yub-ylb)))
# mydata = mydata.clone()

# mydatadf = pd.read_csv('medicare/PartD_Prescriber_PUF_NPI_Drug_15_total_claims_wide.csv')
# mydata = torch.log(torch.FloatTensor(mydatadf.drop(['npi'], axis=1).as_matrix()+1))

# mydata = mydata.as_matrix()
# mydata = mydata * mydata.shape[0] / np.expand_dims((mydata>0).sum(0),0)  # try tf/idf
# mydata = np.log(1+mydata)
# mydata = mydata.transpose() # comment out if you want to cluster over physicians instead.

clustercsv = 'medicare/small_wide.csv'
mypd = pd.read_csv(clustercsv)
mypd = mypd.drop(columns='npi')
mydata = Variable(torch.log(torch.FloatTensor(mypd.as_matrix())+1))

data_size = mydata.size()[0]
g_input_size = 256  # noise input size
hidden_size = 256  # latent space size
g_output_size = hidden_size
c_input_size = hidden_size
c_output_size = 50  # # clusters
o_input_size = hidden_size
o_output_size = mydata.size()[1]

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
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        # self.hd = hd[int(np.log2(hidden_size))-1]
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
        x = F.elu(self.map1(x))
        # x = self.batchnorm1(x.matmul(self.hd))
        xs = x.matmul(self.permuteTensor)  # PN x B x H
        # pdb.set_trace()
        # x = torch.max(x,2)[0]
        x = xs.permute((1,2,0))
        x = self.batchnorm2(self.pool(x)).sum(2)
        x = F.leaky_relu(self.map2(x))
        x = self.batchnorm3(self.map3(x))
        return F.leaky_relu(self.map4(F.elu(x)),1e-2)

    def lastlayer(self, x):
        return F.leaky_relu(self.map4(F.elu(x)),1e-2)

    
class Clusterer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Clusterer, self).__init__()
        self.map1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = F.softmax(self.map1(x))
        return x

    
class Outputter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Outputter, self).__init__()
        self.map1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        return x
    

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
        temp = nn.functional.normalize(data)
    else:
        temp = data
    return temp.matmul(temp.t())


def arangeIntervals(stop, step):
    numbers = np.arange(stop, step=step)
    if np.any(numbers == stop):
        pass
    else:
        numbers = np.concatenate(numbers,stop)
    return zip(numbers[:-1], numbers[1:])


# d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
Gen = Generator(input_size=g_input_size, hidden_size=g_output_size, output_size=g_output_size)
Out = Outputter(input_size=o_input_size, hidden_size=0, output_size=o_output_size)
Clu = Clusterer(input_size=c_input_size, output_size=c_output_size)

if gpu_mode:
    Gen = Gen.cuda()
    Gen.permuteTensor = Gen.permuteTensor.cuda()
    Out = Out.cuda()
    Clu = Clu.cuda()

# I = Info(input_size=d_hidden_size, hidden_size = d_hidden_size, output_size=c_size, hd=hd)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()
# d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas, weight_decay=1e-6)
g_optimizer = optim.Adam(Gen.parameters(), lr=g_learning_rate)
o_optimizer = optim.RMSprop(itertools.chain(Out.parameters(),Gen.parameters()),
                            lr=o_learning_rate)  # , weight_decay=1e-3)
c_optimizer = optim.Adam(Clu.parameters(), lr=c_learning_rate, weight_decay=1e-6)
# i_optimizer = optim.RMSprop(itertools.chain(G.parameters(), D.parameters()), lr=i_learning_rate)

mynoise = Variable(gi_sampler(data_size, g_input_size))

alpha = 1e-6

    
for epoch in range(num_epochs):
    epoch_losses = Variable(torch.zeros(3))
    if gpu_mode:
        epoch_losses = epoch_losses.cuda()
        
    g_epoch_indices = torch.LongTensor(np.random.choice(data_size, size=data_size))
    for i0, i1 in arangeIntervals(data_size, minibatch_size):
        g_minibatch_epoch_indices = \
            g_epoch_indices[i0:i1]
        # torch.LongTensor(np.random.choice(data_size, size=minibatch_size))
    
        Gen.zero_grad()
        Out.zero_grad()
        Clu.zero_grad()

        # noise_data = mynoise[np.random.choice(data_size, size=minibatch_size),:]
        noise_data = mynoise[g_minibatch_epoch_indices]
        if gpu_mode:
            noise_data = noise_data.cuda()
            mydata = mydata.cuda()
            g_minibatch_epoch_indices = g_minibatch_epoch_indices.cuda()
            
        hidden = Gen(noise_data)
        output = Out(hidden)
        g_norms = torch.norm(hidden, 2, 1)
        g_loss = torch.max(g_norms,1/(g_norms+1e-10)).mean()
        g_loss += 1e-6*torch.pow(batch_cosine(hidden).sum(1)-0.5*minibatch_size,2).sum()  # spread out
        g_loss.backward(retain_graph=True)
        g_optimizer.step()
        
        o_loss = MSE(output, mydata[g_minibatch_epoch_indices])
        o_loss.backward()
        o_optimizer.step()
        
        chidden = Variable(hidden.data, requires_grad=False)
        clusters = Clu(chidden)
        c_loss = MSE(
            batch_cosine(torch.sqrt(clusters), normalize=False),
            batch_cosine(chidden))
        c_loss.backward()
        c_optimizer.step()

        epoch_losses += torch.cat((g_loss, o_loss, c_loss))

    if epoch % print_interval == 0:
        print("%s: H: %6.3f;  O: %6.3f; C: %6.3f" % (epoch,
                                                     epoch_losses.cpu().data.numpy()[0],
                                                     epoch_losses.cpu().data.numpy()[1],
                                                     epoch_losses.cpu().data.numpy()[2]))


assignments = np.argmax(Clu(Gen(mynoise.cuda())).cpu().data.numpy(), axis=1)
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(assignments))

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

mydatadf = pd.read_csv(clustercsv)
npidf = pd.DataFrame({'npi':mydatadf['npi'],
                      'cluster':assignments})
npidf.to_csv('180412clusters.csv')

image_w_cluster = mydata.cpu().data.numpy()[:,np.random.permutation(mydata.shape[1])]
image_ordered = image_w_cluster[np.argsort(assignments),:]
plt.imshow(image_ordered, aspect='auto')
plt.show()



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
