import itertools
import jcw_pywavelets as jpw
import numpy as np
import os
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

# ##### DATa: Target data and generator input data
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
        # self.dropout = nn.Dropout()
        # self.dropout2 = nn.Dropout()
        self.map1 = nn.Linear(input_size, hidden_size)
        # self.hd = hd[int(np.log2(hidden_size))-1].t()  # value (input) to freq (output) space, to pool
        # self.permute_number = 4
        # self.permuteTensor = torch.cat(tuple([SignMatrix(hidden_size).matmul(MakePermuteMatrix(hidden_size)).unsqueeze(0) for p in range(self.permute_number)]),0)  # PN x H x H
        # self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.pool = nn.MaxPool1d(3,stride=2)

    def forward(self, x):
        # x0 = 0
        # x = F.leaky_relu(self.map1(self.dropout(x)))
        # x = x.matmul(self.hd)  # x.matmul(self.hd)
        # xs = x.matmul(self.permuteTensor)  # PN x B x H
        # # pdb.set_trace()
        # # x = torch.max(x,2)[0]
        # x = xs.permute((1,2,0))  # B x H x PN
        # x = self.batchnorm2(self.pool(x)).sum(2) + x0
        x = F.leaky_relu(self.map1(x))
        x = self.batchnorm2(x)
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
        self.outputs = None
        if isinstance(output_size, np.ndarray):
            self.outputs = tuple(output_size.tolist())
            output_size = int(np.sum(output_size))
        elif isinstance(output_size, (list,)):
            self.outputs = tuple(output_size)
            output_size = int(np.sum(output_size))
        elif isinstance(output_size, torch.Tensor):
            self.outputs = tuple(output_size.detach().cpu().numpy().tolist())
            output_size = int(np.sum(output_size))
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        if self.outputs is not None:
            sms = [F.softmax(o) for o in self.map3(x).split(self.outputs,1)]
            x = torch.cat(sms,1)
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
            return self.loss(self.similarity(x), self.similarity2(y))
    

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

def log_clamped(x, clamp=-32):
    x[x == 0] = np.exp(clamp)
    x = torch.log(x)
    return x

def batch_dot(x):
    return (x.unsqueeze(0) * x.unsqueeze(1)).sum(2)

def batch_equals(x, negation=False):
    ''' If negation is True: returns counts of not equals, which is not total_N_count - N_equals because of possible missingness '''
    if not negation:
        result = (x.unsqueeze(0) == x.unsqueeze(1)).float()
    else:
        result = (x.unsqueeze(0) != x.unsqueeze(1)).float()
    # denom = (1-torch.isnan(result)).sum(2).float()
    numer = result
    numer[torch.isnan(numer)] = 0
    if len(numer.shape) == 3:
        numer = numer.sum(2)
    result = numer  # result = numer/denom
    result[numer==0] = 0
    return result

def batch_2norm(x):
    ''' Compute 2-norms of rows of x '''
    return (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(2)

def batch_1norm(x):
    ''' Compute 1-norms of rows of x '''
    return (x.reshape(-1,x.shape[0],x.shape[1]) - x.reshape(x.shape[0], -1, x.shape[1])).abs().sum(2)
    # return (x.unsqueeze(0) - x.unsqueeze(1)).abs().sum(2)


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
        elif dist == 'l1':
            numer = torch.exp(-batch_1norm(data)/2/sigma2) - torch.eye(data.shape[0]).to(data.device.type)
        numer = (1-1e-16)*numer + 1e-16/(numer.shape[0]-1)
        denom = numer.sum(1, keepdim=True)
        numer = numer + torch.eye(data.shape[0]).to(data.device.type)  # avoid negative infinities; diagonal is ignored in tsne_kl so long as not inf or nan
        return torch.log(0.5 * (numer/denom + numer/denom.t()))
    return tsne_similarity


def tsne_kl(x, y):
    ''' x and y are in log probability space; this ignores the diagonal '''
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


class Expander(nn.Module):
    def __init__(self, hot_size, expand_dim_width, extra=16):
        super(Expander, self).__init__()
        self.weights = nn.Parameter(1e-3 * torch.randn(tuple([hot_size, extra*expand_dim_width])))
        self.register_parameter('weight', self.weights)
        self.map1 = nn.Linear(extra*expand_dim_width, expand_dim_width)
        self.map2 = nn.Linear(expand_dim_width, expand_dim_width)
        self.map3 = nn.Linear(expand_dim_width, expand_dim_width)

    def forward(self, x):
        x = self.weights[x.long(),:]  # use the weights embedding in row given by the assignment
        # x = x.unsqueeze(len(x.shape)) * self.weights + self.bias
        x = self.map1(F.tanh(x))
        x = self.map2(F.leaky_relu(x))
        x = self.map3(F.leaky_relu(x))
        return x

def embedding_prediction(hidden, contribution_size=10):
    return torch.sigmoid(contribution_size*hidden.sum(1))


def run(mydata, side_label_assignments, model_parameters, outcomes, true_assignments,
        keep_data_on_gpu=True):

    mp = model_parameters

    # Define sizes.
    data_size = mydata.shape[0]
    g_input_size = mydata.shape[1]
    o_output_size = mydata.shape[1]

    # Determine if side labels are given and parameters.
    using_side_labels = (side_label_assignments is not None)
    if using_side_labels:
        side_labels = torch.tensor(side_label_assignments)

    # Add extra clustering if side labels are given.
    c_output_size = mp.c_output_size
    if using_side_labels:
        num_side_assignments = min(len(np.unique(true_assignments)[~np.isnan(np.unique(true_assignments))]), 1000)  # slight hack
        c_output_size = np.append(mp.c_output_size, [num_side_assignments])

    # Determine if we should use the GPU or CPU.
    # gpu_mode = True
    gpu_mode = torch.cuda.is_available()
    device = 'cuda' if gpu_mode else 'cpu'
    print("Device is:", device)

    # Initialize noise.
    # mynoise = Variable(gi_sampler(data_size, g_input_size))
    if gpu_mode and keep_data_on_gpu:
        mydata = mydata.cuda()

    mynoise = mydata  # autoencoder

    outputter_enforce_pd = False
    # outputter_enforce_pd = True  # if you want the outputter to be injective up to rank |H|
    outputter_enforce_pd_str = '' if not outputter_enforce_pd else '_pdon'

    # d_sampler = get_distribution_sampler(data_mean, data_stddev)
    # gi_sampler = get_generator_input_sampler()
    Gen = Generator(input_size=g_input_size, hidden_size=mp.g_output_size, output_size=mp.g_output_size, hd=hd)
    Out = Outputter(input_size=mp.o_input_size, hidden_size=mp.o_hidden_size, output_size=o_output_size, injectivity_by_positivity=outputter_enforce_pd)
    Clu = Clusterer(input_size=mp.c_input_size, hidden_size=mp.c_hidden_size,output_size=c_output_size)

    if mp.using_tsne:
        tsne_sigma2 = 1
        Enf = Enforcer(tsne_kl, tsne_functional(np.nan, 't'), tsne_functional(tsne_sigma2, 'normal'))  # t-SNE objective
    else:
        Enf = Enforcer(nn.MSELoss(), batch_cosine)  # cosine sim
    using_tsne_str = '' if mp.using_tsne is False else '_tsne' + str(tsne_sigma2)

    tzero = Variable(torch.zeros(1))
    if gpu_mode:
        Gen = Gen.cuda()
        # Gen.permuteTensor = Gen.permuteTensor.cuda()
        # Gen.hd = Gen.hd.cuda()
        Out = Out.cuda()
        Clu = Clu.cuda()
        tzero = tzero.cuda()

        
    # I = Info(input_size=d_hidden_size, hidden_size = d_hidden_size, output_size=c_size, hd=hd)
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    # d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas, weight_decay=1e-6)
    # g_optimizer = optim.Adam(Gen.parameters(), lr=mp.g_learning_rate)
    # o_optimizer = optim.RMSprop(itertools.chain(Out.parameters(),Gen.parameters()),
    #                          lr=mp.o_learning_rate)  # , weight_decay=1e-3)
    o_optimizer = optim.Adam(itertools.chain(Out.parameters(),Gen.parameters()),
                             lr=mp.o_learning_rate, weight_decay=1e-3)  # , weight_decay=1e-3)
    c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters()), lr=mp.c_learning_rate, weight_decay=1e-10)
    # c_optimizer = optim.Adam(itertools.chain(Gen.parameters(),Clu.parameters(), Out.parameters()), lr=c_learning_rate, weight_decay=1e-10)
    # i_optimizer = optim.RMSprop(itertools.chain(G.parameters(), D.parameters()), lr=i_learning_rate)

    # alpha = 1e-6

    # if gpu_mode:
    #     mynoise = mynoise.cuda()
    #     mydata = mydata.cuda()

    pr_g_update = 1
    g_lambda = 1e-4  # hidden is on hypersphere
    g_l1_lambda = 1e-4  # hidden is on hypersphere
    pr_c_update = 1
    c_only = False
    c_lambda = 1e-0  # clusters p match 
    o_lambda = 1e-1  # autoencoder
    e_lambda = 1e-0  # H and O similarity
    r_lambda = 1e-1

    # num_epochs = 1000
    # burn_in = 0
    for epoch in range(mp.num_epochs):
        epoch_losses = Variable(torch.zeros(5))
        if gpu_mode:
            epoch_losses = epoch_losses.cuda()

        gcounter, ccounter = 0, 0
        g_epoch_indices = torch.LongTensor(np.random.choice(data_size, size=data_size, replace=False))
        for i0, i1 in arangeIntervals(data_size, mp.minibatch_size):
            g_minibatch_epoch_indices = g_epoch_indices[i0:i1]
                # Variable(g_epoch_indices[i0:i1]).cuda()
            # torch.LongTensor(np.random.choice(data_size, size=mp.minibatch_size))
            if outputter_enforce_pd:
                Out.enforce_pd()
        
            Gen.zero_grad()
            Out.zero_grad()
            Clu.zero_grad()

            # noise_data = mynoise[np.random.choice(data_size, size=mp.minibatch_size),:]
            batch_noise = mynoise[g_minibatch_epoch_indices]
            batch_mydata = mydata[g_minibatch_epoch_indices]
            batch_outcomes = outcomes[g_minibatch_epoch_indices]
            if using_side_labels:
                batch_side_labels = side_labels[g_minibatch_epoch_indices]
            if gpu_mode:
                batch_noise = batch_noise.cuda()
                batch_mydata = batch_mydata.cuda()
                g_minibatch_epoch_indices = g_minibatch_epoch_indices.cuda()
                batch_outcomes = batch_outcomes.cuda()
            if using_side_labels and gpu_mode:
                batch_side_labels = batch_side_labels.cuda()

            hidden = Gen(batch_noise)

            if np.random.uniform() < pr_g_update:
                g_norms = torch.norm(hidden, 1, 1)  # on a diamond (L1)
                g_loss = g_lambda*torch.max(g_norms,1-torch.log(g_norms+1e-10)).mean()
                # g_loss.backward(retain_graph=True)  # incorporate into o_optimizer instead.
                # g_optimizer.step()
                gcounter += 1
            else:
                g_loss = tzero

            if not c_only:
                output = Out(hidden)
                o_loss = o_lambda * MSE(output, batch_mydata)
                o_alone = o_loss
                e_loss = e_lambda * Enf(hidden[:,mp.side_channel_size:], batch_mydata)  # Enforce H and O similarity
                # e_loss = tzero
                o_loss += e_loss
                o_loss += g_loss
                o_loss += (Gen.map1.weight.abs().sum() +
                           Gen.map2.weight.abs().sum() +
                           Gen.map3.weight.abs().sum() +
                           Gen.map4.weight.abs().sum()) * g_l1_lambda

                prediction = embedding_prediction(hidden)
                r_loss = F.binary_cross_entropy(prediction, batch_outcomes)
                o_loss += r_lambda*r_loss
                o_loss.backward()
                [p.grad.clamp_(-1,1) for p in Gen.parameters() if p.grad is not None]
                o_optimizer.step()
            else:
                o_loss, e_loss = tzero, tzero


            if np.random.uniform() < pr_c_update:
                chidden = Variable(hidden[:,mp.side_channel_size:].abs().data, requires_grad=False)  # separator
                # chidden = hidden[:,mp.side_channel_size:]
                clusters = Clu(chidden)
                if epoch < mp.burn_in:
                    c_loss = torch.tensor(0.).cuda() if gpu_mode else torch.tensor(0.)
                    pass
                else:
                    # Old:
                    # c_loss += c_lambda * (
                    #     batch_cosine(torch.sqrt(clusters+1e-10), normalize=False) -
                    #     F.relu(batch_cosine(chidden))).pow(2).sum(1).mean()
                    # From Jeremy's new jcw_cluster_multiple.py:
                    clis = 0
                    c_losses = [None] * len(c_output_size)
                    for clsi, cl_size in enumerate(c_output_size):
                        if using_side_labels and clsi + 1 == len(c_output_size):
                            # compute probability vector similarity as dot products. then use cross-entropy
                            # based on label agreement for the batch.
                            sl_loss = mp.lambda_side_labels / len(c_output_size) * -1 * (
                                batch_equals(batch_side_labels) * log_clamped(batch_dot(clusters[:,clis:(clis+cl_size)])) +
                                (batch_equals(batch_side_labels, negation=True)) * log_clamped(1 - batch_dot(clusters[:,clis:(clis+cl_size)]))).mean()
                            if torch.isnan(sl_loss):
                                pdb.set_trace()
                            c_losses[clsi] = sl_loss
                        else:
                            c_losses[clsi] = c_lambda / len(c_output_size) * (
                                batch_1norm(clusters[:,clis:(clis+cl_size)]+1e-10,
                                             ) - # normalize=False) -
                                batch_1norm(chidden)).pow(2).mean(1).mean()
                        clis += cl_size
                    c_loss = torch.stack(c_losses).sum()
                    # c_loss += c_lambda * (
                    #     batch_cosine(torch.sqrt(clusters+1e-10),
                    #                  normalize=False).pow(batch_cosine(chidden)
                    # c_loss += o_loss
                    c_loss.backward()
                    c_optimizer.step()
                    ccounter += 1
            else:
                c_loss = tzero

            epoch_losses += torch.stack((g_loss, o_alone, e_loss, c_loss, r_loss))

        if epoch % mp.print_interval == 0:
            el = epoch_losses.cpu().data.numpy()
            el *= 1*mp.minibatch_size/data_size*\
                  np.array([1.*data_size/mp.minibatch_size/gcounter,
                            1,
                            1,
                            1.*data_size/mp.minibatch_size/(ccounter+1e-10),
                            1])
            print("%s: [H: %6.4f]; [O: %6.4f; e: %6.4f]; C: %6.4f; R: %6.4f" %
                  (epoch, el[0], el[1], el[2], el[3], el[4]))

    # Old:
    # Gen = Gen.eval()
    # assignments = np.zeros(mynoise.shape[0])
    # for i0, i1 in arangeIntervals(data_size, 100):
    #     assignments[i0:i1] = np.argmax(
    #         Clu( Gen( mynoise[i0:i1].to(device) )[:,mp.side_channel_size:] ).        cpu().data.numpy(), axis=1)
    #     # for i0, i1 in arangeIntervals(data_size, mp.minibatch_size):
    # print(np.bincount(assignments.astype(int)))
    # From Jeremy's new jcw_cluster_multiple.py
    Gen = Gen.eval()
    assignments = np.zeros((mynoise.shape[0], len(c_output_size)))
    risks = np.zeros((mynoise.shape[0], 1))
    for i0, i1 in arangeIntervals(data_size, 100):
        embeddedv = Gen( mynoise[i0:i1].to(device))
        risks[i0:i1, 0] = embedding_prediction(embeddedv).cpu().data.numpy()
        membership_pr = Clu( embeddedv.abs()[:,mp.side_channel_size:] ).\
            cpu().data.numpy()
        cli = 0
        for cl_vi, cl_size in enumerate(c_output_size):
            assignments[i0:i1, cl_vi] = np.argmax(membership_pr[:,cli:cli+cl_size], axis=1)
            cli += cl_size
        # for i0, i1 in arangeIntervals(data_size, minibatch_size):
    _ = [print(len(np.unique(assignments[:,assni])),
               '/',
               c_output_size[assni],
               ':',
               np.bincount(assignments[:,assni].astype(int))) for assni in np.arange(assignments.shape[1])]

    def model(mynoise, Gen=Gen, Clu=Clu):
        Gen = Gen.eval()
        assignments = np.zeros((mynoise.shape[0], len(c_output_size)))
        risks = np.zeros((mynoise.shape[0], 1))
        data_size = mynoise.shape[0]
        for i0, i1 in arangeIntervals(data_size, 100):
            embeddedv = Gen(mynoise[i0:i1].to(device))
            risks[i0:i1, 0] = embedding_prediction(embeddedv).cpu().data.numpy()
            membership_pr = Clu(embeddedv.abs()[:,mp.side_channel_size:]).\
                            cpu().data.numpy()
            cli = 0
            for cl_vi, cl_size in enumerate(c_output_size):
                assignments[i0:i1, cl_vi] = np.argmax(membership_pr[:,cli:cli+cl_size], axis=1)
                cli += cl_size
        return {'assignments':assignments,
                'risks': risks,
                'embeddedv_asample': embeddedv}
    
    return {"assignments": assignments,
            "mynoise": mynoise,
            "risks": risks,
            "model": model}

class ModelParameters:
    def __init__(self, **kwargs):
        self.num_epochs = 1000  # 2000
        self.minibatch_size = 64
        self.burn_in = 10
        self.print_interval = 1

        self.o_learning_rate = 1e-3  # 2e-4
        # self.g_learning_rate = 1e-5
        self.c_learning_rate = 1e-3
        self.using_tsne = False
        self.lambda_side_labels = 1e-3

        self._latent_size = 128
        self._hidden_size = 8  # latent space size
        self.g_output_size = self._hidden_size
        self.side_channel_size = 1
        self.c_input_size = self._hidden_size - self.side_channel_size
        self.c_hidden_size = 128
        # c_output_size = 50 # number of clusters
        self.c_output_size = [8]  # np.arange(5, 15)
        self.o_input_size = self._hidden_size
        self.o_hidden_size = self._latent_size

        for k, v in kwargs.items():
            assert k in self.__dict__.keys()
            self.__dict__[k] = v

    def __repr__(self):
        dict_repr = self.__dict__.__repr__()
        return "ModelParameters(**" + dict_repr + ")"

### What's different when risk stratifying and clustering is that the values determine
### the clustering, and the risk stratification is determined by signs (and magnitude for gradient).
### the first layer of the risk stratification determines the weights of the cossim (does this work?)

if __name__ == "__main__":
    # if True:    
    ### Synthetic data
    # desired_centroids = 10
    # noise_sd = 0.01
    # explode_factor = 10000
    # mydatasize = torch.Size((100, 100))
    # centroidsize = torch.Size((desired_centroids, mydatasize[1]))
    # centroids = F.normalize(torch.FloatTensor(centroidsize).normal_(),2,1)
    # mydata = torch.cat([torch.FloatTensor(torch.Size((int(mydatasize[0]/centroidsize[0]),
    #                                                   mydatasize[1]))).normal_(std=noise_sd) +
    #                     c for c in centroids])
    # mydata = mydata * torch.FloatTensor(torch.Size([mydatasize[0]])).\
    #     random_(1,explode_factor).unsqueeze(1)
    # mydata = mydata / torch.min(mydata.norm(2,1),torch.ones_like(mydata[:,0])).unsqueeze(1)
    # true_assignments = np.repeat(np.arange(centroidsize[0]),int(mydatasize[0]/centroidsize[0]))

    # risk = torch.sigmoid((mydata.norm(2,dim=1)-mydata.norm(2,dim=1).mean())/
    #                      mydata.norm(2,dim=1).std()*
    #                      torch.log(1+torch.tensor(true_assignments).float()))
    # risk_bits = torch.distributions.bernoulli.Bernoulli(risk).sample()

    # mydata = F.normalize(Variable(mydata),2,0)
    # # mydata = Variable(mydata)
    # # mydata = Variable((1-2*(mydata > 0).float())*torch.log(1+torch.abs(mydata)))
    # #mydatadf = pd.DataFrame(mydata.data.numpy())
    # #mydatadf['npi'] = true_assignments

    import pandas as pd
    pd_mydata = pd.read_csv('~/workspace/marshfield/recode_like_data/training_wide.csv')
    pd_myoutcomes = pd.read_csv('~/workspace/marshfield/recode_like_data/training_outcomes_wide.csv')
    pd_mytestdata = pd.read_csv('~/workspace/marshfield/recode_like_data/holdout_wide.csv')
    pd_mytestoutcomes = pd.read_csv('~/workspace/marshfield/recode_like_data/holdout_outcomes_wide.csv')

    pd_mydata = pd_mydata.drop(columns='STUDY_ID')
    mydata = torch.log1p(torch.tensor(pd_mydata.as_matrix()).float())
    pd_myoutcomes = pd_myoutcomes.drop(columns='STUDY_ID')
    myoutcomes = torch.tensor(torch.tensor(pd_myoutcomes.as_matrix()).float().sum(1) > 0).float()
    pd_mytestdata = pd_mytestdata.drop(columns='STUDY_ID')
    mytestdata = torch.log1p(torch.tensor(pd_mytestdata.as_matrix()).float())
    pd_mytestoutcomes = pd_mytestoutcomes.drop(columns='STUDY_ID')
    mytestoutcomes = torch.tensor(torch.tensor(pd_mytestoutcomes.as_matrix()).float().sum(1) > 0).float()

    true_assignments = None
    risk_bits = myoutcomes
    
    # true_assignments = None
    side_label_assignments = None

    mp = ModelParameters()
    mp.num_epochs = 500
    # run(mydata, None, mp)
    result = run(mydata, side_label_assignments, mp, risk_bits, true_assignments)

    result['model'](mytestdata)
    result['model'](mydata)['risks']

    # Our method (risk and cluster): AUC
    from sklearn.metrics import roc_auc_score
    print('Training AUC:', roc_auc_score(risk_bits, result['model'](mydata)['risks'][:,0]))
    print('Test AUC:', roc_auc_score(mytestoutcomes, result['model'](mytestdata)['risks'][:,0]))

    # Clusters found by method: AUC
    def group_risk(assignments, outcomes):
        subgroup_df = pd.DataFrame({'assignments': assignments,
                                    'outcomes': outcomes})
        sg_risk = subgroup_df.groupby(['assignments']).mean().to_dict()['outcomes']
        return sg_risk
    sg_risk = group_risk(result['assignments'][:,0], risk_bits)
    print('Our clustering, risk by label; training AUC:',
          roc_auc_score(risk_bits,
                        np.array([sg_risk[s] for s in result['model'](mydata)['assignments'][:,0]])))
    print('Our clustering, risk by label: test AUC:',
          roc_auc_score(mytestoutcomes.cpu().numpy(),
                        np.array([sg_risk[s] if s in sg_risk else 0
                                  for s in
                                  result['model'](mytestdata)['assignments'][:,0]])))


    # Cluster found by k-means: AUC
    from sklearn.cluster import KMeans
    km = KMeans(mp.c_output_size[0]).fit(mydata.cpu().numpy())
    km_sg_risk = group_risk(km.labels_, risk_bits)
    print('K-means, risk by label; training AUC:',
          roc_auc_score(risk_bits,
                        np.array([km_sg_risk[s] for s in km.predict(mydata.cpu().numpy())])))
    print('K-means, risk by label; training AUC:',
          roc_auc_score(mytestoutcomes.cpu().numpy(),
                        np.array([km_sg_risk[s] for s in km.predict(mytestdata.cpu().numpy())])))    
    
    # LASSO: AUC
    from sklearn.linear_model import LassoCV
    lasso = LassoCV().fit(pd_mydata, myoutcomes.cpu().numpy())
    print('Lasso train: AUC',
          roc_auc_score(myoutcomes, lasso.predict(mydata.cpu().numpy())))
    print('Lasso test: AUC',
          roc_auc_score(mytestoutcomes, lasso.predict(mytestdata.cpu().numpy())))


    from sklearn.linear_model import LogisticRegressionCV
    lrcv = LogisticRegressionCV([3e-4, 1e-3, 1e-2, 1e-1],
                                penalty='l1', solver='liblinear', n_jobs=-1, cv=5)
    lr_lasso = lrcv.fit(pd_mydata.as_matrix(),
                        myoutcomes.cpu().numpy())
    print('LR Lasso train: AUC',
          roc_auc_score(myoutcomes, lr_lasso.predict(mydata.cpu().numpy())))
    print('LR Lasso test: AUC',
          roc_auc_score(mytestoutcomes, lr_lasso.predict(mytestdata.cpu().numpy())))
    

    # Now compare:
    # highest predicted risk (above threshold) intersected with highest risk cluster
    # k-means highest risk
    # lassoCV highest risk
    # in terms of (1) risk and (2) homogeneity
    riskdf = pd.DataFrame.from_dict(sg_risk, orient='index')
    riskdf = riskdf.reset_index()
    riskdf.columns = ['assignment','risk']
    riskdf['count'] = [s for s in np.bincount(result['assignments'][:,0].astype(int)) if s!=0]
    riskdf['score'] = np.log1p(riskdf['count']) * riskdf.risk
    highest_risk_group = riskdf.iloc[riskdf.idxmax()[-1],0]
    in_hrg = result['model'](mytestdata)['assignments'][:,0] == highest_risk_group
    print('High-risk intersection (ours, ours): AUC',
          roc_auc_score(mytestoutcomes.cpu().numpy()[in_hrg],
                        result['model'](mytestdata)['risks'][in_hrg]))
    test_sg_risk = group_risk(result['model'](mytestdata)['assignments'][:,0], mytestoutcomes)
    test_riskdf = pd.DataFrame.from_dict(test_sg_risk, orient='index')
    test_riskdf = test_riskdf.reset_index()
    test_riskdf.columns = ['assignment','risk']
    test_riskdf['count'] = [s for s in
                            np.bincount(result['model'](mytestdata)['assignments'][:,0].\
                                        astype(int)) if s!=0]
    test_riskdf['score'] = np.log1p(test_riskdf['count']) * test_riskdf.risk
    print(riskdf)
    print(test_riskdf)

    
    
    # # If you have ground truth clustering
    # from sklearn.metrics import adjusted_rand_score
    # print('ARI:', adjusted_rand_score(true_assignments, result['assignments'][:,0]))
