import numpy as np
import h5py
import itertools

import torch
import torch.nn.functional as F
from torch import optim, Tensor, nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

input_size = 5000
learning_rate = 0.0001
num_epochs = 1000
minibatch_size = 32
print_interval = 25

class TrainData(object):
    '''training data object'''

    def __init__(self,h5_file,batch_size):
        self.h5_file = h5_file
        self.batch_size = batch_size

    def batcher(self):
        iterable = xrange(0,self.h5_file.values()[0].shape[0] - \
            self.batch_size,self.batch_size)
        for i in itertools.cycle(iterable):
            yield {key: val[i:i+self.batch_size] for key,val in \
                self.h5_file.items()}

# autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.map1 = torch.nn.Linear(input_size, 1000)
        self.map2 = torch.nn.Linear(1000, 500)
        self.bn1 = torch.nn.BatchNorm1d(500)
        self.map3 = torch.nn.Linear(500, 50)
        self.map4 = torch.nn.Linear(50, 500)
        self.bn2 = torch.nn.BatchNorm1d(500)
        self.map5 = torch.nn.Linear(500,1000)
        self.map6 = torch.nn.Linear(1000,input_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        rep = F.leaky_relu(self.bn1(self.map3(x)))
        x = F.leaky_relu(self.map4(rep))
        x = F.leaky_relu(self.bn2(self.map5(x)))
        return rep, self.map6(x)

# variational autoencoder
class VAE(nn.Module):
    def __init__(self, input_size, z_dim=50):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000,100),
            nn.LeakyReLU(),
            nn.Linear(100, z_dim*2))  # 2 for mean and variance.
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100,1000),
            nn.LeakyReLU(),
            nn.Linear(1000, input_size),
            nn.Sigmoid())
    
    def reparametrize(self, mu, sd):
        eps = Variable(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps*sd
        return z
                     
    def forward(self, x):
        h = self.encoder(x)
        mu, sd = torch.chunk(h, 2, dim=1)  # mean and log variance.
        z = self.reparametrize(mu, sd)
        out = self.decoder(z)
        return z, out, mu, sd
    
    def sample(self, z):
        return self.decoder(z)

    def train(self,train_data):
        model.zero_grad()
        input_data = Variable(torch.Tensor(train_data))
        rep, output, mu, sd = model(input_data)
        reconstruct_loss = loss1(output,input_data) + loss2(output,input_data)

        # update parameters
        reconstruct_loss.backward()
        optimizer.step()

        return reconstruct_loss

loss1 = nn.BCELoss()
loss2 = nn.KLDivLoss()

# training data file
train_h5file = 'RNASeqNorm5000Data.h5'
train_file = h5py.File(train_h5file,'r')
train_data = TrainData(train_file,minibatch_size)
train_batcher = train_data.batcher()

# instantiate Autoencoder
model = VAE(input_size=input_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trackLoss = []
for epoch in range(num_epochs):

    # train on noisy data
    train_data = np.clip(train_batcher.next()['rnaseq'] + \
        np.random.normal(0,0.1,minibatch_size*input_size).reshape(minibatch_size,input_size),0,1)

    reconstruct_loss = model.train(train_data)

    trackLoss.append(reconstruct_loss.data.numpy()[0])

    if epoch % print_interval == 0:
    	print('Epoch: ' + str(epoch))
    	print('Loss: ' + str(np.mean(trackLoss)))
    	trackLoss = []

torch.save(model.state_dict(),'VAE.pt')