# -*- coding: utf-8 -*-
"""
Some utility functions, including the one making pytorch dataset to iterate.
Created on Thu Nov  9 15:41:21 2017

@author: tingt
"""
#%% Some modules.
import time
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import optim, Tensor, nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

tensortype = torch.FloatTensor
#%% Useful autograd functions...

# PyTorch function for calcuating log \phi(x). From cs281 homework.
# example usage: normlogcdf1 = NormLogCDF()((h-b_r)/sigma)
class NormLogCDF(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        input_numpy = input.numpy()
        output = torch.Tensor(norm.logcdf(input_numpy))
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        input_numpy = input.numpy()
        grad_input = grad_output.clone()
        grad_input = grad_input * torch.Tensor(np.exp(norm.logpdf(input_numpy) - norm.logcdf(input_numpy)))
        # clip infinities to 1000
        grad_input[grad_input==float('inf')] = 1000
        # clip -infinities to -1000
        grad_input[grad_input==float('-inf')] = -1000
        # set nans to 0
        grad_input[grad_input!=grad_input] = 0
        return grad_input

# First, the loss function for the cases where PFS_flag is 1, i.e. progressed at the time observed.
# Since we assume a normal distribution for the PFS time, the likelihood is estimated as always, 
# as a point in the gaussian...

def GaussianLogLikelihood(x, mu, sigma2, size_average=False):
    """
    Return the negative log-likelihood deduced from Gaussian distribution, 
    x ~ N(mu, sigma2).
    """
    assert len(x) == len(mu)
    
    #result = torch.sum(-0.5 * math.log(2*math.pi) - 0.5 * math.log(sigma2) \
    result = torch.sum( 1./(2*sigma2) * (x - mu).pow(2))
    if size_average:
        result = torch.div(result, len(x))
    return result

## Then, for the cases where PFS_flag is 0, i.e. not progressed at the time observed.
## In this case, the likelihood would be the CDF above the observation time.
## -------|++++++
## Thus, the negative log-likelihood would be the log-normalcdf...

def CDFLogLikelihood(x, mu, sigma2, size_average=False):
    assert len(x) == len(mu)
    
    # We want the area beyond x... the same area below 2*mu - x
    result =  NormLogCDF()((mu - x) / torch.sqrt(sigma2))
    
    if size_average:
        result = torch.div(result, len(x))
    return -result

def combinedLoss(predPFS, targPFS, targPFSFlag, sigma2, size_average=False):
    """ Compute the loss based on PFS_flag.."""
    assert len(predPFS) == len(targPFS) == len(targPFSFlag)
    
    batch_size = len(predPFS)
    
    TF_0 = (targPFSFlag.numpy() == 0)
    TF_1 = (targPFSFlag.numpy() == 1)
    
    if TF_0.sum() > 0:
        index_0 = torch.from_numpy(np.array(range(batch_size), dtype=int)[TF_0]).type(torch.LongTensor)
        predPFS_0 = predPFS[index_0]
        targPFS_0 = targPFS[index_0]
        # Note, somehow, NormLogCDF only takes floatTensor for backward.
        loss_0 = torch.sum(CDFLogLikelihood(targPFS_0, predPFS_0, sigma2).type(tensortype))
    
    else:
        loss_0 = 0
    
    
    if TF_1.sum() > 0:
        index_1 = torch.from_numpy(np.array(range(batch_size), dtype=int)[TF_1]).type(torch.LongTensor)
        predPFS_1 = predPFS[index_1]
        targPFS_1 = targPFS[index_1]
        loss_1 = GaussianLogLikelihood(targPFS_1, predPFS_1, sigma2).type(tensortype)
    else:
        loss_1 = 0
    
    
    if size_average:
        return torch.div(loss_0 * TF_0.sum() + loss_1 * TF_1.sum(), batch_size)
    else:
        return loss_0 + loss_1
#%% PyTorch dataset class to load genomic / clinical data as torch datasets, and iteratable.
class GeneData(Dataset):
    """
    Dataset for clinical + genomic data.
    When loading, use:
        dataloader = DataLoader(data, batch_size = 4, shuffle = True)
        for i_batch, sample in enumerate(dataloader):
            ...
        
    """
    def __init__(self, clinical, genomic):
        """
        Both clinical/genomic data should be in pandas.dataframe, with patient 
        id as index, and other features as columns.
        """
        self.Npatients_total = clinical.shape[0]
        self.Npatients_genom, self.Ngenom_features = genomic.shape
        self.clinical = clinical.copy()
        # Transformation for the PFS... downgrade the scale.
        self.clinical.D_PFS = self.clinical.D_PFS.div(10)
        
        self.clinical.D_PFS_FLAG = self.clinical.D_PFS_FLAG.astype(int)
        
        self.genomic = genomic.copy()
        self.genomic_name = list(genomic.columns)
        
        self.cyto_columns = [x for x in clinical.columns if x.startswith("CYTO")]
        self.clinical_cyto = clinical.loc[:, self.cyto_columns].fillna(0) # For unpresent cyto data, fill as 0.5 instead of 0 or 1.
        self.Ncyto_features = len(self.cyto_columns)
        
        # Also transform the other information, divide by their maximum to make all the values within [0,1].
        self.other_columns = ["D_Age"] + [x for x in clinical.columns if x.startswith("CBC") \
                              or x.startswith("DIAG") or x.startswith("CHEM")]
        self.clinical_other = clinical.loc[:, self.other_columns].fillna(0)
        self.clinical_other = self.clinical_other / self.clinical_other.max(0)
        self.Nother_features = len(self.other_columns)
        
    def __len__(self):
        """
        Currently, only care about genomic data. Changeable in the future.
        """
        return self.Npatients_genom
    
    def __getitem__(self, idx):
        """
        Return the idx-th item in the pandas... following the order of genomic data.
        """
        patient_id = self.genomic.index[idx]
        assert patient_id in self.clinical.index
        return   {'patient':    patient_id, \
                  'genomic':    self.genomic.loc[patient_id, :].values, \
                  'cyto':       self.clinical_cyto.loc[patient_id, :].values, \
                  'other':      self.clinical_other.loc[patient_id, :].values, \
                  'PFS':        self.clinical.loc[patient_id, "D_PFS"], \
                  'PFS_FLAG':   self.clinical.loc[patient_id, "D_PFS_FLAG"]   
                  }

def get_train_valid_loader(genedata, batch_size = 4, 
                           random_seed  = 1024,
                           valid_size   = 0.15,
                           test_size    = 0.1,
                           shuffle      = True,
                           num_workers  = 0,
                           pin_memory   = False):
    """
    Utitlity function to split and return the train, valid, and test iterators over the given dataset. 
    By default, train:valid:test = 7:2:1.
    Adapted from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb.
    Params
    ----
    - genedata:     loaded dataset.
    - batch_size:   how many samples per batch to load.
    - random_seed:  fix seed for reproducibility.
    - valid_size:   percentage split on the training set. Float within [0,1].
    - test_size:    percentage split on the training set. Float within [0,1].
    - shuffle:      whether to split sets randomly. 
    - num_workers:  number of subprocesses to use when loading the dataset.
    - pin_memory:   whether to copy tensors into CUDA pinned memory.
    
    Returns
    ----
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - test_loader:  test set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
        
    num_train   = len(genedata)
    indices     = list(range(num_train))
    split_valid = int(np.floor(valid_size * num_train))
    split_test  = int(np.floor(test_size  * num_train))
    
    # Shuffle. otherwise just split based on the given order.
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, valid_idx, test_idx = indices[split_valid + split_test:], indices[:split_valid], indices[split_valid:split_valid + split_test]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler  = SubsetRandomSampler(test_idx)
    
    train_loader  = DataLoader(genedata, sampler = train_sampler,
                               batch_size = batch_size, 
                               num_workers= num_workers, 
                               pin_memory = pin_memory)

    valid_loader  = DataLoader(genedata, sampler = valid_sampler,
                               batch_size = batch_size, 
                               num_workers= num_workers, 
                               pin_memory = pin_memory)
    
    test_loader  = DataLoader(genedata, sampler = test_sampler,
                               batch_size = batch_size, 
                               num_workers= num_workers, 
                               pin_memory = pin_memory)
    
    return (train_loader, valid_loader, test_loader)

#%% The model...
class simple(nn.Module):
    def __init__(self, data):
        """
        simple logistic regression model.
        """
        super(simple, self).__init__()
        self.w_cyto     = Variable(torch.zeros(data.Ncyto_features,1).type(tensortype))
        
        self.w_others   = nn.Parameter(torch.zeros(data.Nother_features, 1).type(tensortype))
        
        self.w_genomic  = nn.Parameter(torch.zeros(data.Ngenom_features,1).type(tensortype))
        self.bias       = Variable(tensortype([data.clinical.D_PFS.mean()]))
        self.sigma2     = Variable(tensortype([data.clinical.D_PFS.std() ** 2]))
        
    def predict(self, genomic, cyto, other):
        """
        Simple linear regression.
        """
        return (genomic @ self.w_genomic + cyto @ self.w_cyto + other @ self.w_others + self.bias).squeeze()
    
    def forward(self, genomic, cyto, other, PFS, PFS_FLAG):
        """
        Compute the loss.
        """
        predPFS = self.predict(genomic, cyto, other)
        return combinedLoss(predPFS, PFS, PFS_FLAG, self.sigma2, size_average=False)
        #target = PFS + Variable(1- PFS_FLAG).type(tensortype) * 50
        #return nn.MSELoss()(predPFS, target)

def train(model, optimizer, genomic, cyto, other, PFS, PFS_FLAG):
    # Resets the gradients to 0
    optimizer.zero_grad()
    # Computes the function above.
    # Computes a loss. Gives a scalar. 
    output = model.forward(genomic, cyto, other, PFS, PFS_FLAG)
    # Magically computes the gradients. 
    output.backward()
    # updates the weights
    optimizer.step()
    return output.data[0]

def main(clinical, genomic, batch_size = 5, num_epochs = 1):
    report_every= 100
    plot_every  = 100
    # Load the data. Clinical & Genomic are two pd.DataFrames that contain the 
    # clinical information, and cleaned genomic features, respectively.
    genedata = GeneData(clinical, genomic)
    train_loader, valid_loader, test_loader = get_train_valid_loader(genedata, batch_size=batch_size,
                                                                     random_seed  = 1024,
                                                                     valid_size   = 0.15,
                                                                     test_size    = 0.1,
                                                                     shuffle      = True)
    
    # Build the model and the corresponding optimizer. 
    # Loss function is integrated in the model already, no need to specify again.
    model       = simple(genedata)
    optimizer   = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    
    
    # Now go!
    cost = 0
    total_batches = len(train_loader)
    for epoch in range(num_epochs):
        #print ('Starting epoch %d'%(epoch+1))
        start = time.time()
        for i_batch, batch in enumerate(train_loader):    
            cost    += train(model, optimizer, \
                             Variable(batch['genomic']).type(tensortype), \
                             Variable(batch['cyto']).type(tensortype), \
                             Variable(batch['other']).type(tensortype), \
                             Variable(batch['PFS']).type(tensortype), batch['PFS_FLAG'])
            #print(cost)
            if i_batch%report_every==0 and i_batch>0:
                #print ('Epoch %d %.2f%%. Loss %f'%(epoch+1, 100.*i_batch / total_batches, cost/report_every))
                cost = 0
        end = time.time()
        #print ("Time per epoch: " + str(end-start))
        
        PFS      = np.array([])
        PFS_FLAG = np.array([], dtype=int)
        predPFS  = np.array([])
        if (epoch + 1) % plot_every == 0 or (epoch + 1) == num_epochs:
            for i_batch, batch in enumerate(valid_loader):
                PFS         = np.append(PFS, batch['PFS'].numpy())
                PFS_FLAG    = np.append(PFS_FLAG, batch['PFS_FLAG'].numpy())   
                predPFS     = np.append(predPFS, \
                                        model.predict(Variable(batch['genomic']).type(tensortype), \
                                                      Variable(batch['cyto']).type(tensortype), \
                                                      Variable(batch['other']).type(tensortype)).data.numpy())
            color = np.array(['g', 'r'])[PFS_FLAG]
#                loss = model.forward(Variable(batch['genomic']).type(tensortype), \
#                             Variable(batch['cyto']).type(tensortype), \
#                             Variable(batch['other']).type(tensortype), \
#                             Variable(batch['PFS']).type(tensortype), batch['PFS_FLAG'])
            plt.subplot(1,2,1)
            plt.scatter(PFS, predPFS, c = color)
            plt.plot(PFS, PFS)
            plt.title("Epoch %d, validation" %(epoch + 1))
            
            
            PFS_train      = np.array([])
            PFS_FLAG_train = np.array([], dtype=int)
            predPFS_train  = np.array([])
            for i_batch, batch in enumerate(train_loader):
                PFS_train         = np.append(PFS_train, batch['PFS'].numpy())
                PFS_FLAG_train    = np.append(PFS_FLAG_train, batch['PFS_FLAG'].numpy())   
                predPFS_train     = np.append(predPFS_train, \
                                        model.predict(Variable(batch['genomic']).type(tensortype), \
                                                      Variable(batch['cyto']).type(tensortype), \
                                                      Variable(batch['other']).type(tensortype)).data.numpy())
            color = np.array(['g', 'r'])[PFS_FLAG_train]
            plt.subplot(1,2,2)
            plt.scatter(PFS_train, predPFS_train, c = color)
            plt.plot(PFS_train, PFS_train)
            plt.title("Epoch%d, training plot" % (epoch + 1))
            plt.show()    
    
    return model, PFS, PFS_FLAG, predPFS

#%% Final block...
if __name__ == '__main__':
    # Read data.
    mainFile = "E:\\MM\\TrainingData\\Clinical Data\\mmrf.clinical.csv"
    drop_col = ["Study", "Disease_Status", "Disease_Type", "Cell_Type"]
    clinical = pd.read_csv(mainFile, index_col = 1).drop(drop_col, axis=1).dropna(subset=['D_PFS'])
    genomic  = pd.read_csv("genomic_clean.csv", index_col = 0)
    
    model, PFS, PFS_FLAG, predPFS = main(clinical, genomic, batch_size = 50, num_epochs = 2000)
    
    np.savetxt("PFS.csv", PFS * 10, delimiter=',')
    np.savetxt("PFS_FLAG.csv", PFS_FLAG, delimiter=',')
    np.savetxt("predPFS.csv", np.clip(predPFS * 10, 0, 100000), delimiter=',')
    
    color = np.array(['g', 'r'])[PFS_FLAG]
    plt.scatter(PFS, np.clip(predPFS,0, 100000), c = color)
    plt.plot(PFS, PFS)
    plt.show()