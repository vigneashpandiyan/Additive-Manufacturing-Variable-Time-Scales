# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Main Utils file for saliency
"""

#%%
import torch
from torch import optim, cuda
import torchvision.models as models
from torchvision import datasets
import torchvision 
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader,TensorDataset
torch.cuda.empty_cache()
from Utils import *
from Network import *
from Data_Manipulation import *
from Saliency_Utils import *

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

PATH = './CNN_LSTM_Multivariate'+'.pth'
net = torch.load(PATH)
torch.backends.cudnn.enabled = False
net.to(device)

#%% Testdataloader with batch size of 1

testset_2500=dataloading_funtion_saliency('data',2500)
torch.save(testset_2500, 'data/testset2500')
print('done 2500')

testset_5000=dataloading_funtion_saliency('data',5000)
torch.save(testset_5000, 'data/testset5000')
print('done 5000')

testset_7500=dataloading_funtion_saliency('data',7500)
torch.save(testset_7500, 'data/testset7500')
print('done 7500')

testset_10000=dataloading_funtion_saliency('data',10000)
torch.save(testset_10000, 'data/testset10000')
print('done 10000')


#%% Compute saliencies across windows

y_true2500,y_pred2500,saliencies2500 = window_saliency_results(testset_2500,net,device,'2500')
y_true5000,y_pred5000,saliencies5000 = window_saliency_results(testset_5000,net,device,'5000')
y_true7500,y_pred7500,saliencies7500 = window_saliency_results(testset_7500,net,device,'7500')
y_true10000,y_pred10000,saliencies10000 = window_saliency_results(testset_10000,net,device,'10000')


#%% Normalize across windows


testset_2500  = torch.load('data/testset2500')
testset_5000  = torch.load('data/testset5000')
testset_7500  = torch.load('data/testset7500')
testset_10000 = torch.load('data/testset10000')


saliencies2500 = torch.cat(torch.load('data/salienciesNorm2500'), 0).cpu()
saliencies5000 = torch.cat(torch.load('data/salienciesNorm5000'), 0).cpu()
saliencies7500 = torch.cat(torch.load('data/salienciesNorm7500'), 0).cpu()
saliencies10000 = torch.cat(torch.load('data/salienciesNorm10000'), 0).cpu()

#%%

saliencies2500PerCat=compute_normalize_window(saliencies2500,testset_2500,'data',2500)
saliencies5000PerCat=compute_normalize_window(saliencies5000,testset_5000,'data',5000)
saliencies7500PerCat=compute_normalize_window(saliencies7500,testset_7500,'data',7500)
saliencies1000PerCat=compute_normalize_window(saliencies10000,testset_10000,'data',10000)


distribution_plot(saliencies2500PerCat,"0.83")
distribution_plot(saliencies5000PerCat,"1.65")
distribution_plot(saliencies7500PerCat,"2.50")
distribution_plot(saliencies1000PerCat,"3.30")
