# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Main file to execute the model on the LPBF dataset
"""

#%%
#import libraries

import torch 
from torch.optim.lr_scheduler import StepLR
torch.cuda.empty_cache()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

from Utils import *
from Network import *
from Data_Manipulation import *
from Saliency_Utils import *

#%%
# Data--> https://polybox.ethz.ch/index.php/s/MUmJXXXBxpK1Ejc
# data = '../data/' #place the Data inside the folder
#Loading dataset

trainset_2500,testset_2500,classspace_2500=dataloading_funtion('data',2500)
trainset_5000,testset_5000,classspace_5000=dataloading_funtion('data',5000)
trainset_7500,testset_7500,classspace_7500=dataloading_funtion('data',7500)
trainset_10000,testset_10000,classspace_10000=dataloading_funtion('data',10000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


class_num = np.concatenate((classspace_2500, classspace_5000,classspace_7500,classspace_10000), axis=0)
values, counts = np.unique(class_num, return_counts=True)
class_weights=classweight(values,counts)
class_weights = torch.from_numpy(class_weights)


#%% Data Loader

data_2500=[trainset_2500,testset_2500]
data_5000=[trainset_5000,testset_5000]
data_7500=[trainset_7500,testset_7500]
data_10000=[trainset_10000,testset_10000]

Training_batch =[data_2500,data_5000,data_7500,data_10000]

#%% Model initiation
epoch=10
n_features = 10
n_classes=3
net= CNN(n_features, n_classes)
net.apply(initialize_weights)
net.to(device)
class_weights=class_weights.to(device,dtype=torch.float)

print(class_weights.is_cuda) 

#%% Network Training

costFunc = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer =  torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=0.003)
scheduler = StepLR(optimizer, step_size = 100, gamma= 0.3 )

Loss_value =[]
Train_loss =[]
Iteration_count=0
iteration=[]
Epoch_count=0
Total_Epoch =[]
Accuracy=[]
Learning_rate=[]
Training_loss_mean = []
Training_loss_std = []
Times=[]

for epoch in range(epoch):
    epoch_smoothing=[]
    learingrate_value = get_lr(optimizer)
    Learning_rate.append(learingrate_value)
    closs = 0
    scheduler.step()
    num=random.randint(0, 3)
    print(num)
    Times.append(num)
    trainset,testset =Training_batch[num]
    # print(trainset)
    # print(testset)
    
    for i,batch in enumerate(trainset,0):
        
        data,output = batch
        data,output = data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        # print("Input data",data.shape)
        # print("Input label",output.shape)
        # print("Modified Input label",torch.squeeze(output, 1))
        prediction = net(data) 
        # print("prediction label",prediction.shape)
        loss = costFunc(prediction,output.squeeze()) #torch.Size([100, 3]),#torch.Size([100])
        
        # Specify L1 and L2 weights
        factor = 0.00005
        
        # Compute L1 and L2 loss component
        l1_crit = nn.L1Loss(size_average=False)
        reg_loss = 0
        
        for param in net.parameters():
            reg_loss += l1_crit(param,target=torch.zeros_like(param))

        # Add L1 loss components
        
        loss += factor * reg_loss
        # loss += l2
        
        closs = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_smoothing.append(closs)
        
        if i%100 == 0:
            print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs))
            
    loss_train = closs / len(trainset)
    Loss_value.append(loss_train)
    
    Training_loss_mean.append(np.mean(epoch_smoothing))
    Training_loss_std.append(np.std(epoch_smoothing))
    
    correctHits=0
    total=0
    
    for batches in testset:
        
        net.eval()
        data,output = batches
        data,output =data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        
        output=output.squeeze()
        # output=torch.squeeze(output, 1)
        
        prediction = net(data)
        prediction = torch.argmax(prediction, dim=1) 
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
        net.train()
   
    Epoch_count = epoch+1
    Total_Epoch.append (Epoch_count)
    Epoch_accuracy = (correctHits/total)*100
    Accuracy.append(Epoch_accuracy)
    print('Accuracy on epoch [%d] window [%d]  :  %.3f' %(epoch+1,data.shape[2],Epoch_accuracy))
    
PATH = './CNN_LSTM_Multivariate'+'.pth'
torch.save(net.state_dict(), PATH)
torch.save(net, PATH)
model = torch.load(PATH)


   
#%% Training Plots

iter_1 = '0.83 ms'
iter_2 = '1.65 ms' 
iter_3 = '2.50 ms'
iter_4 = '3.3 ms'
class_names = [iter_1,iter_3,iter_3,iter_4]
plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std,class_names,Times)
count_parameters(net)


#%% Confusio matrix


windowresults(testset_2500,model,device,'2500')
windowresults(testset_5000,model,device,'5000')
windowresults(testset_7500,model,device,'7500')
windowresults(testset_10000,model,device,'10000')


#%% Validation on unknow time-scales

# trainset_4000,testset_4000,_=dataloading_funtion('data',4000)
# windowresults(testset_4000,model,device,'4000')

# trainset_6000,testset_6000,_=dataloading_funtion('data',6000)
# windowresults(testset_6000,model,device,'6000')

# trainset_9000,testset_9000,_=dataloading_funtion('data',9000)
# windowresults(testset_9000,model,device,'9000')

# trainset_12000,testset_12000,_=dataloading_funtion('data',12000)
# windowresults(testset_12000,model,device,'12000')

# trainset_1500,testset_1500,_=dataloading_funtion('data',1500)
# windowresults(testset_1500,model,device,'1500')

#%%

