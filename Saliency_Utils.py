# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Utils file for saliency
"""

#%%
import torch
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from Data_Manipulation import *
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
# print(device)


#%% Loading the data for saliency model
def dataloading_funtion_saliency(folder,window):
    
    # folder='data'
    rawfile_1 = str(folder)+'/'+'Channel0_'+str(window)+'.npy'
    rawfile_2 = str(folder)+'/'+'Channel1_'+str(window)+'.npy'
    rawfile_3 = str(folder)+'/'+'Channel2_'+str(window)+'.npy'
    rawfile_4 = str(folder)+'/'+'Channel3_'+str(window)+'.npy'
    classfile = str(folder)+'/'+'classspace_'+str(window)+'.npy'
    
    Rawspace_1 = normalize(rawfile_1).astype(np.float64)
    Rawspace_2 = normalize(rawfile_2).astype(np.float64)
    Rawspace_3 = normalize(rawfile_3).astype(np.float64)
    Rawspace_4 = normalize(rawfile_4).astype(np.float64)
    
    classspace= np.load(classfile).astype(np.float64)

    trainset,testset=data_batch_prep_saliency(Rawspace_1,Rawspace_2,Rawspace_3,Rawspace_4,classspace)
    
    return testset

#%%

def data_batch_prep_saliency(Rawspace_1,Rawspace_2,Rawspace_3,Rawspace_4,classspace):
    
    sequences_batch =[]
    for i in range(len(classspace)):
        # print(i)
         
        sequence_features_1 = Rawspace_1[i]
        # sequence_features_1 = normalize(sequence_features_1)
        
        sequence_features_2 = Rawspace_2[i]
        # sequence_features_2 = normalize(sequence_features_2)
        
        sequence_features_3 = Rawspace_3[i]
        # sequence_features_3 = normalize(sequence_features_3)
        
        sequence_features_4 = Rawspace_4[i]
        # sequence_features_4 = normalize(sequence_features_4)
        
        label = classspace[i]
        
        sequences_batch.append((sequence_features_1,sequence_features_2,sequence_features_3,sequence_features_4,label))
        
    sequences_batch = Mechanism(sequences_batch)
    
    train, test = train_test_split(sequences_batch, test_size=0.3, random_state=42)


    trainset = torch.utils.data.DataLoader(train, batch_size=200, num_workers=0,
                                            shuffle=True)

    testset = torch.utils.data.DataLoader(test, batch_size=1, num_workers=0,
                                            shuffle=True)
        
    return trainset, testset


#%%

def compute_saliency_time(input_sample, model, relative=False):
    # Prepare the input 
    input_sample.requires_grad_()
    if (torch.cuda.is_available()):
        input_sample = input_sample.cuda()
    # Collect output from the model 
    output = model(input_sample)
    #print(output.shape)
    # Require the gradient 
    output.requires_grad_()
    # Collect the unit responsible for the classification
    output_max = output.max(1)[0]
    #print(output_max.shape)
    # Retain grads 
    output_max.retain_grad()
    input_sample.retain_grad()
    
    # print(output_max)
    # Compute the gradients
    output_max.backward()
    # Collect gradient
    grad = input_sample.grad
    # slc = (grad - grad.min())/(grad.max()-grad.min())
    # Compute abs value 
    slc = torch.abs(grad)
    
    if relative:
        eps = 1e-5
        slc = slc/(torch.abs(input_sample)+eps)
    
    saliency = slc.detach()
    input_sample.grad = torch.zeros_like(grad)
    return saliency


def window_saliency_results(testset,model,device,window):
    
    y_pred     = []
    y_true     = []
    saliencies = []
    model.eval()
    count = 0
    # iterate over test data
    for batches in testset:
        
        data,output = batches
        data,output = data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        output = output.squeeze()
        
        if count%100==0:
            print('Computing prediction for sample: ', count)
            
        prediction = model(data)
        prediction = torch.argmax(prediction, dim=1) 
        
        prediction = prediction.data.cpu().numpy()
        output = output.data.cpu().numpy()
        
        y_true.append(output) # Save Truth 
        y_pred.append(prediction) # Save Prediction
        
        if count%100==0:
            print('Computing saliency for sample: ', count)
            
        saliency = compute_saliency_time(data, model, relative=True)
        saliencies.append(saliency)
        count += 1
        
        
    # classes = ('LoF pores', 'Conduction mode', 'Keyhole pores')
    # plotname= 'CNN_LSTM_MultivariateNormRel_'+str(window)+'_confusion_matrix'+'.png'
    # plt.figure()
    # plot_confusion_matrix(y_true, y_pred,classes,plotname)
    # torch.save(y_true, 'y_trueNormRel'+str(window))
    # torch.save(y_pred, 'y_predNormRel'+str(window))
    
    torch.save(y_true, 'data/y_trueNorm'+str(window))
    torch.save(y_pred, 'data/y_predNorm'+str(window))
    torch.save(saliencies, 'data/salienciesNorm'+str(window))
    
    return y_true,y_pred,saliencies

def compute_normalize_window(saliences,testset,folder,window):
    medianAmpl = []
    for datum in testset:
        singleDatum, _ = datum
        medianAmpl.append(torch.median(torch.abs(singleDatum), dim=2, keepdim=True)[0])
    medianAmpl = torch.cat(medianAmpl, dim=0)
    medianAmpl.shape
    # print(medianAmpl.shape)
    # print(saliences.shape)
    saliences = saliences/medianAmpl
    salienceCat = torch.median(saliences, dim=2, keepdim=False)[0]
    salienceCat
    salienceCat.shape
    title=str(folder)+'/'+'saliencies'+str(window)+'PerCat'
    torch.save(salienceCat, title)

    return salienceCat