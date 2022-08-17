# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Main file to execute the model on the LPBF dataset
"""

#%%
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split# implementing train-test-split

#%% Normalize the dataset

def normalize(Features):
    Features_1=np.load(Features)
    df = pd.DataFrame(Features_1)  
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df = df.to_numpy()
    return df
#%% Convert into torch tensor

class Mechanism(Dataset):
    
    def __init__(self,sequences):
        self.sequences = sequences
    
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self,idx):
        
        sequence_1,sequence_2,sequence_3,sequence_4,label =  self.sequences [idx]
        sequence_1=torch.Tensor(sequence_1)
        sequence_2=torch.Tensor(sequence_2)
        sequence_3=torch.Tensor(sequence_3)
        sequence_4=torch.Tensor(sequence_4)
        
        sequence1 = sequence_1.view(1, -1)
        sequence2 = sequence_2.view(1, -1)
        sequence3 = sequence_3.view(1, -1)
        sequence4 = sequence_4.view(1, -1)
        
        
        sequence=torch.cat((sequence1, sequence2,sequence3, sequence4), 0)
        
        # print("sequence",sequence.shape)
        label=torch.tensor(label).long()
        # sequence,label
        return sequence,label
    
#%% 
def data_prep(Rawspace_1,Rawspace_2,Rawspace_3,Rawspace_4,classspace):
    
    sequences=[]
    for i in range(len(classspace)):
        # print(i)
        sequence_features_1 = Rawspace_1[i]
        sequence_features_2 = Rawspace_2[i]
        sequence_features_3 = Rawspace_3[i]
        sequence_features_4 = Rawspace_4[i]
        
        label = classspace[i]
        sequences.append((sequence_features_1,sequence_features_2,sequence_features_3,sequence_features_4,label))
    return sequences


#%%  Loading the data for training the model
def dataloading_funtion(folder,window):
    
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
    trainset,testset=data_batch_prep(Rawspace_1,Rawspace_2,Rawspace_3,Rawspace_4,classspace)
    
    return trainset,testset,classspace





#%%

def data_batch_prep(Rawspace_1,Rawspace_2,Rawspace_3,Rawspace_4,classspace):
    
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
    train, test = train_test_split(sequences_batch, test_size=0.3,random_state=42)
    trainset = torch.utils.data.DataLoader(train, batch_size=200, num_workers=0,
                                            shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=200, num_workers=0,
                                            shuffle=True)
      
    return trainset, testset


