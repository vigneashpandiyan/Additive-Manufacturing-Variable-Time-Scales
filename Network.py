# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Network model
"""

#%%

from torch import nn, optim
from torch.nn import functional as F
from prettytable import PrettyTable

class CNN(nn.Module): 
    
    def __init__(self, n_features, n_classes, n_hidden=90, n_layers=1): 
        super().__init__() 
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16) 
        self.bn1 = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16) 
        self.bn2 = nn.BatchNorm1d(16) 
        
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16) 
        self.bn3 = nn.BatchNorm1d(32) 
        
        self.conv4 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn4 = nn.BatchNorm1d(64) 
        
        self.conv5 = nn.Conv1d(64, out_channels=n_features, kernel_size=16)
        self.bn5 = nn.BatchNorm1d(n_features)
        
        self.pool = nn.MaxPool1d(2)
    
        self.dropout = nn.Dropout(0.025)
        self.lstm = nn.LSTM(input_size=n_features, 
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True, 
                dropout=0.2 
                )
        self.classifier = nn.Linear(n_hidden, n_classes) 
        
                
    def forward(self, x): 
        
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.shape)
        x = self.pool(x) 
        # print(x.shape)
        
        # torch.Size([1, 4, 5000])--> torch.Size([1, 10, 141])
        # torch.Size([1, 4, 10000])-->torch.Size([1, 10, 297])
       
        x = x.permute(0, 2, 1)
       
        # print(x.shape) #torch.Size([100, 5000, 1])

        x,(_,hidden, ) = self.lstm(x) 
        out = hidden[-1] 
        
        return self.classifier(out) 

#%%


def initialize_weights(m):
  if isinstance(m, nn.Conv1d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
#%%


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']
    
#%%
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params