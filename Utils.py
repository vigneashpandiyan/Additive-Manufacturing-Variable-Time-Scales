# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Utils file for visualization/ Plots
"""

#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

#%%

def plot_confusion_matrix(y_true, y_pred,classes,plotname):
            
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float')  / cm.sum(axis=1)[:, np.newaxis]
    cmn=cmn*100
    
    fig, ax = plt.subplots(figsize=(12,9))
    sns.set(font_scale=3) 
    b=sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes,cmap="coolwarm",linewidths=0.1,annot_kws={"size": 25},cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts: b.set_text(b.get_text() + " %")
    plt.ylabel('Actual',fontsize=25)
    plt.xlabel('Predicted',fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize= 20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center",fontsize= 20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname=str(plotname)
    plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    plt.clf()

#%%

def plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std,class_names,Times):
    
    
    iteration = np.array(iteration)
    Loss_value = np.array(Loss_value)
    Total_Epoch = np.array(Total_Epoch)
    Accuracy = np.array(Accuracy)
    Learning_rate = np.array(Learning_rate)
    Training_loss_mean = np.array(Training_loss_mean)
    Training_loss_std = np.array(Training_loss_std)
    Times = np.array(Times)
    
    Accuracyfile = 'Accuracy'+'.npy'
    Lossfile = 'Loss_value'+'.npy'
    Timesfile = 'Times'+'.npy'
    
    np.save(Timesfile,Times,allow_pickle=True)
    np.save(Accuracyfile,Accuracy,allow_pickle=True)
    np.save(Lossfile,Loss_value, allow_pickle=True)
    
    
    fig, ax = plt.subplots()
    plt.plot(Loss_value,'r',linewidth =2.0)
    # ax.fill_between(Loss_value, Training_loss_mean - Training_loss_std, Training_loss_mean + Training_loss_std, alpha=0.9)
    plt.title('Iteration vs Loss_Value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss_Value')
    plot_1=  'Loss_value_'+ '.png'
    plt.savefig(plot_1, dpi=600,bbox_inches='tight')
    plt.show()
    plt.clf()
    
    plt.figure(2)
    plt.plot(Total_Epoch,Accuracy,'g',linewidth =2.0)
    plt.title('Total_Epoch vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plot_2=  'Accuracy_'+'.png'
    plt.savefig(plot_2, dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.figure(3)
    plt.plot(Total_Epoch,Learning_rate,'b',linewidth =2.0)
    plt.title('Total_Epoch vs Learning_Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_Rate')
    plot_3=  'Learning_rate_'+ '.png'
    plt.savefig(plot_3, dpi=600,bbox_inches='tight')
    plt.show()
    
    graphname='Iteration'+'_weightage'+'.png'
    fig, ax = plt.subplots(figsize=(7,5), dpi=100)
    ax = sns.countplot(Times,palette=["#fbab17", "#0515bf", "#10a310", "#e9150d"])
    ax.set_xticklabels(class_names);
    ax.xaxis.label.set_size(10)
    plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1,dpi=800)
    plt.show()
    plt.clf()




#%%

def classweight(values,counts):
    class_weight=[]
    tot=sum(counts)
    for group in counts:
        value=1-(group/tot)
        print(value)
        class_weight.append(value)
    class_weight = np.array(class_weight)
    return class_weight

#%%

def windowresults(testset,model,device,window):
    
    y_pred = []
    y_true = []
    model.eval()
    # iterate over test data
    for batches in testset:
        
        data,output = batches
        data,output =data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        output=output.squeeze()
        # print("output",output)
        prediction = model(data)
        
        prediction = torch.argmax(prediction, dim=1) 
        # print("prediction",prediction)
        prediction=prediction.data.cpu().numpy()
        output=output.data.cpu().numpy()
        
        y_true.extend(output) # Save Truth 
        y_pred.extend(prediction) # Save Prediction
        
        
    classes = ('LoF pores', 'Conduction mode', 'Keyhole pores')
    
    
    plotname= 'CNN_LSTM_Multivariate_'+str(window)+'_confusion_matrix'+'.png'
    
    plt.figure()
    plot_confusion_matrix(y_true, y_pred,classes,plotname)

#%%

def distribution_plot(data,window_length):
    
    data=data.cpu().detach().numpy() 
    df = pd.DataFrame(data, columns=['Back reflection', 'Infra red','Visible', 'Acoustic signal'])
    # df=df.div(df.sum(axis=1), axis=0)
    sns.set(style="white")
    fig=plt.subplots(figsize=(5,3), dpi=800)
    
    # sns.displot(data, kind="kde", multiple="stack",alpha=.5,)
    fig = sns.kdeplot(df['Back reflection'], shade=True,alpha=.5, color="red")
    fig = sns.kdeplot(df['Infra red'], shade=True,alpha=.5, color="green")
    fig = sns.kdeplot(df['Visible'], shade=True,alpha=.5, color="#0000FF")
    fig = sns.kdeplot(df['Acoustic signal'], shade=True, alpha=.5,color="#FFD700")
    plt.title("Saliency distribution across sensors \n for a window length of "+str(window_length)+" ms")
    plt.legend(labels=["Back reflection","Infra-red","Visible","Acoustic signal"])
    title=str(window_length)+'_'+'.png'
    plt.xlim([0.0, 0.35])
    plt.ylim([0.0, 100])
    plt.xlabel('Derivative relative amplitude (r.u)') 
    plt.savefig(title, bbox_inches='tight')
    plt.show()