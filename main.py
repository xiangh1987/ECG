# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:32:24 2021

@author: bjorn

main script to run all training/eval scripts from
"""

import copy
import wandb
import torch
import torch.nn as nn
import time
import math
import numpy as np
import pandas as pd
import h5py
import base64
from data_loader import h5py_loader, torch_data_loader
from TransformerModel import TransformerModel
from train_eval import train, evaluate
from torch.utils.data import TensorDataset, DataLoader
from model_utils import get_rri

#%%
def parse_signal(signal):
    raw_adc_values = base64.b64decode(signal)
    raw_adc_values = np.array(list(raw_adc_values))
    magnitudes = np.bitwise_and(raw_adc_values, 0x7F)
    signs = np.where(raw_adc_values & 0x80 == 0x80, -1, 1)
    signal = magnitudes * signs
    signal = signal.astype(np.float32)
    return signal

def autocorr(x):
    n = x.size
    norm = (x-np.mean(x))
    result = np.correlate(norm, norm, mode='full')
    buff = 10
    acorr = result[(n+buff):]/(x.var()*np.arange(n-buff,1,-1))
    m = acorr.size
    acorr_half= acorr[:(m//2)]
    lag = acorr_half.argmax()
    r = acorr_half[lag]
    return acorr_half, r, lag

def corr(string):
    ecg_signal = parse_signal(string)
    step = 6000
    r_final = 1
    x_final = []
    temp_final = 0
    if len(ecg_signal) < 6000:
        ecg_signal = np.pad(ecg_signal, (0, 6000-len(ecg_signal)), 'constant')
    for temp in range(0,len(ecg_signal),step):
        x = ecg_signal[temp:(temp+step)]
        if len(x) < step:
            break
        x_acorr, r, lag = autocorr(x)
        if r < r_final:
            r_final = r
            x_final = ecg_signal[temp:(temp+step)]
            temp_final = temp
    x_final = np.asarray(x_final).astype(np.float32)
    return ecg_signal, x_final, r_final, temp_final

def corr_noparse(ecg_signal):
    #ecg_signal = parse_signal(string)
    step = 3000
    r_final = 1
    x_final = []
    temp_final = 0
    if len(ecg_signal) < 3000:
        ecg_signal = np.pad(ecg_signal, (0, 3000-len(ecg_signal)), 'constant')
    for temp in range(0,len(ecg_signal),step):
        x = ecg_signal[temp:(temp+step)]
        if len(x) < step:
            break
        x_acorr, r, lag = autocorr(x)
        if r < r_final:
            r_final = r
            x_final = ecg_signal[temp:(temp+step)]
            temp_final = temp
    x_final = np.asarray(x_final).astype(np.float32)
    return x_final, r_final, temp_final

def normalize(arr):
    centered = arr - np.mean(arr)
    norm_arr = np.array(centered/np.max(np.abs(centered)))
    return norm_arr

#%%
df_ecg = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
df_ecg['DecSignal'], df_ecg['CorrSignal'], df_ecg['Corr'], df_ecg['CorrIndex'] = zip(*df_ecg['Signal'].apply(corr))
df_ecg['NormSignal'] = df_ecg['CorrSignal'].apply(normalize)
X_N = []
for j in range(len(df_ecg)):
    X_N.append(df_ecg['NormSignal'][j])
X_N = np.array(X_N)
y_ecg = df_ecg['Label']
dg = pd.DataFrame(X_N)
dg['labels'] = y_ecg
gY = y_ecg.tolist()

df = dg
Y = gY

# #%%
# sample = df_ecg['Signal'][1]
# _, corr_sample, _, _ = corr(sample)
# norm_sample = normalize(corr_sample)
# norm_sample = norm_sample.reshape(1,len(norm_sample))
# tensor_sample = torch.from_numpy(norm_sample)
# rri_sample = get_rri(tensor_sample)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# data, rri = torch.tensor(norm_sample, dtype=torch.float, device=device), torch.tensor(rri_sample, dtype=torch.float, device=device)

#%%
import random
from matplotlib import pyplot as plt

def random_rotate(signal):
    rotate_signal = signal
    original_signal_length = len(rotate_signal)
    shift_amount = random.randint(0, original_signal_length-1)
    rotate_signal[:original_signal_length] = np.roll(rotate_signal[:original_signal_length], shift_amount)
    return rotate_signal

def random_stretch(signal):
    stretch_signal = signal
    original_signal_length = len(stretch_signal)
    scale_factor = random.uniform(1-0.5, 1+0.5)
    new_length = int(scale_factor*original_signal_length+1)
    scaled_signal = np.interp(np.linspace(0, original_signal_length, new_length), np.arange(original_signal_length), stretch_signal[:original_signal_length])
    stretch_signal = np.zeros(original_signal_length, dtype=np.float32)
    length = min(len(scaled_signal), original_signal_length)
    stretch_signal[:length] += scaled_signal[:length]
    return stretch_signal

def gaussian_blur(signal):
    blur_signal = signal
    original_signal_length = len(blur_signal)
    random_vals = np.random.normal(loc=0, scale=1, size=original_signal_length)
    blur_signal[:original_signal_length] += random_vals
    return blur_signal

# sig = parse_signal(sample)
# rotate_sig = random_rotate(sig)
# stretch_sig = random_stretch(sig)
# blur_sig = gaussian_blur(sig)

# plt.plot(sig[0:500])
# plt.figure()
# for iter in range(10):
#     aug_sig = random_stretch(sig, len(sig))
#     print(aug_sig)
#     plt.plot(aug_sig[0:500])
#     plt.figure()

#%%
# df_ecg = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
# df_ecg['DecSignal'], df_ecg['CorrSignal'], df_ecg['Corr'], df_ecg['CorrIndex'] = zip(*df_ecg['Signal'].apply(corr))
# df_ecg['NormSignal'] = df_ecg['CorrSignal'].apply(normalize)
# X_N = []
# for j in range(len(df_ecg)):
#     X_N.append(df_ecg['NormSignal'][j])
# X_N = np.array(X_N)
# y_ecg = df_ecg['Label']
# dg = pd.DataFrame(X_N)
# dg['labels'] = y_ecg
# gY = y_ecg.tolist()

# df_ecg1 = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
# df_ecg1['DecSignal'], df_ecg1['CorrSignal'], df_ecg1['Corr'], df_ecg1['CorrIndex'] = zip(*df_ecg1['Signal'].apply(corr))
# df_ecg1['DecSignal'] = df_ecg1['DecSignal'].apply(random_rotate)
# df_ecg1['CorrSignal_noparse'], df_ecg1['Corr_noparse'], df_ecg1['CorrIndex_noparse'] = zip(*df_ecg1['DecSignal'].apply(corr_noparse))
# df_ecg1['NormSignal_noparse'] = df_ecg1['CorrSignal_noparse'].apply(normalize)
# X_N1 = []
# for j in range(len(df_ecg1)):
#     X_N1.append(df_ecg1['NormSignal_noparse'][j])
# X_N1 = np.array(X_N1)
# y_ecg1 = df_ecg1['Label']
# dg1 = pd.DataFrame(X_N1)
# dg1['labels'] = y_ecg1
# gY1 = y_ecg1.tolist()

# df_ecg2 = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
# df_ecg2['DecSignal'], df_ecg2['CorrSignal'], df_ecg2['Corr'], df_ecg2['CorrIndex'] = zip(*df_ecg2['Signal'].apply(corr))
# df_ecg2['DecSignal'] = df_ecg2['DecSignal'].apply(random_stretch)
# df_ecg2['CorrSignal_noparse'], df_ecg2['Corr_noparse'], df_ecg2['CorrIndex_noparse'] = zip(*df_ecg2['DecSignal'].apply(corr_noparse))
# df_ecg2['NormSignal_noparse'] = df_ecg2['CorrSignal_noparse'].apply(normalize)
# X_N2 = []
# for j in range(len(df_ecg2)):
#     X_N2.append(df_ecg2['NormSignal_noparse'][j])
# X_N2 = np.array(X_N2)
# y_ecg2 = df_ecg2['Label']
# dg2 = pd.DataFrame(X_N2)
# dg2['labels'] = y_ecg2
# gY2 = y_ecg2.tolist()

# df_ecg3 = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
# df_ecg3['DecSignal'], df_ecg3['CorrSignal'], df_ecg3['Corr'], df_ecg3['CorrIndex'] = zip(*df_ecg3['Signal'].apply(corr))
# df_ecg3['DecSignal'] = df_ecg3['DecSignal'].apply(gaussian_blur)
# df_ecg3['CorrSignal_noparse'], df_ecg3['Corr_noparse'], df_ecg3['CorrIndex_noparse'] = zip(*df_ecg3['DecSignal'].apply(corr_noparse))
# df_ecg3['NormSignal_noparse'] = df_ecg3['CorrSignal_noparse'].apply(normalize)
# X_N3 = []
# for j in range(len(df_ecg3)):
#     X_N3.append(df_ecg3['NormSignal_noparse'][j])
# X_N3 = np.array(X_N3)
# y_ecg3 = df_ecg3['Label']
# dg3 = pd.DataFrame(X_N3)
# dg3['labels'] = y_ecg3
# gY3 = y_ecg3.tolist()

# df = dg.append(dg1)
# df = df.append(dg2)
# df = df.append(dg3)
# Y = []
# Y.extend(gY)
# Y.extend(gY1)
# Y.extend(gY2)
# Y.extend(gY3)

#%%

#load data
# af_data_path = '/Users/xianghe/Downloads/AF-classification-master/af_full.h5'
# normal_data_path = '/Users/xianghe/Downloads/AF-classification-master/normal_full.h5'
# af_array = h5py_loader(af_data_path, 'af_tot')
# normal_array = h5py_loader(normal_data_path, 'normal_tot')

# af_array = af_array.iloc[:,:]*2-1
# af_array['labels'] = 1
# normal_array = normal_array.iloc[:,:]*2-1
# normal_array['labels'] = 0

# df1 = torch_data_loader(normal_array, 0)[0]
# df2 = torch_data_loader(af_array, 1)[0]

# df1 = normal_array
# df2 = af_array

# dff = df1.append(df2) # combine to one df
# Y1 = [0]*len(df1) # Normal
# Y2 = [1]*len(df2) # AF
# YY = []
# YY.extend(Y1)
# YY.extend(Y2)

# df = df.append(dff)
# Y.extend(YY)
#%%

# df_ecg = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
# alldata = df_ecg['Signal'].values
# y_ecg = df_ecg['Label']
# X_ecg = []
# for i in alldata:
#     signal = parse_signal(i)
#     center = round(len(signal)/2)
#     an_array = signal[center-1000:center+1000]
#     norm = np.linalg.norm(an_array)
#     normal_array = an_array/norm
#     X_ecg.append([normal_array])
        
#     #center = round(len(signal)/2)
#     #X_ecg.append([signal[center-1000:center+1000]])
# X_ecg = np.stack(X_ecg, axis=0)
# newX_ecg = X_ecg.reshape(len(X_ecg),1*2000)
# df = pd.DataFrame(newX_ecg)
# df['labels'] = y_ecg
# Y = y_ecg.tolist()

# # load data
# af_data_path = '/Users/xianghe/Downloads/AF-classification-master/af_full.h5'
# normal_data_path = '/Users/xianghe/Downloads/AF-classification-master/normal_full.h5'
# af_array = h5py_loader(af_data_path, 'af_tot')
# normal_array = h5py_loader(normal_data_path, 'normal_tot')
# df1 = torch_data_loader(normal_array, 0)[0]
# df2 = torch_data_loader(af_array, 1)[0]

# df1 = normal_array
# df2 = af_array

# df = df1.append(df2) # combine to one df
# Y1 = [1]*len(df1) # Normal
# Y2 = [0]*len(df2) # AF
# Y = []
# Y.extend(Y1)
# Y.extend(Y2)
# del df1, df2, Y1, Y2 # delete individual dataframes if needed

#%%

train_loader, val_loader = torch_data_loader(df, Y, train_size=0.9, batch_size=10, random_state=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set to gpu if possible

'''
def main():
  #model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
  # criterion = nn.CrossEntropyLoss()
  model = TransformerModel(64,4,256,4,2,2,0.25,0.1).to(device)
  criterion = nn.BCEWithLogitsLoss() # pass logits as input (not probabilities)
  # criterion = nn.BCELoss() # pass probabilities (using sigmoid) as input
  best_val_loss = float("inf")
  epochs = 400 # The number of epochs
  best_model = None
  lr = 0.0001 # learning rate
  # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98)) #weight_decay=1e-6
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)
  # WandB – "wandb.watch()" automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
  #wandb.watch(model, log="all")
  history = dict(train=[], val=[], val_acc=[], val_sens=[], val_spec=[])

  for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      model, train_loss = train(model, optimizer, criterion, device, train_loader)
      val_loss, cm, val_acc, val_sens, val_spec, val_f1 = evaluate(eval_model=model, data_source=val_loader, criterion=criterion, device=device)
      epoch_time = time.time() - epoch_start_time
      # history['train'].append(train_loss)
      history['val'].append(val_loss)
      history['val_acc'].append(val_acc)
      history['val_sens'].append(val_sens)
      history['val_spec'].append(val_spec)
      #wandb.log({"Epoch Time [s]": epoch_time}) # log time for epoch
      # table_sens = wandb.Table(data=history['val_sens'], columns=["Sensitivity"])
      # wandb.log({"Test Sensitivity" : wandb.plot.line(table_sens, "Sensitivity", title="Sensitivity over epochs")})
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, epoch_time,
                                      val_loss, math.exp(val_loss)))
      print('-' * 89)
      print('valid conf mat:\n', cm)
      print('valid f1 score:', val_f1)
      print('valid accuracy:', np.round(val_acc, 4))
      print('-' * 89)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
      if epoch > 30:
        scheduler.step()
      if epoch == 50:
          best_model_wts50 = copy.deepcopy(best_model.state_dict())
          torch.save(best_model_wts50, '/home/arstanbek/AF-classification-master/best_aug_50.pth')
      if epoch == 100:
          best_model_wts50 = copy.deepcopy(best_model.state_dict())
          torch.save(best_model_wts50, '/home/arstanbek/AF-classification-master/best_aug_100.pth')
      if epoch == 200:
          best_model_wts50 = copy.deepcopy(best_model.state_dict())
          torch.save(best_model_wts50, '/home/arstanbek/AF-classification-master/best_aug_200.pth')
  return best_model
'''

# WandB - Initialize run
wandb.init(entity="rockhx1987", project="ecg")
# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 10          # input batch size for training (default: 64)
# config.test_batch_size = batch_size    # input batch size for testing (default: 1000)
config.epochs = 2             # number of epochs to train (default: 10)
config.lr = 0.0001               # learning rate (default: 0.01)
# config.momentum = 0.1          # SGD momentum (default: 0.5) 
# config.seed = 42               # random seed (default: 42)
config.log_interval = 1     # how many batches to wait before logging training status
config.emsize = 64 # embedding dimension == d_model
config.dim_feedforward = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
config.nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
config.nhead = 4 # the number of heads in the multiheadattention models
config.n_conv_layers = 2 # number of convolutional layers (before transformer encoder)
config.dropout = 0.25 # the dropout value
config.dropout_other = 0.1 # dropout value for feedforward output layers
config.n_class = 2
config.use_rri_features = True         # using raw signal + rri features during training
def main():
  model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
  # criterion = nn.CrossEntropyLoss()
  criterion = nn.BCEWithLogitsLoss() # pass logits as input (not probabilities)
  # criterion = nn.BCELoss() # pass probabilities (using sigmoid) as input
  best_val_loss = float("inf")
  epochs = config.epochs # The number of epochs
  best_model = None
  lr = config.lr # learning rate
  # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98)) #weight_decay=1e-6
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)
  # WandB – "wandb.watch()" automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
  wandb.watch(model, log="all")
  history = dict(train=[], val=[], val_acc=[], val_sens=[], val_spec=[])
  
  for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      model, train_loss = train(config, model, optimizer, criterion, device, train_loader)
      val_loss, cm, val_acc, val_sens, val_spec = evaluate(args=config, eval_model=model, data_source=val_loader, criterion=criterion, device=device)
      epoch_time = time.time() - epoch_start_time
      # history['train'].append(train_loss)
      history['val'].append(val_loss)
      history['val_acc'].append(val_acc)
      history['val_sens'].append(val_sens)
      history['val_spec'].append(val_spec)
      wandb.log({"Epoch Time [s]": epoch_time}) # log time for epoch
      # table_sens = wandb.Table(data=history['val_sens'], columns=["Sensitivity"])
      # wandb.log({"Test Sensitivity" : wandb.plot.line(table_sens, "Sensitivity", title="Sensitivity over epochs")})
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, epoch_time,
                                      val_loss, math.exp(val_loss)))
      print('-' * 89)
      print('valid conf mat:\n', cm)
      print('valid accuracy:', np.round(val_acc, 4))
      print('-' * 89)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
      if epoch > 30:
        scheduler.step()  

  return best_model        

if __name__ == '__main__':
  best_model = main()     

# save model 
best_model_wts = copy.deepcopy(best_model.state_dict())
torch.save(best_model_wts, '/Users/xianghe/Downloads/AF-classification-master/best_aug.pth')

