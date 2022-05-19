#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:54:57 2022

@author: xianghe
"""
#%%
import torch
import torch.nn as nn
from TransformerModel import TransformerModel
import pandas as pd
import numpy as np
import base64
from model_utils import get_rri

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
sample = df_ecg['Signal'][1]
corr_sample, _, _ = corr(sample)
norm_sample = normalize(corr_sample)
norm_sample = norm_sample.reshape(1,len(norm_sample))
tensor_sample = torch.from_numpy(norm_sample)
rri_sample = get_rri(tensor_sample)
tensor_rri = torch.from_numpy(rri_sample)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
data, rri= torch.tensor(norm_sample, dtype=torch.float, device=device), torch.tensor(rri_sample, dtype=torch.float, device=device)
#%%
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
the_model = TransformerModel(64, 4, 256, 4, 2, 2, 0.25, 0.1).to(device)
the_model.load_state_dict(torch.load('best_test.pth', map_location=device))
the_model.eval()

true_label = np.array([])
predictions = np.array([])
loss_list = []
tot_val_loss = 0.
val_batch_nr = 0
with torch.no_grad():
    output = the_model(data, rri)[:,0]
    preds = np.round(torch.sigmoid(output).cpu().detach())
prediction = int(preds.item())
print("prediction: ", prediction)


