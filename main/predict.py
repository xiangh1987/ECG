#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:54:57 2022

@author: xianghe
"""
#%%
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
import numpy as np
import base64
import math
from ecgdetectors import Detectors

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(1), :].squeeze(1)
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

def get_rri(sig, fs, n_beats): #fs=300
  """
  Input: 10s ECG signal (torch tensor)
  Output: RRI, time in between each beat (10 beats otherwise 0-padded)
  """
  detectors = Detectors(fs)
  sig_n = sig.numpy()  
  # if type(sig).__module__ != np.__name__:
  #  sig_n = sig.numpy()
  #else:
   # sig_n = sig
  # print(sig_n.shape)
  rri_list = []
  for i in range(sig_n.shape[0]):
    r_peaks = detectors.pan_tompkins_detector(sig_n[i])
    rri = np.true_divide(np.diff(r_peaks), fs)
    if len(rri) < n_beats:
      rri = np.pad(rri, (0, n_beats-len(rri)), 'constant', constant_values=(0))
    if len(rri) > n_beats:
      rri = rri[0:n_beats]
    rri_list.append(rri)
  
  rri_stack = np.stack(rri_list, axis=0)
  # print(rri_stack.shape) 
  return rri_stack  

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

def corr(string, stp):
    ecg_signal = parse_signal(string)
    step = stp
    r_final = 1
    x_final = []
    temp_final = 0
    if len(ecg_signal) < stp:
        ecg_signal = np.pad(ecg_signal, (0, stp-len(ecg_signal)), 'constant')
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
class TransformerModel(nn.Module):

    def __init__(self, dim_position, d_model, nhead, dim_feedforward, nlayers, n_conv_layers=2, n_class=2, dropout=0.5, dropout_other=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.relu = torch.nn.ReLU()
        self.pos_encoder = PositionalEncoding(dim_position, dropout) # original is 748 
        self.pos_encoder2 = PositionalEncoding(6, dropout)
        self.self_att_pool = SelfAttentionPooling(d_model)
        self.self_att_pool2 = SelfAttentionPooling(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.flatten_layer = torch.nn.Flatten()
        # Define linear output layers
        if n_class == 2:
          self.decoder = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                       nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        # else:
        #   self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(0.1),
        #                                nn.Linear(d_model, d_model), nn.Dropout(0.1), 
        #                                nn.Linear(d_model, n_class))
        if n_class == 2:
          self.decoder2 = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                      #  nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        # Linear output layer after concat.
        self.fc_out1 = torch.nn.Linear(64+64, 64)
        self.fc_out2 = torch.nn.Linear(64, 1) # if two classes problem is binary  
        # self.init_weights()
        # Transformer Conv. layers
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(d_model)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.1)
        # self.avg_maxpool = nn.AdaptiveAvgPool2d((64, 64))
        # RRI layers
        self.conv1_rri = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.conv2_rri = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3) 

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src2):      
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # size input: [batch, sequence, embedding dim.]
        # src = self.pos_encoder(src) 
        # print('initial src shape:', src.shape)
        src = src.view(-1, 1, src.shape[1]) # Resize to --> [batch, input_channels, signal_length]
        src = self.relu(self.conv1(src))
        src = self.relu(self.conv2(src))
        # src = self.maxpool(self.relu(src))
        # print('src shape after conv1:', src.shape)
        for i in range(self.n_conv_layers):
          src = self.relu(self.conv(src))
          src = self.maxpool(src)

        # src = self.maxpool(self.relu(src))
        src = self.pos_encoder(src)   
        # print(src.shape) # [batch, embedding, sequence]
        src = src.permute(2,0,1) # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        # print('src shape:', src.shape)
        output = self.transformer_encoder(src) # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        # print('output shape 1:', output.shape)
        # output = self.avg_maxpool(output)
        # output = torch.mean(output, dim=0) # take mean of sequence dim., output: [batch, embedding dim.] 
        output = output.permute(1,0,2)
        output = self.self_att_pool(output)
        # print('output shape 2:', output.shape)
        logits = self.decoder(output) # output: [batch, n_class]
        # print('output shape 3:', logits.shape)
        # output_softmax = nn.functional.softmax(logits, dim=1) # get prob. of logits dim.  # F.log_softmax(output, dim=0)
        # output = torch.sigmoid(output)
        # RRI layers
        src2 = src2.view(-1, 1, src2.shape[1]) # Resize to --> [batch, input_channels, signal_length]
        src2 = self.relu(self.conv1_rri(src2))
        src2 = self.relu(self.conv2_rri(src2))
        src2 = self.pos_encoder2(src2)  
        src2 = src2.permute(2,0,1) # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        output2 = self.transformer_encoder2(src2)
        output2 = output2.permute(1,0,2)
        output2 = self.self_att_pool2(output2)
        logits2 = self.decoder2(output2) # output: [batch, n_class]
        logits_concat = torch.cat((logits, logits2), dim=1)
        # Linear output layer after concat.
        xc = self.flatten_layer(logits_concat)
        # print('shape after flatten', xc.shape)
        xc = self.fc_out2(self.dropout(self.relu(self.fc_out1(xc)))) 

        return xc

#%%
from collections import Counter

res = []
def classify(signal):
    corr_sample, _, _ = corr(signal, 6000)
    norm_sample = normalize(corr_sample)
    norm_sample = norm_sample.reshape(1,len(norm_sample))
    tensor_sample = torch.from_numpy(norm_sample)
    rri_sample = get_rri(tensor_sample, 128, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    data, rri= torch.tensor(norm_sample, dtype=torch.float, device=device), torch.tensor(rri_sample, dtype=torch.float, device=device)
    the_model = TransformerModel(1498, 64, 4, 256, 4, 2, 2, 0.25, 0.1).to(device)
    the_model.load_state_dict(torch.load('best_fs128_96.pth', map_location=device))
    the_model.eval()
    with torch.no_grad():
        output = the_model(data, rri)[:,0]
        preds = np.round(torch.sigmoid_(output).cpu().detach())
    prediction = int(preds.item())
    print("prediction: ", prediction)
    res.append(prediction)
    
    corr_sample, _, _ = corr(signal, 3000)
    norm_sample = normalize(corr_sample)
    norm_sample = norm_sample.reshape(1,len(norm_sample))
    tensor_sample = torch.from_numpy(norm_sample)
    rri_sample = get_rri(tensor_sample, 300, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    data, rri= torch.tensor(norm_sample, dtype=torch.float, device=device), torch.tensor(rri_sample, dtype=torch.float, device=device)
    the_model = TransformerModel(748, 64, 4, 256, 4, 2, 2, 0.25, 0.1).to(device)
    the_model.load_state_dict(torch.load('checkpoint_100.pth', map_location=device))
    the_model.eval()
    with torch.no_grad():
        output = the_model(data, rri)[:,0]
        preds = np.round(torch.sigmoid(output).cpu().detach())
    prediction = int(preds.item())
    print("prediction1: ", prediction)
    res.append(prediction)

    corr_sample, _, _ = corr(signal, 3000)
    norm_sample = normalize(corr_sample)
    norm_sample = norm_sample.reshape(1,len(norm_sample))
    tensor_sample = torch.from_numpy(norm_sample)
    rri_sample = get_rri(tensor_sample, 300, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    data, rri= torch.tensor(norm_sample, dtype=torch.float, device=device), torch.tensor(rri_sample, dtype=torch.float, device=device)
    the_model = TransformerModel(748, 64, 4, 256, 4, 2, 2, 0.25, 0.1).to(device)
    the_model.load_state_dict(torch.load('best_aug_200_arstan.pth', map_location=device))
    the_model.eval()
    with torch.no_grad():
        output = the_model(data, rri)[:,0]
        preds = np.round(torch.sigmoid(output).cpu().detach())
    prediction = int(preds.item())
    print("prediction2: ", prediction)
    res.append(prediction)
    
    final = Counter(res).most_common()[0][0]
    return final

df_ecg = pd.read_csv("/Users/xianghe/Desktop/hackathon_dataset/hackathon_dataset.csv")
fi = classify(df_ecg['Signal'][1])
print("predict: ", fi)
    
    
    

