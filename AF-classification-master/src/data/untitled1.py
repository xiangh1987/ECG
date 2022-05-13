#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:41:20 2022

@author: xianghe
"""
import h5py
import pandas as pd
# load af data
path = '/Users/xianghe/Downloads/AF-classification-master/af_full.h5' # for ex. '/content/drive/MyDrive/ecg_data_full/af_full.h5'
h5f = h5py.File(path,'r')
af_array = h5f['af_tot'][:]
h5f.close()
# load normal data
path = '/Users/xianghe/Downloads/AF-classification-master/normal_full.h5' # for ex. '/content/drive/MyDrive/ecg_data_full/normal_full.h5'
h5f = h5py.File(path,'r')
normal_array = h5f['normal_tot'][:]
h5f.close()
# can also load it to pd.DataFrame and drop any NaN values
df_af = pd.DataFrame(data=af_array)
df_af.dropna(inplace=True)
