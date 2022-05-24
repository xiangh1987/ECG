# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:27:48 2021

@author: bjorn

Traininer and Evaluation loops functioning with WandB
"""
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
from model_utils import get_rri, plot_grad_flow

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
    random_vals = np.random.normal(loc=0, scale=0.1, size=original_signal_length)
    blur_signal[:original_signal_length] += random_vals
    return blur_signal

def train(args, model, optimizer, criterion, device, train_loader):
    model.train() # Turn on the train mode
    
    #model.load_state_dict(torch.load('best_test.pth'))
    #print("load pretrained model...")
    
    total_loss = 0.
    batch_nr = 0
    start_time = time.time()
    # src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # data, targets = get_batch(train_data, i)
    
    print("data augmenting..")
    for batch in tqdm(train_loader):   
        batch_nr += 1
        data, targets = batch
        data = data[:,:,0]
        # #---aug---        
        res = []
        for i in range(len(data)):
            aug_np = data[i].numpy()
            foo = ['original', 'rotated', 'stretched', 'blurred']
            pick = random.choice(foo)
            if pick == 'rotated':
                rotate_sig = random_rotate(aug_np)
                res.append(rotate_sig)
            elif pick == 'stretched':
                stretch_sig = random_stretch(aug_np)
                res.append(stretch_sig)
            elif pick == 'blurred':
                blur_sig = gaussian_blur(aug_np)
                res.append(blur_sig)
            else:
                res.append(aug_np)
        data = torch.tensor(res)
        #---aug---
        
        # plt.plot(data[0,:])
        # plt.show()
        rri = get_rri(data)
        data, rri, targets = torch.tensor(data, dtype=torch.float, device=device), torch.tensor(rri, dtype=torch.float, device=device), torch.tensor(targets, dtype=torch.float, device=device)
        optimizer.zero_grad()
        # if data.size(0) != bptt:
        #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        # print(data.shape)
        # print(src_mask.shape)
        # output = model(data) # if n_class>2, and use cross entropy loss
        output = model(data, rri)[:,0] # if n_class==2
        # output = torch.argmax(output, dim=1)
        # output = torch.tensor(output, dtype=torch.float, device=device, requires_grad=True)
        # print('output:', output)
        # print('targets:', targets)
        loss = criterion(output, targets)
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0 , norm_type=2)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 50
        if batch_nr % log_interval == 0 and batch_nr > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            # print('| epoch', epoch, 
            #       '| train loss', np.round(cur_loss, 4),
            #       '| ms/batch', np.round(elapsed*1000/log_interval, 3),
            #       '| lr', np.round(scheduler.get_last_lr()[0], 4)
            #       )
            total_loss = 0.
            start_time = time.time()

    # scheduler.step()
    wandb.log({
    "Train Loss": cur_loss})
    return model, cur_loss

def evaluate(args, eval_model, data_source, criterion, device):
    true_label = np.array([])
    predictions = np.array([])
    loss_list = []
    eval_model.eval() # Turn on the evaluation mode
    tot_val_loss = 0.
    val_batch_nr = 0
    # src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        # for i in range(0, data_source.size(0) - 1, bptt):
        #     data, targets = get_batch(data_source, i)
        for batch in data_source:
            data, targets = batch
            data = data[:,:,0]
            rri = get_rri(data)
            data, rri, targets = torch.tensor(data, dtype=torch.float, device=device), torch.tensor(rri, dtype=torch.float, device=device), torch.tensor(targets, dtype=torch.float, device=device)
            # if data.size(0) != bptt:
            #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, rri)[:,0]
            # output = torch.argmax(output, dim=1)
            # output = torch.tensor(output, dtype=torch.float, device=device, requires_grad=True)
            loss = criterion(output, targets)
            tot_val_loss += loss.item()
            val_batch_nr+=1
            preds = np.round(torch.sigmoid(output).cpu().detach())
            # print('val preds:', preds)
            # print('val targets:', targets.cpu().detach())
            predictions = np.append(predictions, preds)
            true_label = np.append(true_label, targets.cpu().detach())
    # Get losses and accuracy
    cm = confusion_matrix(true_label, predictions, labels=[0,1])
    acc = np.sum(np.diag(cm))/np.sum(cm)
    TN, FP, FN, TP = cm.ravel()
    # FP = cm.sum(axis=0) - np.diag(cm)  
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = np.round(TP/(TP+FN), 4)
    TPR = TP/(TP+FN)
    # print(TPR.item())
    # print(type(TPR.item()))
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # print(TNR) 

    wandb.log({
    "Test Accuracy": 100*acc,
    "Test Sensitivity": 100*TPR,
    "Test Specificity": 100*TNR,
    "Test Loss": tot_val_loss/val_batch_nr})
    return tot_val_loss/val_batch_nr, cm, acc, TPR, TNR



