#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:49:30 2018

@author: sungkyun
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from live_dataloader import LJSpeechDataset
from utils.AttentionGuide import AttentionGuideGen
from tqdm import tqdm
from FastTacotron import Text2Mel

torch.backends.cudnn.benchmark=True

#%% Parsing arguments
parser = argparse.ArgumentParser(description='Fast Tacotron implementation')
parser.add_argument('-exp', '--exp_name', type=str, default='00', metavar='STR',
                    help='Generated samples will be located in the checkpoints/exp<exp_name> directory. Default="00"') # 
parser.add_argument('-e', '--max_epoch', type=int, default=10000, metavar='N',
                    help='Max epoch, Default=10000') 
parser.add_argument('-btr', '--batch_train', type=int, default=16, metavar='N',
                    help='Batch size for training. e.g. -btr 16')
parser.add_argument('-bts', '--batch_test', type=int, default=1, metavar='N',
                    help='Batch size for test. e.g. -bts 1')
parser.add_argument('-load', '--load', type=str, default=None, metavar='STR',
                    help='e.g. --load checkpoints/<expname>/checkpoint_00')
parser.add_argument('-sint', '--save_interval', type=int, default=10, metavar='N',
                    help='Save interval., default=100')
parser.add_argument('-disp', '--sel_display', type=int, default=9, metavar='N',
                    help='Selection of data for display., default=9')
#parser.add_argument('-g', '--gpu_id', type=str, default=None, metavar='STR',
#                    help='Multi GPU ids to use')
args = parser.parse_args()

USE_GPU = torch.cuda.is_available()
RAND_SEED  = 0
CHECKPOINT_DIR = 'checkpoints/' + args.exp_name

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#%%
#for batch_idx, (data_idx, x_text , x_melspec, zs) in enumerate(train_loader):
#    if batch_idx is 2:
#        break
#x_text = Variable(x_text.long(), requires_grad=False)
#x_melspec = Variable(x_melspec.float(), requires_grad=False)


def print_model_sz(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Number of trainable parameters = {}'.format(sum([np.prod(p.size()) for p in model_parameters])) )

def display_spec(dt1, dt2, title='unknown_spec'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [10,5]
    sns.set(font_scale=.7)
    
    plt.subplot(211)
    plt.pcolormesh(dt1, cmap='jet')
    
    plt.subplot(212)
    plt.pcolormesh(dt2, cmap='jet')
    
    plt.title(title); plt.xlabel('Mel-spec frames')
    plt.savefig('images/' + title + '_mspec.png', bbox_inches='tight', dpi=220)
    plt.close('all')
    
    
def display_att(dt, title='unknown_att'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [7,7]
    sns.set(font_scale=.7)   
    plt.pcolormesh(dt, cmap='bone')   
    plt.title(title)
    plt.xlabel('Mel-spec frames'); plt.ylabel('Text characters')
    plt.savefig('images/' + title + '_att.png', bbox_inches='tight', dpi=220)
    plt.close('all')    


def load_checkpoint(filepath):
    '''  Load pre-trained model. '''
    dt = torch.load(filepath)
    print('Loading from epoch{}...'.format(dt['epoch']) )
    model.load_state_dict(dt['state_dict'])
    optimizer.load_state_dict(dt['optimizer'])
    return dt['epoch']

def save_checkpoint(state, exp_name):
    filepath = CHECKPOINT_DIR + '/checkpoint{}.pth.tar'.format(state['epoch'])
    torch.save(state, filepath)
    
    
#%% Data Loading
DATA_ROOT = '/mnt/ssd3/data/LJSpeech-1.1'

dset_train = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=True,  output_mode='melspec')
dset_test  = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=False, output_mode='melspec')
#dset_train = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=True,  output_mode='SSRN')
#dset_test  = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=False, output_mode='SSRN')
train_loader = DataLoader(dset_train,
                          batch_size=args.batch_train,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True
                         ) # number of CPU threads, practically, num_worker = 4 * num_GPU

test_loader = DataLoader(dset_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                         )


#%% Train
USE_GPU = True
if USE_GPU:
    model = Text2Mel().cuda()
else:
    model = Text2Mel().cpu()
    
optimizer = torch.optim.Adam(model.parameters(), lr=2e-04, betas=(0.5, 0.9), eps=1e-06)
print_model_sz(model)
last_epoch = 0

guide_generator = AttentionGuideGen()

loss_L1 = nn.L1Loss(size_average=True, reduce=True)
loss_BCE = nn.BCELoss(weight=None, size_average=True, reduce=True)


# Training...
def train(epoch):
    model.train()
    train_loss = {'Total':0., 'L1':0., 'BCE':0., 'Att':0.}
    #total_data_sz = train_loader.dataset.__len__()
    
    for batch_idx, (data_idx, x_text , x_melspec, zs) in tqdm(enumerate(train_loader)):
        if USE_GPU:
            x_text, x_melspec = Variable(x_text.cuda().long()), Variable(x_melspec.cuda().float())
        else:
            x_text, x_melspec = Variable(x_text.long()), Variable(x_melspec.float())
#        if batch_idx is not None:
#            break
        
        
        optimizer.zero_grad()
        out_y, out_att = model(x_text, x_melspec)
        
        l1 = loss_L1(out_y[:,:,:-1], x_melspec[:,:,1:]) 
        l2 = loss_BCE(out_y[:,:,:-1], x_melspec[:,:,1:])
        
        # l3: Attention loss, W is guide matrices with BxNxT
        W = guide_generator.get_padded_guide(target_sz=(251,218),
                                             pad_sz=(zs).data.cpu().numpy(),
                                             set_silence_state=-1)
        W_sz = W.size
        W = torch.cuda.FloatTensor(W) if USE_GPU else torch.FloatTensor(W).cpu()
        l3 = torch.sum(out_att * W) / W_sz # Normalization
        
        loss = l1 + l2 + l3
        
        loss.backward()
        optimizer.step()
#        print('Train Epoch: {} [{}/{}], Total loss={:.6f}, L1={:.6f}, BCE={:.6f}, A={:.6f}'.format(
#                epoch, batch_idx * train_loader.batch_size, total_data_sz, 
#                loss.item(), l1.item(), l2.item(), 0.))
        train_loss['Total'] += loss.item()
        train_loss['L1'] += l1.item()
        train_loss['BCE'] += l2.item()
        train_loss['Att'] += l3.item()
        
    
        select_data = args.sel_display
        if ((epoch in [1,3,5,10,20,30,40]) | (epoch%50 is 0)) & (select_data in data_idx ):
            sel = np.where(data_idx.cpu()==select_data)[0].data[0]
            
            out_y_cpu = (out_y[sel,:,:]).data.cpu().numpy()
            out_att_cpu = (out_att[sel,:,:]).data.cpu().numpy()
            #org_text  = (x_text[sel,:]).data.cpu().numpy()
            org_melspec =(x_melspec[sel,:,:]).data.cpu().numpy()
            
            display_spec(out_y_cpu, org_melspec, 'Sample {}: epoch = {}'.format(select_data, epoch))
            display_att(out_att_cpu, 'Sample {}: epoch = {}'.format(select_data, epoch))
        
    return train_loss

#%% Train Main Loop
df_hist = pd.DataFrame(columns=('Total', 'L1', 'BCE','Att'))
last_epoch = 0

if args.load is not None:
    last_epoch = load_checkpoint(args.load)
    aa = pd.read_csv(CHECKPOINT_DIR + '/hist.csv')

for epoch in range(last_epoch, args.max_epoch):
    torch.manual_seed(RAND_SEED + epoch)
    
    df_hist.loc[epoch] = train(epoch)
    print(df_hist.loc[epoch])
    df_hist.to_csv(CHECKPOINT_DIR + '/hist.csv', index=False)
    
    if (epoch % args.save_interval) is 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),}, args.exp_name)
        
    
    
            
        
    


