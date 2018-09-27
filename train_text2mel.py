#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:49:30 2018

@author: sungkyun
"""
import os, sys, shutil, pprint, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from live_dataloader import LJSpeechDataset
from util.AttentionGuide import AttentionGuideGen
from util.save_load_config import save_config, load_config
from tqdm import tqdm

torch.backends.cudnn.benchmark=True

#%% Parsing arguments

USE_GPU = torch.cuda.is_available()
RAND_SEED  = 0

''' USAGE: python train_text2mel.py <exp_name> <fresh-start or continue> '''
argv_inputs = sys.argv
#argv_inputs = ['','00'] # This line is only for debug...

args = argparse.Namespace()
if len(argv_inputs) < 2:
    print('USAGE: python train_text2mel.py <exp_name> <fresh-start or continue>')
    exit()
else:
    CHECKPOINT_DIR = 'checkpoints/' + argv_inputs[1]
    config_fpath = 'checkpoints/{}/config.json'.format(argv_inputs[1])
    
    if (len(argv_inputs) == 3):
        if argv_inputs[2].lower() == 'fresh-start':
            # Copy template config file
            os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
            shutil.copyfile('config_template.json', config_fpath)
            args = load_config(config_fpath)
            args.exp_name = argv_inputs[1].lower()
            save_config(args, config_fpath)
        else:
            args = load_config(config_fpath)
    else:
        args = load_config(config_fpath)


# Model type selection:
if args.model_type == 'base':
    from model.FastTacotron import Text2Mel
elif args.model_type is 'BN':
    from model.FastTacotron_BN import Text2Mel
else:
    print('Error in args.model_type: {} is unknown model_type. Please fix config.json, model_type.'.format(args.model_type))
    exit()
    
pprint.pprint(vars(args)) # Display settings..
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
    
    plt.title(args.exp_name + ','+title); plt.xlabel('Mel-spec frames')
    os.makedirs(CHECKPOINT_DIR + '/images', exist_ok=True) 
    plt.savefig(CHECKPOINT_DIR + '/images/'+ title + '_mspec.png', bbox_inches='tight', dpi=220)
    plt.close('all')
    
    
def display_att(att, guide, title='unknown_att'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [7,7]
    sns.set(font_scale=.7)   
    plt.pcolormesh(att, cmap='bone')   
    plt.title(title)
    plt.xlabel('Mel-spec frames'); plt.ylabel('Text characters')
    plt.savefig(CHECKPOINT_DIR + '/images/'+ title + '_att.png', bbox_inches='tight', dpi=220)
    plt.close('all') 
    
    plt.pcolormesh(guide, cmap='summer' ); plt.savefig(CHECKPOINT_DIR + '/images/'+ 'att_guide.png', bbox_inches='tight', dpi=100)
    plt.close('all')

def load_checkpoint(filepath):
    '''  Load pre-trained model. '''
    dt = torch.load(filepath)
    print('Loading from expname: {}, epoch: {}...'.format(args.exp_name, dt['epoch']) )
    model.load_state_dict(dt['state_dict'])
    optimizer.load_state_dict(dt['optimizer'])    
    return dt['epoch']

def save_checkpoint(state):
    filepath = CHECKPOINT_DIR + '/checkpoint{}.pth.tar'.format(state['epoch'])
    torch.save(state, filepath)
    shutil.copyfile(filepath, CHECKPOINT_DIR + '/checkpoint_latest.pth.tar')
    # Save argparse config..
    save_config(args, CHECKPOINT_DIR + '/config.json')
    
#%% Data Loading
DATA_ROOT = args.data_root#'/mnt/ssd2/data/LJSpeech-1.1'

dset_train = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=True,  output_mode='melspec')
dset_test  = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=False, output_mode='melspec')

train_loader = DataLoader(dset_train,
                          batch_size=args.batch_train,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True
                         ) # number of CPU threads, practically, num_worker = 4 * num_GPU

test_loader = DataLoader(dset_test,
                          batch_size=args.batch_test,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                         )


#%% Train
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    model = Text2Mel().cuda()
else:
    model = Text2Mel().cpu()
    
optimizer = torch.optim.Adam(model.parameters(), lr=(2e-04)*(args.batch_train/16), betas=(0.5, 0.9), eps=1e-06)

print_model_sz(model)

guide_generator = AttentionGuideGen()

loss_L1 = nn.L1Loss(size_average=True, reduce=True) # Sigmoid is not included
loss_BCE = nn.BCEWithLogitsLoss(weight=None, size_average=True, reduce=True) # BCE with sigmoid


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
        n_batch = len(train_loader) # number of iteration for one epoch.
#        if batch_idx is not None:
#            break
        
        optimizer.zero_grad()
        out_y, out_y_sig, out_att = model(x_text, x_melspec)
        
        l1 = loss_L1(out_y_sig[:,:,:-1], x_melspec[:,:,1:]) 
        l2 = loss_BCE(out_y[:,:,:-1], x_melspec[:,:,1:])
        
        # l3: Attention loss, W is guide matrices with BxNxT        
        W = guide_generator.get_padded_guide(target_sz=(x_text.shape[1],x_melspec.shape[2]),
                                             pad_sz=(zs).data.cpu().numpy(),
                                             set_silence_state=args.silence_state_guide)
        W_sz = W.size
        W = torch.cuda.FloatTensor(W) if USE_GPU else torch.FloatTensor(W).cpu()
        l3 = torch.sum(out_att * W) / W_sz # Normalization
        
        loss = l1 + l2 + l3
        
        loss.backward()
        optimizer.step()
#        print('Train Epoch: {} [{}/{}], Total loss={:.6f}, L1={:.6f}, BCE={:.6f}, A={:.6f}'.format(
#                epoch, batch_idx * train_loader.batch_size, total_data_sz, 
#                loss.item(), l1.item(), l2.item(), 0.))
        train_loss['Total'] += loss.item() / n_batch
        train_loss['L1'] += l1.item() / n_batch
        train_loss['BCE'] += l2.item() / n_batch
        train_loss['Att'] += l3.item() / n_batch
        
    
        select_data = args.sel_display
        if ((epoch in [1,3,5,10,20,30,40]) | (epoch%args.save_interval is 0)) & (select_data in data_idx ):
            sel = np.where(data_idx.cpu()==select_data)[0].data[0]
            
            out_y_sig_cpu = (out_y_sig[sel,:,:]).data.cpu().numpy()
            out_att_cpu = (out_att[sel,:,:]).data.cpu().numpy()
            #org_text  = (x_text[sel,:]).data.cpu().numpy()
            org_melspec =(x_melspec[sel,:,:]).data.cpu().numpy()
            
            display_spec(out_y_sig_cpu, org_melspec, 'Sample {}: epoch = {}'.format(select_data, epoch))
            display_att(out_att_cpu, W[sel,:,:], 'Sample {}: epoch = {}'.format(select_data, epoch))
        
    return train_loss

#def generate_text2mel(model_load=None, new_text=None):
#    '''
#    Args:
#    - text: <str> or <list index(in test data) to display>. ex) 'Hello' or [0, 3, 5]
#    - model_load: <existing model to load> or <exp_name>. exp_name must have a directory of checkpoint containing config.json        
#    '''
#    
#    if isinstance(model_load, str):
#        import os, shutil, pprint #, argparse
#        import numpy as np
#        import torch
#        import torch.nn as nn
#        from torch.utils.data import DataLoader
#        from torch.autograd import Variable
#        from live_dataloader import LJSpeechDataset
#        from util.save_load_config import save_config, load_config
#        from model.FastTacotron import Text2Mel
#        
#        
#        
#        
#        
#    
#    
#        
#        
#        
#    model.eval()
#    torch.set_grad_enabled(False) # Pytorch 0.4: "volatile=True" is deprecated. 
#    
#    for batch_idx, (data_idx, x_text , x_melspec_org, zs) in tqdm(enumerate(test_loader)):
#        if USE_GPU:
#            x_text, x_melspec_org = Variable(x_text.cuda().long(), requires_grad=False), Variable(x_melspec_org.cuda().float(), requires_grad=False)
#        else:
#            x_text, x_melspec_org = Variable(x_text.long(), requires_grad=False), Variable(x_melspec_org.float(), requires_grad=False)
#        if batch_idx is disp_sel:
#            break
#        
#        x_melspec = Variable(torch.FloatTensor(1,80,1).cuda()*0, requires_grad=False)
#        
#        import matplotlib.pyplot as plt
#     
#        for i in range(220):  
#            out_y, out_att = model(x_text[:,:], x_melspec)
#            x_melspec = torch.cat((x_melspec, out_y[:,:,-1].view(1,80,-1)), dim=2)
#            #plt.imshow(out_att[0,:,:].data.cpu().numpy())
#            #plt.show()
#            
#   
#    plt.imshow(x_melspec[0,:,:].data.cpu().numpy())
#    plt.show()
#    
#    plt.imshow(x_melspec_org[0,:,:].data.cpu().numpy())
#    plt.show()
#    
#    plt.imshow(out_att[0,:,:].data.cpu().numpy())
#    plt.show()
#    
    
#%% Train Main Loop
df_hist = pd.DataFrame(columns=('Total', 'L1', 'BCE','Att'))
last_epoch = 0

if args.load is not None:
    last_epoch = load_checkpoint(args.load)
    df_hist = pd.read_csv(CHECKPOINT_DIR + '/hist.csv')

for epoch in range(last_epoch, args.max_epoch):
    torch.manual_seed(RAND_SEED + epoch)
    
    df_hist.loc[epoch] = train(epoch)
    print(df_hist.loc[epoch])
    df_hist.to_csv(CHECKPOINT_DIR + '/hist.csv', index=False)
    
    # Save model parameters & argparser configurations.
    if (epoch % args.save_interval) is 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),} )
        
        if args.generate is not None:
            pass
            
