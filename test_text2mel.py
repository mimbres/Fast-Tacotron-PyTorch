#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:06:30 2018

@author: sungkyun

USAGE: python test_text2mel.py <exp_name> <text_input> <max_output_length>

Args:
- model_load: <nn.Module> or <exp_name>. exp_name must have a directory of checkpoint containing config.json
- text_input: <str> or <list index(in test data) to display> or None. ex) 'Hello' or [0, 3, 5].
              If None, generate all from test data. Default is None.
- max_output_length: 200 frames by default


"""
import os, sys, shutil, pprint, argparse
import string
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from live_dataloader import LJSpeechDataset
from util.save_load_config import save_config, load_config
from tqdm import tqdm


DATA_ROOT = '/mnt/ssd3/data/LJSpeech-1.1'


# Argv input parser:
argv_inputs = sys.argv
exp_name = argv_inputs[1]
text_input = argv_inputs[2]
max_output_length = argv_inputs[3]

args = argparse.Namespace()
if len(argv_inputs) < 2:
    print('Required arguments: python train_text2mel.py <exp_name> <str_new_text or index_test_data>')
    exit()
else:
    CHECKPOINT_DIR = 'checkpoints/' + argv_inputs[1]
    config_fpath = 'checkpoints/{}/config.json'.format(argv_inputs[1])
    # Load config:
    args = load_config(config_fpath)


# Model type selection:
if args.model_type is 'base':
    from model.FastTacotron import Text2Mel
elif args.model_type is 'BN':
    from model.FastTacotron_BN import Text2Mel
else:
    print('Error in args.model_type: {} is unknown model_type. Please fix config.json, model_type.'.format(args.model_type))
    exit()

# Load Model:
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    model = Text2Mel().cuda()
else:
    model = Text2Mel().cpu()




#%% Function Definitions: ------------------------------------------------------
def generate_text2mel(model=nn.Module,
                      x_text=torch.autograd.Variable,
                      args=argparse.Namespace,
                      max_output_len=int):
    
    model.eval()
    torch.set_grad_enabled(False) # Pytorch 0.4: "volatile=True" is deprecated.

    _melspec = Variable(torch.FloatTensor(1,80,1).cuda()*0, requires_grad=False)

    for i in range(max_output_len):
        out_y, out_att = model(x_text, _melspec)
        _melspec = torch.cat((_melspec, out_y[:,:,-1].view(1,80,-1)), dim=2)

        if i>10:
            if (torch.sum(out_y[0,:,-5:]) < 1e-08):
                break
    
    return _melspec


def save_melspec(out_filepath, melspec):
    np.save(out_filepath, melspec)
    

def display_spec(dt1, dt2, outfile_dir, title='unknown_spec'):
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [10,5]
    sns.set(font_scale=.7)
    
    plt.subplot(211)
    plt.pcolormesh(dt1, cmap='jet')
    
    plt.subplot(212)
    plt.pcolormesh(dt2, cmap='jet')
    
    plt.title(args.exp_name + ','+title); plt.xlabel('Mel-spec frames')
    os.makedirs(outfile_dir, exist_ok=True) 
    plt.savefig(outfile_dir + '/images/'+ title + '_mspec.png', bbox_inches='tight', dpi=220)
    plt.close('all')
    
    
def display_att(att, outfile_dir, title='unknown_att'):
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [7,7]
    sns.set(font_scale=.7)   
    plt.pcolormesh(att, cmap='bone')   
    plt.title(title)
    plt.xlabel('Mel-spec frames'); plt.ylabel('Text characters')
    plt.savefig(outfile_dir + '/images/'+ title + '_att.png', bbox_inches='tight', dpi=220)
    plt.close('all') 
    #plt.pcolormesh(guide, cmap='summer' ); plt.savefig(CHECKPOINT_DIR + '/images/'+ 'att_guide.png', bbox_inches='tight', dpi=100)
    #plt.close('all')
    

def chr2int(text):
    # 'City$,' ==> ['c','i','t','y',','] ==> [7,13,24,29,2]
    chr2int_table = dict(zip(" ',-." + string.ascii_lowercase, np.arange(0, 31)))
    reduce_punc_table = str.maketrans(string.ascii_uppercase, string.ascii_lowercase,
                                     '0123456789!#"$%&\()*+/:;<=>?@[\\]^_`{|}~')

    text = list(text.translate(reduce_punc_table))
    return np.asarray([chr2int_table[c] for c in text])
#-------------------------------------------------------------------------------




#%% Text preparation: 
MELSPEC_DIR = 'checkpoints/' + args.exp_name + '/gen_melspec'
os.makedirs(MELSPEC_DIR, exist_ok=True) 

if isinstance(text_input, str): # case: ex) "Hello"
    # Convert input string into one-hot vectors
    x_text = chr2int(text_input)
    if USE_GPU:
        x_text = Variable(x_text.cuda().long(), requires_grad=False)
    else:
        x_text = Variable(x_text.long(), requires_grad=False)

    out_melspec = generate_text2mel(model=model, x_text=x_text, args=args, max_output_len=max_output_length)


elif isinstance(text_input, int) | isinstance(text_input, list) | (text_input is None):
    text_sel = text_input
    
    dset_test  = LJSpeechDataset(data_root_dir=DATA_ROOT, train_mode=False, output_mode='melspec', data_sel=text_sel)
    test_loader = DataLoader(dset_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                         )
    
    
    for batch_idx, (data_idx, x_text , x_melspec, zs) in tqdm(enumerate(test_loader)):
        if USE_GPU:
            x_text, x_melspec = Variable(x_text.cuda().long(), requires_grad=False), Variable(x_melspec.cuda().float(), requires_grad=False)
        else:
            x_text, x_melspec = Variable(x_text.long(), requires_grad=False), Variable(x_melspec.float(), requires_grad=False)
        
        
        #n_batch = len(test_loader) # number of iteration for one epoch.
        out_melspec = generate_text2mel(model=model, x_text=x_text, args=args, max_output_len=max_output_length)
        
        # Save to .npy file
        data_id = text_sel[batch_idx]
        save_melspec(MELSPEC_DIR + '/gen_{0:05d}.npy'.format(data_id), out_melspec) # save gen<original data id>.npy
        
        # Save images
        display_spec(dt1=(x_melspec[0,:,:]).data.cpu().numpy(),
                     dt2=(out_melspec[0,:,:]).data.cpu().numpy(),
                     outfile_dir=MELSPEC_DIR,
                     title='generated_{0:05d}'.format(data_id))
        

else:
    print('Error: text_input must be STR or int or list!')
    exit()






    for batch_idx, (data_idx, x_text , x_melspec_org, zs) in tqdm(enumerate(test_loader)):
        if USE_GPU:
            x_text, x_melspec_org = Variable(x_text.cuda().long(), requires_grad=False), Variable(x_melspec_org.cuda().float(), requires_grad=False)
        else:
            x_text, x_melspec_org = Variable(x_text.long(), requires_grad=False), Variable(x_melspec_org.float(), requires_grad=False)
        if batch_idx is disp_sel:
            break

        x_melspec = Variable(torch.FloatTensor(1,80,1).cuda()*0, requires_grad=False)

        import matplotlib.pyplot as plt

        for i in range(220):
            out_y, out_att = model(x_text[:,:], x_melspec)
            x_melspec = torch.cat((x_melspec, out_y[:,:,-1].view(1,80,-1)), dim=2)
            #plt.imshow(out_att[0,:,:].data.cpu().numpy())
            #plt.show()




#if isinstance(text_input, int) | isinstance(text_input, list): # case: ex) 1 or [0,3,100]
