#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:46:20 2018

@author: sungkyun

- Live Data Loader for LJ-Speech Dataset.
"""
import pandas as pd
import numpy as np
import torch
import string, glob
import librosa
from torch.utils.data.dataset import Dataset
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, MemoryCacheDataset

DATA_ROOT = '/mnt/ssd3/data/LJSpeech-1.1'
N_TRAIN = 13000 # N_TEST = 100 (13000~13099) 
X_SPEC_MAX = 193.99077 # Required for feature normalization
X_MELSPEC_MAX = 0.04071
OMIT_DATA_ROWS =[945, 1119, 1149, 1371, 1853, 1929, 1939, 2965, 3405, 4246, 4670,
                  4684, 4699, 4706, 4713, 5163, 5164, 5165, 5166, 5167, 5169, 5172,
                   5174, 5182, 5185, 5532, 5935, 5964, 5974, 6039] # Reduce sentences with foreign characters.
MAX_LEN_TEXT           = 250+2 # Max-lengths are required for z-padding
MAX_LEN_MELSPEC        = 217+2
MAX_LEN_SPEC           = 868+8
#MAX_LEN_PAIRED_TEXT    = 360
#MAX_LEN_PAIRED_MELSPEC = 291# Max-lengths are required for z-padding
#MAX_LEN_PAIRED_SPEC    = 1164
    
class SpecSource(FileDataSource):
    '''
    USAGE:
        Mel-spec-only : mode = ['melspec']
        Spec, Mel-spec: mode = ['SSRN']
    '''
    def __init__(self, wav_data_root=None, file_sel_range=None, target_sr = 22050, output_mode=['melspec'], norm_factors=[X_SPEC_MAX,X_MELSPEC_MAX], max_num_files=-1):
        
        file_list = sorted(glob.glob(wav_data_root + "/*.wav")) # All *.wav files into list
        for i in sorted(OMIT_DATA_ROWS, reverse=True):
            del file_list[i] # Omitting foreign language files
        self.input_wav_files = file_list
        
        self.target_sr = target_sr
        self.output_mode = output_mode
        self.norm_factors = norm_factors # [max_X_spec, max_X_melspec]
        if max_num_files is not -1 :self.input_wav_files = self.input_wav_files[:max_num_files]
        if file_sel_range is not None: self.input_wav_files = self.input_wav_files[file_sel_range[0]:file_sel_range[1]]
        
        self.mel_filter = librosa.filters.mel(self.target_sr, 1024, n_mels=80)
        
        
    def collect_files(self):
        # This class method is required..
        return self.input_wav_files
    
    
    def collect_features(self, path):
        x, fs = librosa.load(path, sr=self.target_sr, mono=True, dtype=np.float32)
        x_spec = np.abs(librosa.core.stft(y=x, n_fft=1024, hop_length=256, window='hann')) #STFT size : [513,4T]
        x_spec = np.power(x_spec / self.norm_factors[0], 0.6)  # normalize by max(abs(X_spec)) 
        x_spec_fixed_t = int(x_spec.shape[1]/4) * 4
        x_spec = x_spec[:, :x_spec_fixed_t]
        
        x_melspec = np.matmul(self.mel_filter, x_spec)
        x_melspec = x_melspec[:, np.arange(0, x_spec_fixed_t, 4)] 
        #x_melspec = np.power(x_melspec / self.norm_factors[1], 0.6) # normalize by max(abs(X_melspec))
        x_melspec = np.power((x_melspec / np.max(x_melspec)) * np.random.uniform(0.9,1.0), 0.6)
        
        
        if 'SSRN' in self.output_mode:
            return x_spec, x_melspec
        else:
            return x_melspec



#%%
class LJSpeechDataset(Dataset):
    '''
    Live Data Loader for LJ-Speech Dataset.
    OUTPUT MODE:
        mode = ['melspec'] : return index, text, melspec
        mode = ['SSRN']    : return index, spec, melspec
    '''
    def __init__(self, data_root_dir=DATA_ROOT, train_mode=False , output_mode='melspec', transform=None):
        
        
        self.wav_root_dir = data_root_dir + '/wavs/'
        self.train_mode  = train_mode
        self.output_mode = output_mode
        self.transform   = transform
        
        self.max_len_text    = MAX_LEN_TEXT
        self.max_len_melspec = MAX_LEN_MELSPEC 
        self.max_len_spec    = MAX_LEN_SPEC
        
#        self.max_len_paired_text = MAX_LEN_PAIRED_TEXT # Max-lengths are required for z-padding
#        self.max_len_paired_spec = MAX_LEN_PAIRED_SPEC
#        self.max_len_paired_melspec = MAX_LEN_PAIRED_MELSPEC

        # Preparing Text:
        self.text_csv_path = data_root_dir + '/metadata.csv'
        self.reduce_punc_table = str.maketrans(string.ascii_uppercase, string.ascii_lowercase, 
                                         '0123456789!#"$%&\()*+/:;<=>?@[\\]^_`{|}~') 
        self.chr2int_table = dict(zip(" ',-." + string.ascii_lowercase, np.arange(0, 31)))                                         

        df = pd.read_csv(self.text_csv_path, index_col=False, sep='|', header=None, memory_map=True) # memory_map: speed-up reading. 
        nan_rows = df[df[2].isnull()].index.values
        df.iloc[nan_rows,2] = df.iloc[nan_rows,1]  # fixing dataset NaN value bugs...
        df = df.drop(1, axis=1);  df.columns=['file_id', 'text']
        df = df.drop(OMIT_DATA_ROWS, axis=0).reset_index(drop=True) # Omitting foreign language..

        if self.train_mode is True:
            self.file_ids = df.iloc[0:N_TRAIN, 0] # file_ids: LJ**-**** (13,000)
            self.texts    = df.iloc[0:N_TRAIN, 1] 
        else:
            self.file_ids = df.iloc[N_TRAIN:, 0].reset_index(drop=True) # (100)
            self.texts    = df.iloc[N_TRAIN:, 1].reset_index(drop=True)       
        
        
        # Prepraing Audio:
        if self.train_mode is True:
            self.spec_features = MemoryCacheDataset(
                    FileSourceDataset(
                            SpecSource(
                                    wav_data_root=self.wav_root_dir,
                                    file_sel_range=[0, N_TRAIN], 
                                    output_mode=self.output_mode)),
                            cache_size=len(self.file_ids))
        else:
            self.spec_features = MemoryCacheDataset(
                    FileSourceDataset(
                            SpecSource(
                                    wav_data_root=self.wav_root_dir,
                                    file_sel_range=[N_TRAIN, None],
                                    output_mode=self.output_mode)),
                            cache_size=len(self.file_ids))
        assert(len(self.file_ids) == len(self.spec_features))
        
        
#        # Pairing: Sort and divide by feature lengths, then concat small + large 
#        lengths       = np.load('mspec_length_train_13000.npy')
#        sorted_by_len = np.argsort(lengths)
#        n_org       = len(sorted_by_len)
#        n_pairs     = int(n_org / 2)
#        self.paired_items = list()
#        for i in range(n_pairs):
#            self.paired_items.append([sorted_by_len[i], sorted_by_len[n_org - 1 - i]])
    
        return None
        
        
    def __getitem__(self, index): # = Index of self.paired_items
        
#        # Decouple paired index to idx1, idx2
#        _choice = np.random.choice([0,1])
#        idx1, idx2 = paired_items[index][_choice], paired_items[index][1-_choice]
        if self.output_mode is 'melspec':
            text = self.chr2int(self.texts[index])
            melspec = self.spec_features[index]
            
            text, nz_text = self.zeropad(text, self.max_len_text)
            melspec, nz_melspec = self.zeropad(melspec, self.max_len_melspec)
            return index, torch.LongTensor(text), melspec, np.asarray([nz_text, nz_melspec])
        
        else: # if self.output_mode is 'SSRN':
            spec, melspec = self.spec_features[index]
            
            spec, nz_spec = self.zeropad(spec, self.max_len_spec)
            melspec, nz_melspec = self.zeropad(melspec, self.max_len_melspec)
            
            return index, spec, melspec, np.asarray([nz_spec, nz_melspec])
        
        
        #np.flip(paired_items[30], axis=0)

    def __len__(self):
        #return len(self.paired_items)
        return len(self.file_ids)
    
    def chr2int(self, text):
        # 'City$,' ==> ['c','i','t','y',','] ==> [7,13,24,29,2]
        text = list(text.translate(self.reduce_punc_table))            
        return np.asarray([self.chr2int_table[c] for c in text]) 
    
    def zeropad(self, x, target_length):
        
        if len(x.shape) is 1:
            # 1D input:
            n_zeros = target_length - len(x)
            #x = np.pad(x, (n_zeros,0), 'constant', constant_values=(0,0))
            x = np.pad(x, (1,n_zeros-1), 'constant', constant_values=(0,0))  # letf 1 + right (all-1)-zpading!
        else:
            # 2D input: D x T
            n_zeros = target_length - x.shape[1]
            xz = np.zeros((x.shape[0], x.shape[1]+n_zeros))
            #xz[:, n_zeros:] = x
            xz[:, 1:(xz.shape[1] - n_zeros + 1)] = x # right-zpading!!
            x = xz
        return x, n_zeros
            
    
    
        
        
 

