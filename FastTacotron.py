#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 06:54:33 2018

@author: sungkyun
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.AttentionGuide import AttentionGuideGen


# Highway Layer & Block:  
class HighwayLayer(nn.Module):
    
    def __init__(self, in_ch=int, out_ch=int, k_sz=3, dil=int, causality=False):        
        super(HighwayLayer, self).__init__()
        
        self.out_ch    = out_ch
        self.k_sz      = k_sz        
        self.dil       = dil
        self.causality = causality       
        
        self.L = nn.Conv1d(in_ch, out_ch*2, kernel_size=k_sz, dilation=dil)
        return None
    
    def forward(self, x):
        
        if self.k_sz is not 1:
            if self.causality is True:
                pad = (self.dil*2, 0) # left-padding
            else:
                pad = (self.dil, self.dil) # padding to both sides
        else:            
            pad = (0, 0) # in this case, just 1x1 conv..
        
        h = self.L(F.pad(x, pad))
        h1, h2 = h[:, :self.out_ch,:], h[:, self.out_ch:, :]
        return F.sigmoid(h1) * h2 + (1-F.sigmoid(h1)) * x
    
    
    
class HighwayBlock(nn.Module):

    def __init__(self, io_chs=list, k_szs=list, dils=list, causality=False):
        super(HighwayBlock, self).__init__()
        
        #assert(len(io_chs)==len(k_szs) & len(io_chs)==len(dils)) 
        self.causality = causality
        self.hlayers   = nn.Sequential()
        self.construct_hlayers(io_chs, k_szs, dils)
        return None

    def construct_hlayers(self, io_chs, k_szs, dils):
        
        total_layers = len(io_chs) # = len(k_szs)
        
        for l in range(total_layers):
            self.hlayers.add_module(str(l),
                                    HighwayLayer(in_ch=io_chs[l],
                                                 out_ch=io_chs[l],
                                                 k_sz=k_szs[l],
                                                 dil=dils[l],
                                                 causality=self.causality
                                                 ))
        return
    
    def forward(self, x):
        return self.hlayers(x)
    
    
    
# Text Encoder:
class TextEnc(nn.Module):  
    def __init__(self, input_dim=31, e_dim=128, d_dim=256,
                 h_io_chs=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
                 h_k_szs=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                 h_dils=[1, 3, 9, 27, 1, 3, 9, 27, 1, 1, 1, 1]):
        super(TextEnc, self).__init__()
        self.d_dim = d_dim
        
        # Layers:
        self.embed_layer = nn.Embedding(input_dim, e_dim) 
        self.conv1x1_0   = nn.Conv1d(e_dim,   2*d_dim, kernel_size=1)
        self.conv1x1_1   = nn.Conv1d(2*d_dim, 2*d_dim, kernel_size=1)
        self.h_block     = HighwayBlock(h_io_chs, h_k_szs, h_dils, causality=False)
        return None
 
    def forward(self, x):
        x = self.embed_layer(x).permute(0,2,1)     # BxT -> BxCxT with C=e
        x = F.relu(self.conv1x1_0(x))
        x = self.conv1x1_1(x)                      # BxCxT=Bx512xT]
        x = self.h_block(x)                        # BxCxT with C=2d
        return x[:, :self.d_dim, :], x[:, :self.d_dim, :]    # Split C={d,d}, to be used as K, V. 



# Audio Encoder:
class AudioEnc(nn.Module):
    def __init__(self, input_dim=80, d_dim=256,
                 h_io_chs=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
                 h_k_szs=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                 h_dils=[1, 3, 9, 27, 1, 3, 9, 27, 3, 3]):
        super(AudioEnc, self).__init__()
        
        # Layers:
        self.conv1x1_0 = nn.Conv1d(input_dim, d_dim, kernel_size=1)
        self.conv1x1_1 = nn.Conv1d(d_dim, d_dim, kernel_size=1)
        self.conv1x1_2 = nn.Conv1d(d_dim, d_dim, kernel_size=1)
        self.h_block   = HighwayBlock(h_io_chs, h_k_szs, h_dils, causality=True)
        return None
    
    def forward(self, x):
        x = F.relu(self.conv1x1_0(x))
        x = F.relu(self.conv1x1_1(x))
        x = F.relu(self.conv1x1_2(x))
        return self.h_block(x)             # output BxCxT with C=d
    


# Audio Decoder:
class AudioDec(nn.Module):
    def __init__(self, input_dim=512, output_dim=80, d_dim=256,
                 h_io_chs=[256, 256, 256, 256, 256, 256],
                 h_k_szs=[3, 3, 3, 3, 3, 3],
                 h_dils=[1, 3, 9, 27, 1, 1]):
        super(AudioDec, self).__init__()
        
        # Layers:
        self.conv1x1_0 = nn.Conv1d(input_dim, d_dim, kernel_size=1)
        self.h_block   = HighwayBlock(h_io_chs, h_k_szs, h_dils, causality=True)
        self.conv1x1_1 = nn.Conv1d(d_dim, d_dim, kernel_size=1)
        self.conv1x1_2 = nn.Conv1d(d_dim, d_dim, kernel_size=1)
        self.conv1x1_3 = nn.Conv1d(d_dim, d_dim, kernel_size=1)
        self.conv1x1_4 = nn.Conv1d(d_dim, output_dim, kernel_size=1)
        return None
        
    def forward(self, x):
        x = self.conv1x1_0(x)
        x = self.h_block(x)
        x = F.relu(self.conv1x1_1(x))
        x = F.relu(self.conv1x1_2(x))
        x = F.relu(self.conv1x1_3(x))
        return F.sigmoid(self.conv1x1_4(x))
    


# SSRN:
class SSRN(nn.Module):
    def __init__(self, input_dim=80, output_dim=513, c_dim=512):
        super(SSRN, self).__init__()
        
        # Layers:
        self.ssrn_layers = nn.Sequential()
        self.ssrn_layers.add_module('conv1x1_0', nn.Conv1d(input_dim, c_dim, kernel_size=1))
        self.ssrn_layers.add_module('h_block_0', HighwayBlock([c_dim, c_dim], [3,3], [1,3])) # By default, causality=False
        
        self.ssrn_layers.add_module('deconv2x1_0', nn.ConvTranspose1d(c_dim, c_dim, kernel_size=2))
        self.ssrn_layers.add_module('h_block_1', HighwayBlock([c_dim, c_dim], [3,3], [1,3]))
        self.ssrn_layers.add_module('deconv2x1_1', nn.ConvTranspose1d(c_dim, c_dim, kernel_size=2))
        self.ssrn_layers.add_module('h_block_2', HighwayBlock([c_dim, c_dim], [3,3], [1,3]))
        
        self.ssrn_layers.add_module('conv1x1_1', nn.Conv1d(c_dim, 2*c_dim, kernel_size=1))
        self.ssrn_layers.add_module('h_block_3', HighwayBlock([2*c_dim, 2*c_dim], [3,3], [1,1]))
        self.ssrn_layers.add_module('conv1x1_2', nn.ConvTranspose1d(2*c_dim, output_dim, kernel_size=1))
        
        self.ssrn_layers.add_module('conv1x1_3', nn.Conv1d(output_dim, output_dim, kernel_size=1))
        self.ssrn_layers.add_module('relu_0', nn.ReLU())
        self.ssrn_layers.add_module('conv1x1_4', nn.Conv1d(output_dim, output_dim, kernel_size=1))
        self.ssrn_layers.add_module('relu_1', nn.ReLU())
        
        self.ssrn_layers.add_module('conv1x1_5', nn.Conv1d(output_dim, output_dim, kernel_size=1))
        return None
        
    def forward(self, x):
        return F.sigmoid(self.ssrn_layers(x))
        
        
    
#%% Text2Mel:
class Text2Mel(nn.Module):
    def __init__(self, text_dim=31, melspec_dim=80, e_dim=128, d_dim=256):
        super(Text2Mel, self).__init__()    
        
        self.e_dim = e_dim; self.d_dim = d_dim
        self.text_enc  = TextEnc(input_dim=text_dim)
        self.audio_enc = AudioEnc(input_dim=melspec_dim)
        self.audio_dec = AudioDec(input_dim=2*d_dim, output_dim=melspec_dim)
        
        self.optional_output=False
        
        return None
    
    def forward(self, x_text, x_melspec):
        
        # K,V: encoded text.  Q: encoded audio
        K, V = self.text_enc(x_text)     # Key, Value: Bx256xN with N=nth text 
        Q    = self.audio_enc(x_melspec) # Query     : Bx256xT with T=T_audio
        
        # Attention: 
        K_T = K.permute(0,2,1)        # K, transposed as BxTx256
        A = F.softmax(torch.matmul(K_T, Q) / np.sqrt(self.d_dim), dim=1) # softmax along with Text length dimension, resulting BxNxT
        
        # Attentional seed to audio_decoder, RQ = Att(K,V,Q)
        R = torch.matmul(V, A)        # Bx256xT with T=T_audio
        RQ = torch.cat([R,Q], dim=1)  # Bx512xT
        
        # Decoding Mel-spectrogram, Y
        Y = self.audio_dec(RQ)        # Bx80xT with T=T_audio
        
        if self.optional_output is True:
            return Y, A, K, V, Q
        else:
            return Y, A
    