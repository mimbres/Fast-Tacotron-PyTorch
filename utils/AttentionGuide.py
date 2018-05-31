#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:01:58 2018

@author: sungkyun
"""
import numpy as np
from skimage.transform import resize

class AttentionGuideGen():
    def __init__(self, g=0.2, base_size=(300,300)):
        super(AttentionGuideGen, self).__init__()
        
        self.g = g
        self.base_size = base_size # (N, T), where N is text index; T is mel-spec index
        self.base_guide = np.ndarray(base_size, dtype=np.float32)
        
        # Initiazlie base_guide array
        N = base_size[0]
        T = base_size[1]
        for n in range(N):
            for t in range(T):
                self.base_guide[n,t] = 1 - np.exp( -np.power((n/N - t/T), 2)/ (2*np.power(g, 2)))
        
        print('AttentionGuide: Succesfully initialized with g={}.'.format(self.g))
        return None                
        
        
    def get_guide(self, target_size=np.asarray([10, 15])):
        W = resize(self.base_guide, target_size, mode='constant').astype(np.float32)

#        import matplotlib.pyplot as plt
#        plt.imshow(W)
        return W
    
    def get_padded_guide(self, target_sz=(10,15), pad_sz=np.asarray([[3,5],[2,4]]), set_silence_state=-1):
        '''
        target_sz = (N, T)
        pad_sz: Bx2 numpy array, pad_sz[i] = [text_left_pad_length, melspec_left_pad_length]
        
        NOTE: every input data (text and audio) must contain at least one zero pad in the left!
        '''
        batch_sz = pad_sz.shape[0]
        W = np.zeros((batch_sz, target_sz[0], target_sz[1]), dtype=np.float32) + 1
        
        # Generate guide for each batch dimension:
        for i in range(batch_sz):
            _new_guide_sz = [target_sz[0]-pad_sz[i,0], target_sz[1]-pad_sz[i,1]]
            _new_guide = resize(self.base_guide, _new_guide_sz, mode='constant').astype(np.float32)
            
            W[i, pad_sz[i,0]:, pad_sz[i,1]:] = _new_guide
            W[i, :pad_sz[i,0], pad_sz[i,1]-1] = 0.
        
            if set_silence_state is not -1:
                W[i, pad_sz[i,0]:, pad_sz[i,1]-1] = set_silence_state
                
        return W
    
#    def get_padded_guide(self, target_sz=(10,15), pad_sz=np.asarray([3,5]), set_silence_state=-1):
#        '''
#        target_sz = (N, T)
#        pad_sz = [text_left_pad_length, melspec_left_pad_length]
#        '''
#        W = np.zeros(target_sz, dtype=np.float32) + 1
#            
#        new_guide_sz = [target_sz[0]-pad_sz[0], target_sz[1]-pad_sz[1]]
#        new_guide = resize(self.base_guide, new_guide_sz, mode='constant').astype(np.float32)
#        
#        W[pad_sz[0]:, pad_sz[1]:] = new_guide
#        W[:pad_sz[0],pad_sz[1]] = 0.
#        
#        if set_silence_state is not -1:
#            W[pad_sz[0]:,pad_sz[1]] = set_silence_state
#        
##        import matplotlib.pyplot as plt
##        plt.imshow(W)
#        
#        return W
    
    