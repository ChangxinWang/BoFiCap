# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label
from random import random
from turtle import Turtle
from typing import ForwardRef

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from thop import profile

import copy
import math
import numpy as np
import time

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
import captioning.utils.glo as glo

"""
this part is for predict length and syntactic label with [LEN] markbit
decoder generate all words together
we call this arch Non-Autoregressive (NAIC)
"""

NA_LENGTH_DIM=20
NA_SYN_DIM=10
NA_SYN_LOWER=4
NA_SYN_UPPER=6
class LengthPredictor_NAIC(nn.Module):
    """
    Non-Autoregressive Length Predictor
    each time generate one phrase length and one phrase syntactic label
    input is syn seq
    """
    def __init__(self, d_model, length_attn, ff, N_len, dropout=0.1):
        super(LengthPredictor_NAIC, self).__init__()
        self.d_model = d_model
        self.length_attn = length_attn
        self.ff = ff
        self.N_len = N_len
        self.norm = LayerNorm(d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.Length_classifier1 = nn.Linear(d_model, 100)
        self.Length_classifier2 = nn.Linear(100, NA_LENGTH_DIM)
        self.Syntactic_classifier1 = nn.Linear(d_model, 100)
        self.Syntactic_classifier2 = nn.Linear(100, NA_SYN_DIM)
        # if N_len = 0, only do cross attention once
        if self.N_len == 0:
            self.LengthPredictor = SublayerConnection(d_model, dropout)
        else:
            c = copy.deepcopy
            self.LengthPredictor = clones(LengthPredictorLayer(d_model, c(length_attn), c(length_attn), c(ff), dropout), N_len)

    def forward(self, syn_seq_embed, memory, src_mask, tgt_mask):
        """
        memory: (batch*seq_per_img) * boxes * d_model
        syn_seq_embed: (batch*seq_per_img) * (max_length + 2) * d_model

        predict_phrase_length_N: (batch*seq_per_img)
        predict_phrase_length_logprob: (batch*seq_per_img) * NA_LENGTH_DIM
        predict_phrase_syn_N: (batch*seq_per_img)
        predict_phrase_syn_logprob: (batch*seq_per_img) * NA_LENGTH_DIM
        """
        x = syn_seq_embed
        m = memory
        if self.N_len == 0:
            output = self.norm(self.LengthPredictor(x, lambda x: self.length_attn(x, m, m, src_mask)))
        else:
            for layer in self.LengthPredictor:
                x = layer(x, m, src_mask, tgt_mask)
            output = self.norm(x)
        output = output[:, 0, :] # we only need [LEN] hidden state
        length_output = self.Dropout(F.relu(self.Length_classifier1(output)))
        predict_phrase_length_logprob = F.log_softmax(self.Length_classifier2(length_output), dim=-1)
        syn_output = self.Dropout(F.relu(self.Syntactic_classifier1(output)))
        predict_phrase_syn_logprob = F.log_softmax(self.Syntactic_classifier2(syn_output), dim=-1)

        _, predict_phrase_length_N = torch.max(predict_phrase_length_logprob.data, 1)
        _, predict_phrase_syn_N = torch.max(predict_phrase_syn_logprob.data, 1)
        return predict_phrase_length_N.int(), predict_phrase_length_logprob, predict_phrase_syn_N.long(), predict_phrase_syn_logprob


class EncoderDecoder_NAIC(nn.Module):
    """
    Non-Autoregressive Encoder-Decoder with LengthPredictorNAIC
    """
    def __init__(self, opt, d_model, encoder, decoder, src_embed, syn_embed, tgt_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_NAIC, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        """
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        """
        memory = self.encode(src, src_mask)
        predict_phrase_length_logprob, predict_phrase_syn_logprob, syn_mask = \
            self.get_predict_phrase_length_syn_NA(memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq)
        predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
        return predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_NA(self, memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        B, L = phrase_length.shape[0:2]
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, NA_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, NA_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        syn_mask = phrase_length.new_zeros([B, L-2, L-2], dtype=torch.bool)
        for i in range(0, B):
            syn_mask[i, :, :last[i]-1] = True
        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :], syn_mask

    def get_predict_phrase_length_syn_part_NA(self, syn_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.syn_embed(syn_seq)), memory, src_mask, tgt_mask)
    
    def decode_NA(self, memory, syn_seq, src_mask, tgt_mask):
        input_embed = memory.new_full(syn_seq.shape, self.bos_idx, dtype=torch.long)
        input_embed = self.pos_embed(self.tgt_embed(input_embed) + self.syn_embed(syn_seq))
        return self.decoder(input_embed, memory, src_mask, tgt_mask)


"""
this part is for predict length and syntactic label with [LEN] markbit
but input is real word seq instead of syn seq
and decoder generate one phrase each time
we call this arch Semi-Autoregressive (SAIC)
"""
SA_LENGTH_DIM=20
SA_SYN_DIM=10
SA_SYN_LOWER=4
SA_SYN_UPPER=6
class LengthPredictor_SAIC(nn.Module):
    """
    Semi-Autoregressive Length Predictor
    each time generate one phrase length and one phrase syntactic label
    with real seq as input
    """
    def __init__(self, d_model, length_attn, ff, N_len, dropout=0.1):
        super(LengthPredictor_SAIC, self).__init__()
        self.d_model = d_model
        self.length_attn = length_attn
        self.ff = ff
        self.N_len = N_len
        self.norm = LayerNorm(d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.Length_classifier1 = nn.Linear(d_model, 100)
        self.Length_classifier2 = nn.Linear(100, SA_LENGTH_DIM)
        self.Syntactic_classifier1 = nn.Linear(d_model, 100)
        self.Syntactic_classifier2 = nn.Linear(100, SA_SYN_DIM)
        # if N_len = 0, only do cross attention once
        if self.N_len == 0:
            self.LengthPredictor = SublayerConnection(d_model, dropout)
        else:
            c = copy.deepcopy
            self.LengthPredictor = clones(LengthPredictorLayer(d_model, c(length_attn), c(length_attn), c(ff), dropout), N_len)

    def forward(self, seq_embed, memory, src_mask, tgt_mask):
        """
        memory: (batch*seq_per_img) * boxes * d_model
        seq_embed: (batch*seq_per_img) * (max_length + 2) * d_model

        predict_phrase_length_N: (batch*seq_per_img)
        predict_phrase_length_logprob: (batch*seq_per_img) * SA_LENGTH_DIM
        predict_phrase_syn_N: (batch*seq_per_img)
        predict_phrase_syn_logprob: (batch*seq_per_img) * SA_LENGTH_DIM
        """
        x = seq_embed
        m = memory
        if self.N_len == 0:
            output = self.norm(self.LengthPredictor(x, lambda x: self.length_attn(x, m, m, src_mask)))
        else:
            for layer in self.LengthPredictor:
                x = layer(x, m, src_mask, tgt_mask)
            output = self.norm(x)
        output = output[:, 0, :] # we only need [LEN] hidden state
        length_output = self.Dropout(F.relu(self.Length_classifier1(output)))
        predict_phrase_length_logprob = F.log_softmax(self.Length_classifier2(length_output), dim=-1)
        syn_output = self.Dropout(F.relu(self.Syntactic_classifier1(output)))
        predict_phrase_syn_logprob = F.log_softmax(self.Syntactic_classifier2(syn_output), dim=-1)

        _, predict_phrase_length_N = torch.max(predict_phrase_length_logprob.data, 1)
        _, predict_phrase_syn_N = torch.max(predict_phrase_syn_logprob.data, 1)
        return predict_phrase_length_N.int(), predict_phrase_length_logprob, predict_phrase_syn_N.long(), predict_phrase_syn_logprob


class EncoderDecoder_SAIC(nn.Module):
    """
    Semi-Autoregressive Encoder-Decoder with LengthPredictor_SAIC
    """
    def __init__(self, opt, d_model, encoder, decoder, src_embed, syn_embed, tgt_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_SAIC, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, labels, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask):
        """
        labels: (batch*seq_per_img) * (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_syn_seq: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_seq : (batch*seq_per_img) * (max_length)
        extend_phrase_seq_mask : (batch*seq_per_img) * (max_length)
        """
        memory = self.encode(src, src_mask)
        predict_phrase_length_logprob, predict_phrase_syn_logprob = \
            self.get_predict_phrase_length_syn_SA(memory, src_mask, phrase_num, phrase_length, labels)
        predict_phrase = self.decode_SA(memory, extend_phrase_seq, extend_phrase_syn_seq[:, 1:-1], src_mask, extend_phrase_seq_mask)
        return predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_SA(self, memory, src_mask, phrase_num, phrase_length, labels):
        B, L = phrase_length.shape[0:2]
        word_seq = labels.clone().long()
        word_seq[:, 0] = self.len_idx
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, SA_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, SA_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :]

    def get_predict_phrase_length_syn_part_SA(self, word_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.tgt_embed(word_seq)), memory, src_mask, tgt_mask)
    
    def decode_SA(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        input_embed = self.pos_embed(self.tgt_embed(word_seq) + self.syn_embed(syn_seq))
        return self.decoder(input_embed, memory, src_mask, tgt_mask)


"""
this part is for predict length and syntactic label with [LEN] markbit
Non-Autoregressive and Semi-Autoregressive Unified arch
"""
U_LENGTH_DIM=20
U_SYN_DIM=10
U_SYN_LOWER=4
U_SYN_UPPER=6
class LengthPredictor_UIC(nn.Module):
    """
    Unified Length Predictor
    each time generate one phrase length and one phrase syntactic label
    """
    def __init__(self, d_model, length_attn, ff, N_len, dropout=0.1):
        super(LengthPredictor_UIC, self).__init__()
        self.d_model = d_model
        self.length_attn = length_attn
        self.ff = ff
        self.N_len = N_len
        self.norm = LayerNorm(d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.Length_classifier1 = nn.Linear(d_model, 100)
        self.Length_classifier2 = nn.Linear(100, U_LENGTH_DIM)
        self.Syntactic_classifier1 = nn.Linear(d_model, 100)
        self.Syntactic_classifier2 = nn.Linear(100, U_SYN_DIM)
        # if N_len = 0, only do cross attention once
        if self.N_len == 0:
            self.LengthPredictor = SublayerConnection(d_model, dropout)
        else:
            c = copy.deepcopy
            self.LengthPredictor = clones(LengthPredictorLayer(d_model, c(length_attn), c(length_attn), c(ff), dropout), N_len)

    def forward(self, input_embed, memory, src_mask, tgt_mask):
        """
        memory: (batch*seq_per_img) * boxes * d_model
        input_embed: (batch*seq_per_img) * (max_length + 2) * d_model

        predict_phrase_length_N: (batch*seq_per_img)
        predict_phrase_length_logprob: (batch*seq_per_img) * SA_LENGTH_DIM
        predict_phrase_syn_N: (batch*seq_per_img)
        predict_phrase_syn_logprob: (batch*seq_per_img) * SA_LENGTH_DIM
        """
        x = input_embed
        m = memory
        if self.N_len == 0:
            output = self.norm(self.LengthPredictor(x, lambda x: self.length_attn(x, m, m, src_mask)))
        else:
            for layer in self.LengthPredictor:
                x = layer(x, m, src_mask, tgt_mask)
            output = self.norm(x)
        output = output[:, 0, :] # we only need [LEN] hidden state
        length_output = self.Dropout(F.relu(self.Length_classifier1(output)))
        predict_phrase_length_logprob = F.log_softmax(self.Length_classifier2(length_output), dim=-1)
        syn_output = self.Dropout(F.relu(self.Syntactic_classifier1(output)))
        predict_phrase_syn_logprob = F.log_softmax(self.Syntactic_classifier2(syn_output), dim=-1)

        _, predict_phrase_length_N = torch.max(predict_phrase_length_logprob.data, 1)
        _, predict_phrase_syn_N = torch.max(predict_phrase_syn_logprob.data, 1)
        return predict_phrase_length_N.int(), predict_phrase_length_logprob, predict_phrase_syn_N.long(), predict_phrase_syn_logprob


class EncoderDecoder_UIC(nn.Module):
    """
    Unified Encoder-Decoder with LengthPredictor_UIC
    """
    def __init__(self, opt, d_model, encoder, decoder, src_embed, syn_embed, tgt_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_UIC, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        # self.norm = LayerNorm(d_model)
        # self.Dropout = nn.Dropout(p=0.1)
        self.input_mode = getattr(opt, 'decoder_input_mode', 'add')
        if self.input_mode == 'gate':
            self.SAIC_gate = nn.Linear(d_model*2, d_model)
            self.NAIC_gate = nn.Linear(d_model*2, d_model)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, labels, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask, glat_p=0.3):
        r"""
        labels: (batch*seq_per_img) * (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_syn_seq: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_seq : (batch*seq_per_img) * (max_length)
        extend_phrase_seq_mask : (batch*seq_per_img) * (max_length)
        glat_p : unmasked token ratio, normally from 0.5 to 0.1 annelly.
        """
        torch.cuda.synchronize()
        start = time.time()
        memory = self.encode(src, src_mask)
        torch.cuda.synchronize()
        end = time.time()
        print("encode time:", (end-start)*1000)
        SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob = \
            self.get_predict_phrase_length_syn_SA(memory, src_mask, phrase_num, phrase_length, labels)
        SA_predict_phrase = self.decode_SA(memory, extend_phrase_seq, extend_phrase_syn_seq[:, 1:-1], src_mask, extend_phrase_seq_mask)
        
        NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, syn_mask = \
            self.get_predict_phrase_length_syn_NA(memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq)

        if glat_p >= 0:
            with torch.no_grad():    
                # get predict tokens with no glat decoder input.
                NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
                NA_predict_phrase_logprob = self.generator(NA_predict_phrase)
                # (batch*seq_per_img) * (max_length)
                NA_pred_tokens = NA_predict_phrase_logprob.argmax(-1)
                
                # get predict acc. 
                B = labels.shape[0]
                real_phrase_label = labels[:, 1:-1]
                phrase_mask = labels.new_full(real_phrase_label.shape, False, dtype=torch.bool)
                tokens_length = phrase_length.sum(1) - 1
                # phrase_mask[:, :tokens_length] = True
                for i in range(B):
                    phrase_mask[i, 0:int(tokens_length[i])] = True # because phrase has no eos/bos to compare
                same_num = ((NA_pred_tokens == real_phrase_label) & phrase_mask).sum(1)
                mismatch_prob = (tokens_length - same_num) / tokens_length
                keep_prob = (mismatch_prob * glat_p).unsqueeze(-1) * phrase_mask.float()

                keep_word_mask = (torch.rand(real_phrase_label.shape, device=real_phrase_label.device) < keep_prob).bool()

                bos_input = memory.new_full(real_phrase_label.shape, self.bos_idx, dtype=torch.long)
                glanced_input = bos_input.masked_fill(keep_word_mask, 0) + real_phrase_label.masked_fill(~keep_word_mask, 0)
                
                # std log
                print ("mismatch ratio: {:.3f}. glat_p: {:.1f}".format( mismatch_prob.mean().item(), glat_p))
            NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask, glanced_input)
        else:
            NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
        return SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase, \
            NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        # flops, params = profile(self.encoder, inputs=(self.src_embed(src), src_mask))
        # print("encoder:", flops/1e9, params/1e6)
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_SA(self, memory, src_mask, phrase_num, phrase_length, labels):
        B, L = phrase_length.shape[0:2]
        word_seq = labels.clone().long()
        word_seq[:, 0] = self.len_idx
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        # torch.cuda.synchronize()
        # start = time.time()
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
        # torch.cuda.synchronize()
        # end = time.time()
        # print("box time:", (end-start))
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :]

    def get_predict_phrase_length_syn_part_SA(self, word_seq, memory, src_mask, tgt_mask):
        # flops, params = profile(self.length_predictor, inputs=(self.pos_embed(self.tgt_embed(word_seq)), memory, src_mask, tgt_mask))
        # print("box: ", flops/1e9, params/1e6)
        return self.length_predictor(self.pos_embed(self.tgt_embed(word_seq)), memory, src_mask, tgt_mask)
    
    def decode_SA(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.tgt_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.tgt_embed(word_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.tgt_embed(word_seq)
            frac = F.sigmoid(self.SAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder(input_embed, memory, src_mask, tgt_mask)

    def get_predict_phrase_length_syn_NA(self, memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        B, L = phrase_length.shape[0:2]
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        syn_mask = phrase_length.new_zeros([B, L-2, L-2], dtype=torch.bool)
        for i in range(0, B):
            syn_mask[i, :, :last[i]-1] = True
        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :], syn_mask

    def get_predict_phrase_length_syn_part_NA(self, syn_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.syn_embed(syn_seq)), memory, src_mask, tgt_mask)
    
    def decode_NA(self, memory, syn_seq, src_mask, tgt_mask, glat_input=None):
        if glat_input is None:
            word_seq = memory.new_full(syn_seq.shape, self.bos_idx, dtype=torch.long)
        else:
            word_seq = glat_input

        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.tgt_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.syn_embed(syn_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.tgt_embed(word_seq)
            frac = F.sigmoid(self.NAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        # flops, params = profile(self.decoder, inputs=(input_embed, memory, src_mask, tgt_mask ))
        # print("decoder:", flops/1e9, params/1e6)
        return self.decoder(input_embed, memory, src_mask, tgt_mask)


class EncoderDecoder_UIC_ds(nn.Module):
    """
    Unified Encoder-Decoder with LengthPredictor_UIC
    """
    def __init__(self, opt, d_model, encoder, decoder_SA, decoder_NA, src_embed, syn_embed, tgt_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_UIC_ds, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder_SA = decoder_SA
        self.decoder_NA = decoder_NA
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        # self.norm = LayerNorm(d_model)
        # self.Dropout = nn.Dropout(p=0.1)
        self.input_mode = getattr(opt, 'decoder_input_mode', 'add')
        if self.input_mode == 'gate':
            self.SAIC_gate = nn.Linear(d_model*2, d_model)
            self.NAIC_gate = nn.Linear(d_model*2, d_model)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, labels, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask):
        """
        labels: (batch*seq_per_img) * (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_syn_seq: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_seq : (batch*seq_per_img) * (max_length)
        extend_phrase_seq_mask : (batch*seq_per_img) * (max_length)
        """
        memory = self.encode(src, src_mask)
        SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob = \
            self.get_predict_phrase_length_syn_SA(memory, src_mask, phrase_num, phrase_length, labels)
        SA_predict_phrase = self.decode_SA(memory, extend_phrase_seq, extend_phrase_syn_seq[:, 1:-1], src_mask, extend_phrase_seq_mask)
        
        NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, syn_mask = \
            self.get_predict_phrase_length_syn_NA(memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq)
        NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
        return SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase, \
            NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_SA(self, memory, src_mask, phrase_num, phrase_length, labels):
        B, L = phrase_length.shape[0:2]
        word_seq = labels.clone().long()
        word_seq[:, 0] = self.len_idx
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_SA(word_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :]

    def get_predict_phrase_length_syn_part_SA(self, word_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.tgt_embed(word_seq)), memory, src_mask, tgt_mask)
    
    def decode_SA(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.tgt_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.tgt_embed(word_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.tgt_embed(word_seq)
            frac = F.sigmoid(self.SAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder_SA(input_embed, memory, src_mask, tgt_mask)

    def get_predict_phrase_length_syn_NA(self, memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        B, L = phrase_length.shape[0:2]
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part_NA(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        syn_mask = phrase_length.new_zeros([B, L-2, L-2], dtype=torch.bool)
        for i in range(0, B):
            syn_mask[i, :, :last[i]-1] = True
        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :], syn_mask

    def get_predict_phrase_length_syn_part_NA(self, syn_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.syn_embed(syn_seq)), memory, src_mask, tgt_mask)
    
    def decode_NA(self, memory, syn_seq, src_mask, tgt_mask):
        word_seq = memory.new_full(syn_seq.shape, self.bos_idx, dtype=torch.long)
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.tgt_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.syn_embed(syn_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.tgt_embed(word_seq)
            frac = F.sigmoid(self.NAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder_NA(input_embed, memory, src_mask, tgt_mask)


class EncoderDecoder_UIC_s(nn.Module):
    """
    Unified Encoder-Decoder with LengthPredictor_UIC
    for AIC + SAIC + NAIC (seperatly)
    """
    def __init__(self, opt, d_model, encoder, decoder_A, decoder_SA, decoder_NA, src_embed, syn_embed, word_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_UIC_s, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder_A = decoder_A
        self.decoder_SA = decoder_SA
        self.decoder_NA = decoder_NA
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.word_embed = word_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        self.input_mode = getattr(opt, 'decoder_input_mode', 'add')
        # if self.input_mode == 'gate':
        #     self.SAIC_gate = nn.Linear(d_model*2, d_model)
        #     self.NAIC_gate = nn.Linear(d_model*2, d_model)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, labels, labels_mask, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask):
        """
        labels: (batch*seq_per_img) * (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_syn_seq: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_seq : (batch*seq_per_img) * (max_length)
        extend_phrase_seq_mask : (batch*seq_per_img) * (max_length)
        """
        memory = self.encode(src, src_mask)
        predict_phrase_length_logprob, predict_phrase_syn_logprob, syn_mask = \
            self.get_predict_phrase_length_syn(memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq)
        A_predict_phrase = self.decode_A(memory, labels[:, :-2], extend_phrase_syn_seq[:, 1:-1], src_mask, labels_mask[:, 1:-1, 1:-1])
        SA_predict_phrase = self.decode_SA(memory, extend_phrase_seq, extend_phrase_syn_seq[:, 1:-1], src_mask, extend_phrase_seq_mask)
        NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
        return predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase, SA_predict_phrase, NA_predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_part(self, syn_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.syn_embed(syn_seq)), memory, src_mask, tgt_mask)

    def get_predict_phrase_length_syn(self, memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        B, L = phrase_length.shape[0:2]
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        syn_mask = phrase_length.new_zeros([B, L-2, L-2], dtype=torch.bool)
        for i in range(0, B):
            syn_mask[i, :, :last[i]-1] = True
        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :], syn_mask
    
    def decode_A(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        return self.decoder_A(input_embed, memory, src_mask, tgt_mask)

    def decode_SA(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.word_embed(word_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.word_embed(word_seq)
            frac = F.sigmoid(self.SAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder_SA(input_embed, memory, src_mask, tgt_mask)

    def decode_NA(self, memory, syn_seq, src_mask, tgt_mask):
        word_seq = memory.new_full(syn_seq.shape, self.bos_idx, dtype=torch.long)
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.syn_embed(syn_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.word_embed(word_seq)
            frac = F.sigmoid(self.NAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder_NA(input_embed, memory, src_mask, tgt_mask)


class EncoderDecoder_UIC_u(nn.Module):
    """
    Unified Encoder-Decoder with LengthPredictor_UIC
    for AIC + SAIC + NAIC (seperatly)
    """
    def __init__(self, opt, d_model, encoder, decoder, src_embed, syn_embed, word_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoder_UIC_u, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.syn_embed = syn_embed
        self.word_embed = word_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        self.input_mode = getattr(opt, 'decoder_input_mode', 'add')
        # if self.input_mode == 'gate':
        #     self.SAIC_gate = nn.Linear(d_model*2, d_model)
        #     self.NAIC_gate = nn.Linear(d_model*2, d_model)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
    
    def forward(self, src, src_mask, labels, labels_mask, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask):
        """
        labels: (batch*seq_per_img) * (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) * (max_length + 2)
        phrase_syn: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_syn_seq: (batch*seq_per_img) * (max_length + 2)
        extend_phrase_seq : (batch*seq_per_img) * (max_length)
        extend_phrase_seq_mask : (batch*seq_per_img) * (max_length)
        """
        memory = self.encode(src, src_mask)
        predict_phrase_length_logprob, predict_phrase_syn_logprob, syn_mask = \
            self.get_predict_phrase_length_syn(memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq)
        A_predict_phrase = self.decode_A(memory, labels[:, :-2], extend_phrase_syn_seq[:, 1:-1], src_mask, labels_mask[:, 1:-1, 1:-1])
        SA_predict_phrase = self.decode_SA(memory, extend_phrase_seq, extend_phrase_syn_seq[:, 1:-1], src_mask, extend_phrase_seq_mask)
        NA_predict_phrase = self.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], src_mask, syn_mask)
        return predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase, SA_predict_phrase, NA_predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_predict_phrase_length_syn_part(self, syn_seq, memory, src_mask, tgt_mask):
        return self.length_predictor(self.pos_embed(self.syn_embed(syn_seq)), memory, src_mask, tgt_mask)

    def get_predict_phrase_length_syn(self, memory, src_mask, phrase_num, phrase_length, extend_phrase_syn_seq):
        B, L = phrase_length.shape[0:2]
        tgt_mask = phrase_length.new_zeros([B, L, L], dtype=torch.bool)
        predict_phrase_length_logprob = phrase_length.new_zeros([B, L, U_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprob = phrase_length.new_zeros([B, L, U_SYN_DIM], dtype=torch.float, requires_grad=True)
        last = torch.zeros([B], dtype=torch.int).to(phrase_length.device)
        last[:] = 1
        # predict first phrase info
        tgt_mask[:, :, 0] = True
        cur_phrase_length_N, cur_phrase_length_logprob, \
            cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
        predict_phrase_length_logprob[:, 1] = cur_phrase_length_logprob
        predict_phrase_syn_logprob[:, 1] = cur_phrase_syn_logprob
        
        # predict rest phrase info
        max_phrase_num = max(phrase_num).int()
        for i in range(1, max_phrase_num):
            # skip 0, because idx 0 is bos/len
            for j in range(0, B):
                if phrase_num[j] <= i:
                    continue
                tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                last[j] += phrase_length[j, i]
                tgt_mask[j, 0, :last[j]] = True

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.get_predict_phrase_length_syn_part(extend_phrase_syn_seq, memory, src_mask, tgt_mask)
            predict_phrase_length_logprob[:, i+1] = cur_phrase_length_logprob
            predict_phrase_syn_logprob[:, i+1] = cur_phrase_syn_logprob    

        syn_mask = phrase_length.new_zeros([B, L-2, L-2], dtype=torch.bool)
        for i in range(0, B):
            syn_mask[i, :, :last[i]-1] = True
        return predict_phrase_length_logprob[:, 1:, :], predict_phrase_syn_logprob[:, 1:, :], syn_mask
    
    def decode_A(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        return self.decoder(input_embed, memory, src_mask, tgt_mask)

    def decode_SA(self, memory, word_seq, syn_seq, src_mask, tgt_mask):
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.word_embed(word_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.word_embed(word_seq)
            frac = F.sigmoid(self.SAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder(input_embed, memory, src_mask, tgt_mask)

    def decode_NA(self, memory, syn_seq, src_mask, tgt_mask):
        word_seq = memory.new_full(syn_seq.shape, self.bos_idx, dtype=torch.long)
        if self.input_mode == 'add':
            input_embed = self.pos_embed(self.word_embed(word_seq) + self.syn_embed(syn_seq))
        elif self.input_mode == 'single':
            input_embed = self.pos_embed(self.syn_embed(syn_seq))
        else:
            syn_input = self.syn_embed(syn_seq)
            word_input = self.word_embed(word_seq)
            frac = F.sigmoid(self.NAIC_gate(torch.cat((word_input, syn_input), 2)))
            input_embed = self.pos_embed(frac * word_input + (1-frac) * syn_input)
        return self.decoder(input_embed, memory, src_mask, tgt_mask)


"""
This part is for PBIC
including class: LengthPredictorPB_pad EncoderDecoderPB_pad
"""
def phrase_subsequent_mask(size, phrase_start):
    "Mask out phrase based subsequent positions"
    phrase_masks = np.zeros([1, size, size], dtype='bool')
    max_phrase_num = phrase_start.shape[0]
    if max_phrase_num != 0:
        for i in range(0, max_phrase_num-1):
            phrase_masks[0, phrase_start[i]:phrase_start[i+1], 0:phrase_start[i+1] ] = True
        phrase_masks[0, phrase_start[-1]: ] = True

    return phrase_masks

def CompressEmbedding(word_embed, d_model, seq, phrase_num, phrase_length):
    """
    compress phrase embedding to signle embedding
    """
    B = phrase_num.shape[0]
    max_phrase_num = max(phrase_num).int()

    embed_seq = word_embed(seq)
    compressed_seq = embed_seq.new_zeros([B, max_phrase_num, d_model], dtype=torch.float)
    # compressed_seq = embed_seq.new_zeros([B, max_phrase_num, d_model], dtype=torch.float)
    # torch.cuda.synchronize()
    # T3 = time.time()
    for ix in range(0, B):
        start = 0
        for pix in range(0, phrase_num[ix]):
            compressed_seq[ix, pix] = torch.sum(embed_seq[ix, start:start+phrase_length[ix, pix]], dim=0)
            start += phrase_length[ix, pix]
    # torch.cuda.synchronize()
    # T4 = time.time()
    # print("copy time:{:.3f}".format(T4-T3))
    return compressed_seq


class LengthPredictorLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, ff, dropout):
        super(LengthPredictorLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff= ff
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.ff)


class LengthPredictorPB_pad(nn.Module):
    """
    reshape phrase into (batch*seq_per_img) * (max_phrase_num)
    """
    def __init__(self, d_model, word_embed, pos_embed, length_attn, ff, N_len, dropout=0.1):
        super(LengthPredictorPB_pad, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.pos_embed = pos_embed
        self.length_attn = length_attn
        self.ff = ff
        self.N_len = N_len
        # self.sublayer = SublayerConnection(d_model, dropout)
        self.norm = LayerNorm(d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.L1 = nn.Linear(d_model, 100)
        self.L2 = nn.Linear(100, 20)
        if self.N_len == 0:
            self.LengthPredictor = SublayerConnection(d_model, dropout)
        else:
            c = copy.deepcopy
            self.LengthPredictor = clones(LengthPredictorLayer(d_model, c(length_attn), c(length_attn), c(ff), dropout), N_len)

    def forward(self, seq, memory, src_mask, phrase_num, phrase_length):
        """
        memory: (batch*seq_per_img) × boxes × d_model
        seq: (batch*seq_per_img) × (max_length + 2)
        phrase_num: (batch*seq_per_img)
        phrase_length: (batch*seq_per_img) × max_length
        x: (batch*seq_per_img) × max_phrase_num × d_model
        """
        # torch.cuda.synchronize()
        # T1 = time.time()
        x = self.pos_embed(CompressEmbedding(self.word_embed, self.d_model, seq, phrase_num, phrase_length))
        m = memory
        # torch.cuda.synchronize()
        # T2 = time.time()
        if self.N_len == 0:
            output = self.norm(self.LengthPredictor(x, lambda x: self.length_attn(x, m, m, src_mask)))
        else:
            tgt_mask = x.new_zeros([x.shape[0], x.shape[1]], dtype=torch.bool)
            for i in range(x.shape[0]):
                tgt_mask[i, 0:phrase_num[i]] = True
            tgt_mask = tgt_mask.unsqueeze(-2)
            tgt_mask = tgt_mask & subsequent_mask(tgt_mask.size(-1)).to(tgt_mask)
            for layer in self.LengthPredictor:
                x = layer(x, m, src_mask, tgt_mask)
            output = self.norm(x)
        
        # torch.cuda.synchronize()
        # T3 = time.time()
        output = self.Dropout(F.relu(self.L1(output)))
        predict_phrase_length_logprob = F.log_softmax(self.L2(output), dim=-1)
        # torch.cuda.synchronize()
        # T4 = time.time()
        predict_phrase_length_p, predict_phrase_length_N = torch.max(predict_phrase_length_logprob.data, 2)
        # print("compress time:{:.3f}  length attention:{:.3f}  FFN:{:.3f}".format(T2-T1, T3-T2, T4-T3))
        return predict_phrase_length_N.int(), predict_phrase_length_logprob

class EncoderDecoderPB_pad(nn.Module):
    """
    phrase based Encoder-Decoder
    """
    def __init__(self, opt, d_model, encoder, decoder, src_embed, tgt_embed, pos_embed, generator, length_predictor):
        super(EncoderDecoderPB_pad, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        # here tgt embed is just do a lookup in vocab, because pos embed should be done after compress
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.length_predictor = length_predictor
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.sep_idx = getattr(opt, 'sep_idx', 3)
        self.norm_mode = getattr(opt, 'norm_mode', 'compress')
    
    def forward(self, src, phrase_num, phrase_length, seq, src_mask):
        "return two tuples: length and word predict"
        # torch.cuda.synchronize()
        # T1 = time.time()
        memory = self.encode(src, src_mask)
        # torch.cuda.synchronize()
        # T2 = time.time()
        predict_phrase_length, predict_phrase_length_logprob = self.get_length_predict_N(seq, memory, src_mask, phrase_num, phrase_length)
        # torch.cuda.synchronize()
        # T3 = time.time()
        predict_phrase = self.decode(memory, src_mask, phrase_num, phrase_length, phrase_length[:, 1:], seq)
        # torch.cuda.synchronize()
        # T4 = time.time()
        # print("encode time:{:.3f}  predict length time:{:.3f}  predict phrase time:{:.3f}".format(T2 - T1, T3 - T2, T4 - T3))
        return predict_phrase_length, predict_phrase_length_logprob, predict_phrase

    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_length_predict_N(self, seq, memory, src_mask, phrase_num, phrase_length):
        # return length_predict:(nbatch*seq_per_img) × max_phrase_num
        # at now max_phrase_num = real max phrase num + 1   not +2 !
        return self.length_predictor(seq, memory, src_mask, phrase_num, phrase_length)
    
    def decode(self, memory, src_mask, phrase_num, phrase_length, next_phrase_length, seq):
        # this part shape decoder input according to seq and phrase_num and phrase_length
        # torch.cuda.synchronize()
        # T1 = time.time()
        
        B = phrase_num.shape[0]
        _phrase_num = phrase_num.cpu().numpy()
        _phrase_length = phrase_length.cpu().numpy()
        _next_phrase_length = next_phrase_length.cpu().numpy()
        _seq = seq.cpu().numpy()
        max_phrase_num = max(_phrase_num)
        max_phrase_length = np.max(_next_phrase_length, axis=0)
        
        # max_phrase_length, max_phrase_length_idx = torch.max(next_phrase_length.data, 0)
        # _next_phrase_length = next_phrase_length.cpu().numpy()
        # max_phrase_length = np.zeros([max_phrase_num], dtype='int')
        # for ix in range(0, max_phrase_num):
        #     max_phrase_length[ix] = max([_next_phrase_length[i, ix] for i in range(0, B)])

        phrase_start = np.zeros([max_phrase_num], dtype='int')
        for ix in range(1, max_phrase_num):
            phrase_start[ix] = phrase_start[ix-1] + max_phrase_length[ix-1]
        # torch.cuda.synchronize()
        # T3 = time.time()
        L = (sum(max_phrase_length))

        # phrase_masks = memory.new_full([B, L], False, dtype=torch.bool)
        phrase = np.zeros([B, L], dtype='long')
        phrase_masks = np.zeros([B, L], dtype='bool')
        if self.norm_mode == 'compress' or self.norm_mode == 'complex':
            compressed_seq = CompressEmbedding(self.tgt_embed, self.d_model, seq, phrase_num, phrase_length)
            fake_seq = memory.new_zeros([B, L], dtype=torch.long)
            embed_phrase = self.tgt_embed(fake_seq)
        # for i in range(0, B):
        #     for j in range(L):
        #         embed_phrase[i, j] = pad_emb
        if self.norm_mode == 'compress':
            for i in range(0, B):
                for j in range(0, _phrase_num[i]):
                    embed_phrase[i, phrase_start[j]:phrase_start[j]+next_phrase_length[i, j] ] = compressed_seq[i, j]
                    phrase_masks[i, phrase_start[j]:phrase_start[j]+_next_phrase_length[i, j] ] = True
        elif self.norm_mode == 'copy' or self.norm_mode == 'complex':
            for i in range(0, B):
                start = 0
                for j in range(0, _phrase_num[i]):
                    if _next_phrase_length[i, j] <= _phrase_length[i, j]:
                        pre_pad = _phrase_length[i, j] - _next_phrase_length[i, j] 
                        phrase[i, phrase_start[j]:phrase_start[j]+_next_phrase_length[i, j] ] = _seq[i, start+pre_pad:start+pre_pad+_next_phrase_length[i, j] ]
                    else:
                        pre_less = _phrase_length[i, j] - (_next_phrase_length[i, j] % _phrase_length[i, j])
                        copy_times = _next_phrase_length[i, j] // _phrase_length[i, j]
                        copied = 0
                        for k in range(0, _phrase_length[i, j]):
                            if k < pre_less:       
                                phrase[i, phrase_start[j]+copied:phrase_start[j]+copied+copy_times] = _seq[i, start+k]
                                copied += copy_times
                            else:
                                phrase[i, phrase_start[j]+copied:phrase_start[j]+copied+copy_times+1] = _seq[i, start+k]
                                copied += (copy_times + 1)
                    start += phrase_length[i, j]
                    phrase_masks[i, phrase_start[j]:phrase_start[j]+_next_phrase_length[i, j] ] = True
            embed_phrase = self.tgt_embed(torch.from_numpy(phrase).to(memory.device))
            if self.norm_mode == 'complex':
                for i in range(0, B):
                    for j in range(0, _phrase_num[i]):
                        embed_phrase[i, phrase_start[j]:phrase_start[j]+_next_phrase_length[i, j] ] += compressed_seq[i, j]
                
        phrase_masks = phrase_masks[:, np.newaxis, :]
        phrase_masks = phrase_masks & phrase_subsequent_mask(L, phrase_start)
        # torch.cuda.synchronize()
        # T4 = time.time()
        ret = self.decoder(self.pos_embed(embed_phrase), memory, src_mask, torch.from_numpy(phrase_masks).to(memory.device))
        # torch.cuda.synchronize()
        # T5 = time.time()
        # print("compress time:{:.3f}  prepare time:{:.3f}  prepare time:{:.3f}  decode time:{:.3f}".format(T2-T1, T3-T2, T4-T3, T5-T4))
        return ret


"""
This part is for naive parallel
including class: LengthPredictorNNAIC EncoderDecoderNNAIC
"""

class LengthPredictor_NNAIC(nn.Module):
    """
    predict captioning length.
    """
    def __init__(self, d_model, max_boxes, max_length, dropout=0.1):
        super(LengthPredictor_NNAIC, self).__init__()
        self.d_model = d_model
        self.max_boxes = max_boxes
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)
        self.L1 = nn.Linear(d_model, self.max_length)
        self.L2 = nn.Linear((self.max_boxes * (self.max_length)), self.max_length)

    def forward(self, memory):
        """
        memory: (nbatch*seq_per_img) × boxes × d_model
        length_N: (nbatch*seq_per_img)
        """
        B,N = memory.shape[0:2]
        assert N <= self.max_boxes 

        if N < self.max_boxes:
            tmp = memory.new_zeros(B, self.max_boxes-N, self.d_model)
            memory = torch.cat((memory, tmp), 1)

        length_per_box = self.dropout(F.relu(self.L1(memory)))
        length_per_box = length_per_box.reshape(B, -1)
        predict_length_logprob = F.softmax(self.L2(length_per_box), dim=-1)
        length_p, length_N = torch.max(predict_length_logprob.data, 1)
        
        return length_N.view(-1).int() # (nbatch*seq_per_img)

class EncoderDecoder_NNAIC(nn.Module):
    """
    Naive Non-Autoregressive 
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, length_predictor):
        super(EncoderDecoder_NNAIC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.length_predictor = length_predictor

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        length_predictor = self.get_length_predict_N(memory)
        word_predictor = self.decode(memory, src_mask, tgt, tgt_mask)
        return length_predictor, word_predictor
    
    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)

    def get_length_predict_N(self, memory):
        return self.length_predictor(memory)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


"""
traditional Autoregressive Transformer arch
"""
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        # input: att_feats  att_masks
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) 
   
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    # x: nbatch * feats * embedding  mask: nbatch * feats * feats    
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # draw = False
        # deep = 1
        # for layer in self.layers:
        #     deep += 1
        #     if deep == 6:
        #         draw = True
        #     x = layer(x, memory, src_mask, tgt_mask, draw)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, draw=False):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, draw))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') 
    return torch.from_numpy(subsequent_mask) == 0 

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # scores' shape is nbatch × h × q_len × k_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf')) 
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, draw=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        if draw:
            glo.add_value(self.attn)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) 
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, N_len=0,
               d_model=512, d_ff=2048, h=8, dropout=0.1, max_boxes=100, max_length=20):
        "Helper: Construct a model from hyperparameters. tgt_vocab is type of int"
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        if self.train_mode == 'AIC' or self.train_mode == 'auto':
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                Generator(d_model, tgt_vocab))
        elif self.train_mode == 'NNAIC':
            model = EncoderDecoder_NNAIC(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                Generator(d_model, tgt_vocab),
                LengthPredictor_NNAIC(d_model, max_boxes, max_length, dropout))
        elif self.train_mode == 'NAIC':
            model = EncoderDecoder_NAIC(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, NA_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_NAIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'SAIC':
            model = EncoderDecoder_SAIC(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, SA_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_SAIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'UIC':
            model = EncoderDecoder_UIC(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, U_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_UIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'UIC_ds':
            model = EncoderDecoder_UIC_ds(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, U_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_UIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'UIC_s':
            model = EncoderDecoder_UIC_s(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, U_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_UIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'UIC_u':
            model = EncoderDecoder_UIC_u(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, U_SYN_DIM),
                Embeddings(d_model, tgt_vocab),
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictor_UIC(d_model, c(attn), c(ff), N_len, dropout))
        elif self.train_mode == 'PB_pad':
            model = EncoderDecoderPB_pad(self.opt, d_model,
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout), N_dec),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                Embeddings(d_model, tgt_vocab), 
                c(position),
                Generator(d_model, tgt_vocab),
                LengthPredictorPB_pad(d_model, Embeddings(d_model, tgt_vocab), c(position), c(attn), c(ff), N_len, dropout))
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.N_len = getattr(opt, 'N_len', 0)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        # self.train_mode = getattr(opt, 'train_mode', 'auto')
        self.max_length = getattr(opt, 'max_length', 20) or opt.seq_length
        self.max_boxes = getattr(opt, 'max_boxes', 100)
        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        self.model = self.make_model(0, self.tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            N_len=self.N_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout,
            max_boxes=self.max_boxes,
            max_length=self.max_length
            )

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks
    
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.bool)
        att_masks = att_masks.unsqueeze(-2)
        #att_masks: (B*seq_per_img) * 1 * max_feats_per_img
        #att_feats: (B*seq_per_img) * max_feats_per_img * 512

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx) 
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2) # (nbatch*seq_per_img)*1*seq_len
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask) 
            #最终的seq: (nbatch*seq_per_img) × seq_len  
            #最终的seq_mask: (nbatch*seq_per_img) × seq_len × seq_len

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, phrase_num=None, phrase_length=None, phrase_syn=None, extend_phrase_syn_seq=None, extend_phrase_seq=None, extend_phrase_seq_mask=None, glat_p=-1.0):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
            if phrase_num is not None:
                phrase_num = phrase_num.reshape(-1)
                phrase_length = phrase_length.reshape(-1, phrase_length.shape[2])
                phrase_syn = phrase_syn.reshape(-1, phrase_syn.shape[2])
                extend_phrase_syn_seq = extend_phrase_syn_seq.reshape(-1, extend_phrase_syn_seq.shape[2])
                extend_phrase_seq = extend_phrase_seq.reshape(-1, extend_phrase_seq.shape[2])
                extend_phrase_seq_mask = extend_phrase_seq_mask.reshape(-1, extend_phrase_seq.shape[1], extend_phrase_seq.shape[1])

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        
        #seq: (nbatch*seq_per_img) × seq_len
        #seq_mask: (nbatch*seq_per_img) × seq_len × seq_len
        #att_feats: (nbatch*seq_per_img) × max_feats_per_img × 512
        #att_mask: (nbatch*seq_per_img) × 1 × max_feats_per_img
        if self.train_mode == 'AIC'  or self.train_mode == 'auto':
            out = self.model(att_feats, seq, att_masks, seq_mask) 
            outputs = self.model.generator(out)
            return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        elif self.train_mode == 'NNAIC':
            new_seq = seq.new_full(seq.shape, self.bos_idx, dtype=torch.long)
            zeros = torch.zeros_like(new_seq)
            # seq[...] = self.bos_idx
            if seq is not None:
                NA_seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
                NA_seq_mask[:, 0] = True
                new_seq = torch.where(NA_seq_mask == True, new_seq, zeros)
                NA_seq_mask = NA_seq_mask.unsqueeze(-2)
            else:
                NA_seq_mask = None
            length_N, word = self.model(att_feats, new_seq.to(att_feats.device), att_masks, NA_seq_mask)
            word_logprob = self.model.generator(word)
            return length_N, word_logprob
        elif self.train_mode == 'NAIC':
            predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase = \
                self.model(att_feats, att_masks, phrase_num, phrase_length, extend_phrase_syn_seq)
            predict_phrase_logprob = self.model.generator(predict_phrase)
            return predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob
        elif self.train_mode == 'SAIC':
            predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase = \
                self.model(att_feats, att_masks, seq, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
            predict_phrase_logprob = self.model.generator(predict_phrase)
            return predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob
        elif self.train_mode == 'UIC' or self.train_mode == 'UIC_ds':
            if self.ss_prob > 0:
                memory = self.model.encode(att_feats, att_masks)
                SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob = \
                    self.ss_SAIC(memory, att_masks, seq, phrase_num, phrase_length, phrase_syn)
                NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, syn_mask = \
                    self.model.get_predict_phrase_length_syn_NA(memory, att_masks, phrase_num, phrase_length, extend_phrase_syn_seq)
                NA_predict_phrase = self.model.decode_NA(memory, extend_phrase_syn_seq[:, 1:-1], att_masks, syn_mask)
                NA_predict_phrase_logprob = self.model.generator(NA_predict_phrase)
            else:
                SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase, \
                    NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase = \
                    self.model(att_feats, att_masks, seq, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask, glat_p)
                SA_predict_phrase_logprob = self.model.generator(SA_predict_phrase)
                NA_predict_phrase_logprob = self.model.generator(NA_predict_phrase)
            return SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
                NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob
        elif self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
            predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase, SA_predict_phrase, NA_predict_phrase = \
                self.model(att_feats, att_masks, seq, seq_mask, phrase_num, phrase_length, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
            A_predict_phrase_logprob = self.model.generator(A_predict_phrase)
            SA_predict_phrase_logprob = self.model.generator(SA_predict_phrase)
            NA_predict_phrase_logprob = self.model.generator(NA_predict_phrase)
            A_predict_phrase_prob = F.softmax(self.logit(A_predict_phrase), dim=-1)
            SA_predict_phrase_prob = F.softmax(self.logit(SA_predict_phrase), dim=-1)
            return predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob, \
                    SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob
        elif self.train_mode == 'PB_pad':
            predict_length, predict_length_logprob, predict_phrase = self.model(att_feats, phrase_num, phrase_length, seq, att_masks)
            predict_phrase_logprob = self.model.generator(predict_phrase)
            return predict_length, predict_length_logprob, predict_phrase_logprob

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)  # batch_size*sample × 1
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1) 
        # decode(memory, src_mask, tgt, tgt_mask)    
        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
    
    def core_NNAIC(self, B, memory, att_mask):
        tgt = memory.new_full([B, self.max_length], self.bos_idx, dtype=torch.long)
        # tgt_mask = subsequent_mask(tgt.size(1)).to(memory.device)
        length_N = self.model.get_length_predict_N(memory)
        tgt_mask = memory.new_ones(tgt.shape, dtype=torch.bool)
        for index, v in enumerate(length_N):
            tgt_mask[index, v:] = 0
            tgt[index, v:] = self.pad_idx
        tgt_mask = tgt_mask.unsqueeze(-2)
        # tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(tgt_mask)
        # print(memory.size())
        # print(mask.size())
        # print(tgt.size())
        # print(tgt_mask.size())
        # exit()
        return length_N, self.model.decode(memory, att_mask, tgt, tgt_mask.to(memory.device))

    def core_NAIC(self, B, memory, att_masks):
        # seq_length + 2 is for [LEN] and [EOS]
        phrase_num = memory.new_full([B], 0, dtype=torch.int)
        phrase_length = memory.new_full([B, self.seq_length + 2], 0, dtype=torch.int)
        phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        tgt_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        last = memory.new_zeros([B], dtype=torch.int) # last[i] records extend_phrase_syn[i] last time end position
        finished = memory.new_zeros([B], dtype=torch.bool)

        for i in range(self.seq_length):
            # prepare [LEN] info
            if i == 0:
                extend_phrase_syn[:, 0] = self.len_idx
                tgt_mask[:, :, 0] = True
                last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part_NA(extend_phrase_syn, memory, att_masks, tgt_mask)

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < NA_SYN_LOWER or cur_phrase_syn_N[j] > NA_SYN_UPPER:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + last[j] >= self.seq_length + 1:
                    cur_phrase_length_N[j] = self.seq_length + 1 - last[j]
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    finished[j] = True
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
                else:
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
            
            if False not in finished:
                break
        syn_mask = phrase_syn.new_zeros([B, self.seq_length, self.seq_length], dtype=torch.bool)
        for i in range(B):
            syn_mask[i, :, :last[j]-1] = True
        
        phrase = self.model.decode_NA(memory, extend_phrase_syn[:, 1:-1], att_masks, syn_mask)
        return phrase, phrase_num, phrase_length[:, :-2], phrase_syn[:, :-2]
    
    def core_SAIC(self, B, memory, att_masks, sample_method, temperature, train_mode, output_logsoftmax=1):
        phrase_num = memory.new_full([B], 0, dtype=torch.int)
        phrase_length = memory.new_full([B, self.seq_length + 2], 0, dtype=torch.int)
        phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        seq = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        seq_logprobs = memory.new_zeros([B, self.seq_length+2, self.tgt_vocab], dtype=torch.float, requires_grad=True)

        # extend_phrase_len is prepare for length_predictor input; phrase_len_mask is corresponding masks
        # extend_phrase and extend_phrase_syn is for decoder
        extend_phrase_len = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        extend_phrase = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        phrase_len_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        phrase_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        
        finished = memory.new_zeros([B], dtype=torch.bool)
        # seq_last step after phrase_last one phrase length
        seq_last = memory.new_zeros([B], dtype=torch.int)
        phrase_last = memory.new_zeros([B], dtype=torch.int)

        for i in range(1, self.seq_length+1):
            # prepare [LEN] info
            if i == 1:
                seq[:, 0] = self.bos_idx
                phrase_length[:, 0] = 1
                extend_phrase_len[:, 0] = self.len_idx
                phrase_len_mask[:, :, 0] = True
                phrase_last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part_SA(extend_phrase_len.clone(), memory, att_masks, phrase_len_mask)

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < SA_SYN_LOWER or cur_phrase_syn_N[j] > SA_SYN_UPPER:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + phrase_last[j] >= self.seq_length + 1:
                    cur_phrase_length_N[j] = self.seq_length + 1 - phrase_last[j]
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    finished[j] = True
                else:
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1

            for j in range(B):
                # can't use finished, because someone maybe turn to ture while it still need to be predicted this time 
                if phrase_length[j, i] == 0:
                    continue
                extend_phrase_syn[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                # positionwise copy
                if phrase_length[j, i] <= phrase_length[j, i-1]:
                    pre_pad = phrase_length[j, i-1] - phrase_length[j, i]
                    extend_phrase[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = seq[j, seq_last[j]+pre_pad:seq_last[j]+pre_pad+phrase_length[j, i]]
                else:
                    pre_less = phrase_length[j, i-1] - (phrase_length[j, i] % phrase_length[j, i-1])
                    copy_times = phrase_length[j, i] // phrase_length[j, i-1]
                    copied = 0
                    for k in range(phrase_length[j, i-1]):
                        if k < pre_less:
                            extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times] = seq[j, seq_last[j]+k]
                            copied += copy_times
                        else:
                            extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times+1] = seq[j, seq_last[j]+k]
                            copied += (copy_times+1)
                phrase_mask[j, phrase_last[j]:, :phrase_last[j]+phrase_length[j, i]] = True
            
            phrase = self.model.decode_SA(memory, extend_phrase.clone()[:, 1:-1], extend_phrase_syn.clone()[:, 1:-1], att_masks, phrase_mask[:, 1:-1, 1:-1])
            if output_logsoftmax:
                phrase_logprobs = F.log_softmax(self.logit(phrase), dim=2)
            else:
                phrase_logprobs = self.logit(phrase)

            if True in phrase_logprobs.isnan():
                print("phrase nan!")
                return seq[:, 1:-1], seq_logprobs[:, 1:-1, :].detach(), phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]    
            # try:
                # mk = phrase_logprobs.gt(0.0)
                # if True in mk:
                #     print("has gt 0")
                # if False in mk:
                #     print("has ls 0")

            phrase, _ = self.sample_next_word(phrase_logprobs, sample_method, temperature, train_mode)
            
            for j in range(B):
                if phrase_length[j, i] == 0:
                    continue
                seq[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                seq_logprobs[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase_logprobs[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                extend_phrase_len[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                phrase_len_mask[j, phrase_last[j]:, :phrase_last[j]+phrase_length[j, i]] = True
                phrase_last[j] += phrase_length[j, i]
                phrase_len_mask[j, 0, :phrase_last[j]] = True
                seq_last[j] += phrase_length[j, i-1]
                
            # except Exception as e:
            #     # simply desert this batch data
            #     return seq[:, 1:-1], seq_logprobs[:, 1:-1, :], phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]
            
            if False not in finished:
                break
        
        return seq[:, 1:-1], seq_logprobs[:, 1:-1, :], phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]

    def ss_SAIC(self, memory, src_mask, labels, phrase_num, phrase_length, phrase_syn, sample_method='greedy', temperature=1.0, train_mode='SAIC', output_logsoftmax=1):
        B, L = phrase_length.shape[0:2]
        predict_phrase_num = memory.new_full([B], 0, dtype=torch.int)
        predict_phrase_length = memory.new_full([B, L], 0, dtype=torch.int)
        predict_phrase_syn = memory.new_full([B, L], self.pad_idx, dtype=torch.long)
        predict_phrase_length_logprobs = memory.new_zeros([B, L, SA_LENGTH_DIM], dtype=torch.float, requires_grad=True)
        predict_phrase_syn_logprobs = memory.new_zeros([B, L, SA_SYN_DIM], dtype=torch.float, requires_grad=True)
        seq = memory.new_full([B, L], self.pad_idx, dtype=torch.long)
        seq_logprobs = memory.new_zeros([B, L, self.tgt_vocab], dtype=torch.float, requires_grad=True)

        # extend_phrase_len is prepare for length_predictor input; phrase_len_mask is corresponding masks
        # extend_phrase and extend_phrase_syn is for decoder
        extend_phrase_len = memory.new_full([B, L], self.pad_idx, dtype=torch.long)
        extend_phrase = memory.new_full([B, L], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, L], self.pad_idx, dtype=torch.long)
        phrase_len_mask = memory.new_zeros([B, L, L], dtype=torch.bool)
        phrase_mask = memory.new_zeros([B, L, L], dtype=torch.bool)
        
        finished = memory.new_zeros([B], dtype=torch.bool)
        # seq_last step after phrase_last one phrase length
        label_last = memory.new_zeros([B], dtype=torch.int)
        seq_last = memory.new_zeros([B], dtype=torch.int)
        phrase_last = memory.new_zeros([B], dtype=torch.int)

        for i in range(1, L):
            # prepare [LEN] info
            if i == 1:
                seq[:, 0] = self.bos_idx
                predict_phrase_length[:, 0] = 1
                extend_phrase_len[:, 0] = self.len_idx
                phrase_len_mask[:, :, 0] = True
                phrase_last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part_SA(extend_phrase_len.clone(), memory, src_mask, phrase_len_mask)
            predict_phrase_length_logprobs[:, i] = cur_phrase_length_logprob
            predict_phrase_syn_logprobs[:, i] = cur_phrase_syn_logprob

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < SA_SYN_LOWER or cur_phrase_syn_N[j] > SA_SYN_UPPER or phrase_length[j, i] == 0:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + phrase_last[j] >= L-1:
                    cur_phrase_length_N[j] = L-1 - phrase_last[j]
                    predict_phrase_length[j, i] = cur_phrase_length_N[j]
                    predict_phrase_syn[j, i] = cur_phrase_syn_N[j]
                    predict_phrase_num[j] += 1
                    finished[j] = True
                else:
                    predict_phrase_length[j, i] = cur_phrase_length_N[j]
                    predict_phrase_syn[j, i] = cur_phrase_syn_N[j]
                    predict_phrase_num[j] += 1
                
            for j in range(B):
                # can't use finished, because someone maybe turn to ture while it still need to be predicted this time 
                if predict_phrase_length[j, i] == 0 :
                    continue
                if random() < self.ss_prob:
                    if random() < 0.5:
                        # use last step output as input
                        extend_phrase_syn[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = predict_phrase_syn[j, i]
                        # positionwise copy
                        if predict_phrase_length[j, i] <= predict_phrase_length[j, i-1]:
                            pre_pad = predict_phrase_length[j, i-1] - predict_phrase_length[j, i]
                            extend_phrase[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = seq[j, seq_last[j]+pre_pad:seq_last[j]+pre_pad+predict_phrase_length[j, i]]
                        else:
                            pre_less = predict_phrase_length[j, i-1] - (predict_phrase_length[j, i] % predict_phrase_length[j, i-1])
                            copy_times = predict_phrase_length[j, i] // predict_phrase_length[j, i-1]
                            copied = 0
                            for k in range(predict_phrase_length[j, i-1]):
                                if k < pre_less:
                                    extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times] = seq[j, seq_last[j]+k]
                                    copied += copy_times
                                else:
                                    extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times+1] = seq[j, seq_last[j]+k]
                                    copied += (copy_times+1)
                        phrase_mask[j, phrase_last[j]:, :phrase_last[j]+predict_phrase_length[j, i]] = True
                    else:
                        # use only syn as input
                        extend_phrase_syn[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = predict_phrase_syn[j, i]
                        extend_phrase[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = self.bos_idx
                        phrase_mask[j, phrase_last[j]:, :phrase_last[j]+predict_phrase_length[j, i]] = True
                else:
                    # use gt as input
                    predict_phrase_length[j, i] = min(phrase_length[j, i], L-1-phrase_last[j])
                    extend_phrase_syn[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = phrase_syn[j, i]
                    # positionwise copy
                    if predict_phrase_length[j, i] <= phrase_length[j, i-1]:
                        pre_pad = phrase_length[j, i-1] - predict_phrase_length[j, i]
                        extend_phrase[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = labels[j, label_last[j]+pre_pad:label_last[j]+pre_pad+predict_phrase_length[j, i]]
                    else:
                        pre_less = phrase_length[j, i-1] - (predict_phrase_length[j, i] % phrase_length[j, i-1])
                        copy_times = predict_phrase_length[j, i] // phrase_length[j, i-1]
                        copied = 0
                        for k in range(phrase_length[j, i-1]):
                            if k < pre_less:
                                extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times] = labels[j, label_last[j]+k]
                                copied += copy_times
                            else:
                                extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times+1] = labels[j, label_last[j]+k]
                                copied += (copy_times+1)
                    phrase_mask[j, phrase_last[j]:, :phrase_last[j]+predict_phrase_length[j, i]] = True
                
            phrase = self.model.decode_SA(memory, extend_phrase.clone()[:, 1:-1], extend_phrase_syn.clone()[:, 1:-1], src_mask, phrase_mask[:, 1:-1, 1:-1])
            if output_logsoftmax:
                phrase_logprobs = F.log_softmax(self.logit(phrase), dim=2)
            else:
                phrase_logprobs = self.logit(phrase)

            if True in phrase_logprobs.isnan():
                print("phrase nan!")
                return seq[:, 1:-1], seq_logprobs[:, 1:-1, :].detach(), phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]
            
            phrase, _ = self.sample_next_word(phrase_logprobs, sample_method, temperature, train_mode)
            
            for j in range(B):
                if predict_phrase_length[j, i] == 0:
                    continue
                seq[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+predict_phrase_length[j, i]]
                seq_logprobs[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = phrase_logprobs[j, phrase_last[j]-1:phrase_last[j]-1+predict_phrase_length[j, i]]
                extend_phrase_len[j, phrase_last[j]:phrase_last[j]+predict_phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+predict_phrase_length[j, i]]
                phrase_len_mask[j, phrase_last[j]:, :phrase_last[j]+predict_phrase_length[j, i]] = True
                phrase_last[j] += predict_phrase_length[j, i]
                phrase_len_mask[j, 0, :phrase_last[j]] = True
                seq_last[j] += predict_phrase_length[j, i-1]
                label_last[j] += phrase_length[j, i-1]
            
            if False not in finished:
                break
        
        return predict_phrase_length_logprobs[:, 1:, :], predict_phrase_syn_logprobs[:, 1:, :], seq_logprobs[:, 1:-1, :]

    def corePB_pad(self, memory, att_masks, seq, phrase_num, phrase_length):
        B = memory.shape[0]
        
        next_phrase_length, p = self.model.get_length_predict_N(seq, memory, att_masks, phrase_num, phrase_length)
        _next_phrase_length = next_phrase_length.cpu().numpy()
        _phrase_num = phrase_num.cpu().numpy()
        tmp_phrase_length = np.zeros([B], dtype='int')
        # tmp_phrase_length = memory.new_zeros([B], dtype=torch.int)
        for i in range(0, B):
            tmp_phrase_length[i] = _next_phrase_length[i, _phrase_num[i]-1 ]
        
        phrase = self.model.decode(memory, att_masks, phrase_num, phrase_length, next_phrase_length, seq)

        max_phrase_num = max(_phrase_num)
        # max_phrase_length = np.zeros([max_phrase_num], dtype='int')
        max_phrase_length = np.max(_next_phrase_length, axis=0)
        # for ix in range(0, max_phrase_num):
        #     max_phrase_length[ix] = max([next_phrase_length[i, ix] for i in range(0, B)])
        
        phrase_start = np.zeros([max_phrase_num], dtype='int')
        for ix in range(1, max_phrase_num):
            phrase_start[ix] = phrase_start[ix-1] + max_phrase_length[ix-1]
        
        return torch.from_numpy(tmp_phrase_length).to(memory.device), phrase[:, phrase_start[-1]: ]

    def core_UIC_s_NAIC(self, B, memory, att_masks):
        # seq_length + 2 is for [LEN] and [EOS]
        phrase_num = memory.new_full([B], 0, dtype=torch.int)
        phrase_length = memory.new_full([B, self.seq_length + 2], 0, dtype=torch.int)
        phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        tgt_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        last = memory.new_zeros([B], dtype=torch.int) # last[i] records extend_phrase_syn[i] last time end position
        finished = memory.new_zeros([B], dtype=torch.bool)
        
        for i in range(self.seq_length):
            # prepare [LEN] info
            if i == 0:
                extend_phrase_syn[:, 0] = self.len_idx
                tgt_mask[:, :, 0] = True
                last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part(extend_phrase_syn, memory, att_masks, tgt_mask)

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < NA_SYN_LOWER or cur_phrase_syn_N[j] > NA_SYN_UPPER:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + last[j] >= self.seq_length + 1:
                    cur_phrase_length_N[j] = self.seq_length + 1 - last[j]
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    finished[j] = True
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
                else:
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
            
            if False not in finished:
                break
        syn_mask = phrase_syn.new_zeros([B, self.seq_length, self.seq_length], dtype=torch.bool)
        for i in range(B):
            syn_mask[i, :, :last[i]-1] = True
        
        phrase = self.model.decode_NA(memory, extend_phrase_syn[:, 1:-1], att_masks, syn_mask)
        return phrase, phrase_num, phrase_length[:, :-2], phrase_syn[:, :-2]

    def core_UIC_s_SAIC(self, B, memory, att_masks, sample_method, temperature, output_logsoftmax=1):
        phrase_num = memory.new_full([B], 0, dtype=torch.int)
        phrase_length = memory.new_full([B, self.seq_length + 2], 0, dtype=torch.int)
        phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        seq = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        seq_logprobs = memory.new_zeros([B, self.seq_length+2, self.tgt_vocab], dtype=torch.float, requires_grad=True)

        # extend_phrase_len is prepare for length_predictor input; phrase_len_mask is corresponding masks
        # extend_phrase and extend_phrase_syn is for decoder
        # extend_phrase_len = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        extend_phrase = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        phrase_len_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        phrase_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        
        finished = memory.new_zeros([B], dtype=torch.bool)
        # seq_last step after phrase_last one phrase length
        seq_last = memory.new_zeros([B], dtype=torch.int)
        phrase_last = memory.new_zeros([B], dtype=torch.int)

        for i in range(1, self.seq_length+1):
            # prepare [LEN] info
            if i == 1:
                seq[:, 0] = self.bos_idx
                phrase_length[:, 0] = 1
                extend_phrase_syn[:, 0] = self.len_idx
                phrase_len_mask[:, :, 0] = True
                phrase_last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part(extend_phrase_syn.clone(), memory, att_masks, phrase_len_mask)

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < SA_SYN_LOWER or cur_phrase_syn_N[j] > SA_SYN_UPPER:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + phrase_last[j] >= self.seq_length + 1:
                    cur_phrase_length_N[j] = self.seq_length + 1 - phrase_last[j]
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    finished[j] = True
                else:
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1

            for j in range(B):
                # can't use finished, because someone maybe turn to ture while it still need to be predicted this time 
                if phrase_length[j, i] == 0:
                    continue
                extend_phrase_syn[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                # positionwise copy
                if phrase_length[j, i] <= phrase_length[j, i-1]:
                    pre_pad = phrase_length[j, i-1] - phrase_length[j, i]
                    extend_phrase[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = seq[j, seq_last[j]+pre_pad:seq_last[j]+pre_pad+phrase_length[j, i]]
                else:
                    pre_less = phrase_length[j, i-1] - (phrase_length[j, i] % phrase_length[j, i-1])
                    copy_times = phrase_length[j, i] // phrase_length[j, i-1]
                    copied = 0
                    for k in range(phrase_length[j, i-1]):
                        if k < pre_less:
                            extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times] = seq[j, seq_last[j]+k]
                            copied += copy_times
                        else:
                            extend_phrase[j, phrase_last[j]+copied:phrase_last[j]+copied+copy_times+1] = seq[j, seq_last[j]+k]
                            copied += (copy_times+1)
                phrase_mask[j, phrase_last[j]:, :phrase_last[j]+phrase_length[j, i]] = True
            
            phrase = self.model.decode_SA(memory, extend_phrase.clone()[:, 1:-1], extend_phrase_syn.clone()[:, 1:-1], att_masks, phrase_mask[:, 1:-1, 1:-1])
            if output_logsoftmax:
                phrase_logprobs = F.log_softmax(self.logit(phrase), dim=2)
            else:
                phrase_logprobs = self.logit(phrase)
                
            # try:
                # mk = phrase_logprobs.gt(0.0)
                # if True in mk:
                #     print("has gt 0")
                # if False in mk:
                #     print("has ls 0")
            if True in phrase_logprobs.isnan():
                print("phrase nan!")
                return seq[:, 1:-1], seq_logprobs[:, 1:-1, :].detach(), phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]
            
            phrase, _ = self.sample_next_word(phrase_logprobs, sample_method, temperature, train_mode='SAIC')
            
            for j in range(B):
                if phrase_length[j, i] == 0:
                    continue
                seq[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                seq_logprobs[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase_logprobs[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                # extend_phrase_len[j, phrase_last[j]:phrase_last[j]+phrase_length[j, i]] = phrase[j, phrase_last[j]-1:phrase_last[j]-1+phrase_length[j, i]]
                phrase_len_mask[j, phrase_last[j]:, :phrase_last[j]+phrase_length[j, i]] = True
                phrase_last[j] += phrase_length[j, i]
                phrase_len_mask[j, 0, :phrase_last[j]] = True
                seq_last[j] += phrase_length[j, i-1]
                
            # except Exception as e:
                # simply desert this batch data
            #     return seq[:, 1:-1], seq_logprobs[:, 1:-1, :], phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]
            
            if False not in finished:
                break
        
        return seq[:, 1:-1], seq_logprobs[:, 1:-1, :], phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]    

    def core_UIC_s_AIC(self, B, memory, att_masks, sample_method, temperature, output_logsoftmax=1):
        # seq_length + 2 is for [LEN] and [EOS]
        phrase_num = memory.new_full([B], 0, dtype=torch.int)
        phrase_length = memory.new_full([B, self.seq_length + 2], 0, dtype=torch.int)
        phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        extend_phrase_syn = memory.new_full([B, self.seq_length + 2], self.pad_idx, dtype=torch.long)
        tgt_mask = memory.new_zeros([B, self.seq_length+2, self.seq_length+2], dtype=torch.bool)
        last = memory.new_zeros([B], dtype=torch.int) # last[i] records extend_phrase_syn[i] last time end position
        finished = memory.new_zeros([B], dtype=torch.bool)
        
        for i in range(self.seq_length):
            # prepare [LEN] info
            if i == 0:
                extend_phrase_syn[:, 0] = self.len_idx
                tgt_mask[:, :, 0] = True
                last[:] = 1

            cur_phrase_length_N, cur_phrase_length_logprob, \
                cur_phrase_syn_N, cur_phrase_syn_logprob = self.model.get_predict_phrase_length_syn_part(extend_phrase_syn.clone(), memory, att_masks, tgt_mask)

            for j in range(B):
                if finished[j]:
                    continue
                if cur_phrase_length_N[j] == 0 or cur_phrase_syn_N[j] < NA_SYN_LOWER or cur_phrase_syn_N[j] > NA_SYN_UPPER:
                    # means eos
                    finished[j] = True
                    continue
                elif cur_phrase_length_N[j] + last[j] >= self.seq_length + 1:
                    cur_phrase_length_N[j] = self.seq_length + 1 - last[j]
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    finished[j] = True
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
                else:
                    phrase_length[j, i] = cur_phrase_length_N[j]
                    phrase_syn[j, i] = cur_phrase_syn_N[j]
                    phrase_num[j] += 1
                    extend_phrase_syn[j, last[j]:last[j]+phrase_length[j, i]] = phrase_syn[j, i]
                    tgt_mask[j, last[j]:, :last[j]+phrase_length[j, i]] = True
                    last[j] += phrase_length[j, i]
                    tgt_mask[j, 0, :last[j]] = True
            
            if False not in finished:
                break
        # syn_mask = phrase_syn.new_zeros([B, self.seq_length, self.seq_length], dtype=torch.bool)
        # for i in range(B):
        #     syn_mask[i, :, :last[i]-1] = True
        seq = memory.new_full([B, self.seq_length+2], self.pad_idx, dtype=torch.long)
        seq_logprobs = memory.new_zeros([B, self.seq_length+2, self.tgt_vocab], dtype=torch.float, requires_grad=True)
        # finished = memory.new_zeros([B], dtype=torch.bool)
        for i in range(self.seq_length):
            if i == 0:
                seq[:, 0] = self.bos_idx
            phrase = self.model.decode_A(memory, seq.clone()[:, :i+1], extend_phrase_syn[:, 1:i+2], att_masks, subsequent_mask(i+1).to(memory.device))
            output = phrase[:, -1]

            if output_logsoftmax:
                phrase_logprobs = F.log_softmax(self.logit(output), dim=-1)
            else:
                phrase_logprobs = self.logit(output)
            
            if True in phrase_logprobs.isnan():
                print("phrase nan!")
                return seq[:, 1:-1], seq_logprobs[:, 1:-1, :].detach(), phrase_num, phrase_length[:, 1:-1], phrase_syn[:, 1:-1]
            
            phrase, _ = self.sample_next_word(phrase_logprobs, sample_method, temperature, train_mode='AIC')
            
            for j in range(B):
                if i+1 >= last[j]:
                    continue
                seq[j, i+1] = phrase[j]
                seq_logprobs[j, i+1] = phrase_logprobs[j]

        # phrase = self.model.decode_NA(memory, extend_phrase_syn[:, 1:-1], att_masks, syn_mask)
        return seq[:, 1:-1], seq_logprobs[:, 1:-1, :], phrase_num, phrase_length[:, :-2], phrase_syn[:, :-2]