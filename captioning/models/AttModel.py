# This file contains Att2in2, AdaAtt, AdaAttMO, UpDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# UpDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def sort_pack_padded_sequence(input, lengths):
    #input: B * feats_per_img   lengths: B 
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.train_mode = getattr(opt, 'train_mode', 'AIC')

        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        self.len_idx = getattr(opt, 'len_idx', 3)
        
        self.unk_idx = getattr(opt, 'unk_idx', None)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.tgt_vocab = self.vocab_size + 4

        self.embed = nn.Sequential(nn.Embedding(self.tgt_vocab, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.tgt_vocab)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.tgt_vocab)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

    def init_hidden(self, bsz):
        weight = self.logit.weight \
                 if hasattr(self.logit, "weight") \
                 else self.logit[0].weight
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            #att_feats: B × max_box_per_img × 2048  att_masks: B × max_box_per_img
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)

        outputs = fc_feats.new_zeros(batch_size*seq_per_img, seq.size(1), self.tgt_vocab)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)
        # p_att_feats : memory
        # return state : [(1 × batch_size × predicted_len)]
        # return output: batch_size * d_model
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        
        # logit: linear(rnn_size -> vocab_size)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def get_length_word_logprobs(self, B, memory, att_masks, output_logsoftmax=1):
        length_N, word = self.core_NNAIC(B, memory, att_masks)

        if output_logsoftmax:
            word_logprobs = F.log_softmax(self.logit(word), dim=2)
        else:
            word_logprobs = self.logit(word)
        return length_N, word_logprobs

    def get_word(self, word_logprobs):
        word_p, word_v = torch.max(word_logprobs.data, 2)
        return word_v.to(torch.long)
    
    def get_NA_phrase_length_syn_logprobs(self, B, memory, att_masks, output_logsoftmax=1):
        phrase, phrase_num, phrase_length, phrase_syn = self.core_NAIC(B, memory, att_masks)

        if output_logsoftmax:
            phrase_logprobs = F.log_softmax(self.logit(phrase), dim=2)
        else:
            phrase_logprobs = self.logit(phrase)
        return phrase_logprobs, phrase_num, phrase_length, phrase_syn

    def get_length_phrase_logprobs(self, memory, att_masks, seq, phrase_num, phrase_length, output_logsoftmax=1):
        
        tmp_phrase_length, tmp_phrase = self.corePB_pad(memory, att_masks, seq, phrase_num, phrase_length)

        if output_logsoftmax:
            phrase_logprobs = F.log_softmax(self.logit(tmp_phrase), dim=2)
        else:
            phrase_logprobs = self.logit(tmp_phrase)
        
        return tmp_phrase_length, phrase_logprobs

    def _old_sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 4, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 4)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks = utils.repeat_tensors(beam_size,
                [p_fc_feats[k:k+1], p_att_feats[k:k+1], pp_att_feats[k:k+1], p_att_masks[k:k+1] if att_masks is not None else None]
            )

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_full([beam_size], self.bos_idx, dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.old_beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq[k*sample_n+_n, :] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :] = self.done_beams[k][_n]['logps']
            else:
                seq[k, :] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        torch.cuda.synchronize()
        start = time.time()
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 4, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 4)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        
        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
            [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
        )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        torch.cuda.synchronize()
        end = time.time()
        return seq, seqLogprobs, end-start

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        train_mode = opt.get('train_mode', 'AIC')
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size*sample_n)
        
        #fc_feats = att_feats(no data), att_feats = memory, att_masks = att_masks
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        trigrams = [] # will be a list of batch_size dictionaries
        torch.cuda.synchronize()
        start = time.time()
        if train_mode == 'AIC' or train_mode == 'auto':
            seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
            seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 4)
            
            for t in range(self.seq_length + 1):
                if t == 0: # input <bos>
                    it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)
                #logprobs: batch_size × vocab
                #state: [(1 × batch_size × 1)]
                logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, output_logsoftmax=output_logsoftmax)
                
                if decoding_constraint and t > 0:
                    tmp = logprobs.new_zeros(logprobs.size())
                    tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                    logprobs = logprobs + tmp

                if remove_bad_endings and t > 0:
                    tmp = logprobs.new_zeros(logprobs.size())
                    prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                    # Make it impossible to generate bad_endings
                    tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                    logprobs = logprobs + tmp

                # Mess with trigrams
                # Copy from https://github.com/lukemelas/image-paragraph-captioning
                if block_trigrams and t >= 3:
                    # Store trigram generated at last step
                    prev_two_batch = seq[:,t-3:t-1]
                    for i in range(batch_size): # = seq.size(0)
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        current  = seq[i][t-1]
                        if t == 3: # initialize
                            trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                        elif t > 3:
                            if prev_two in trigrams[i]: # add to list
                                trigrams[i][prev_two].append(current)
                            else: # create list
                                trigrams[i][prev_two] = [current]
                    # Block used trigrams at next step
                    prev_two_batch = seq[:,t-2:t]
                    mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                    for i in range(batch_size):
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        if prev_two in trigrams[i]:
                            for j in trigrams[i][prev_two]:
                                mask[i,j] += 1
                    # Apply mask to log probs
                    #logprobs = logprobs - (mask * 1e9)
                    alpha = 2.0 # = 4
                    logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

                # sample the next word
                if t == self.seq_length: # skip if we achieve maximum length
                    break
                # sample_method=greedy  temp=1.0  it: batch*sample 里面存放预测的单词的index
                it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

                # stop when all finished
                if t == 0:
                    unfinished = it != self.eos_idx
                else:
                    it[~unfinished] = self.pad_idx # This allows eos_idx not being overwritten to 0
                    logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                    unfinished = unfinished & (it != self.eos_idx)
                seq[:,t] = it
                seqLogprobs[:,t] = logprobs                                                         
                # quit loop if all sequences have finished
                if unfinished.sum() == 0:
                    break
            torch.cuda.synchronize()
            end = time.time()
            return seq, seqLogprobs, end-start
        elif train_mode == 'NNAIC':
            length_N, word_logprob = self.get_length_word_logprobs(batch_size*sample_n, pp_att_feats, p_att_masks, output_logsoftmax=output_logsoftmax)
            seq = self.get_word(word_logprob)
            for index, n in enumerate(length_N):
                seq[index, n:] = self.pad_idx
            torch.cuda.synchronize()
            end = time.time()
            return seq, word_logprob, length_N, end - start
        elif train_mode == 'NAIC':
            seq_logprob, phrase_num, phrase_length, phrase_syn = self.get_NA_phrase_length_syn_logprobs(batch_size*sample_n, pp_att_feats, p_att_masks, output_logsoftmax=output_logsoftmax)
            seq, _ = self.sample_next_word(seq_logprob, sample_method, temperature, train_mode=train_mode)
            for index, n in enumerate(phrase_num):
                seq[index, sum(phrase_length[index]):] = self.pad_idx
                # seq_logprob[index, sum(phrase_length[index]):] = 0
            torch.cuda.synchronize()
            end = time.time()
            if True in seq_logprob.isnan():
                return seq, seq_logprob.detach(), phrase_num, phrase_length, phrase_syn, end-start
            return seq, seq_logprob, phrase_num, phrase_length, phrase_syn, end-start
        elif train_mode == 'SAIC':
            seq, seq_logprob, phrase_num, phrase_length, phrase_syn = self.core_SAIC(batch_size*sample_n, pp_att_feats, p_att_masks, \
                sample_method=sample_method, temperature=temperature, train_mode=train_mode, output_logsoftmax=output_logsoftmax)
            torch.cuda.synchronize()
            end = time.time()
            if True in seq_logprob.isnan():
                return seq, seq_logprob.detach(), phrase_num, phrase_length, phrase_syn, end-start
            return seq, seq_logprob, phrase_num, phrase_length, phrase_syn, end-start
        elif train_mode == 'UIC_s_AIC':
            seq, seq_logprob, phrase_num, phrase_length, phrase_syn = self.core_UIC_s_AIC(batch_size*sample_n, pp_att_feats, p_att_masks, \
                sample_method=sample_method, temperature=temperature, output_logsoftmax=output_logsoftmax)
            torch.cuda.synchronize()
            end = time.time()
            if True in seq_logprob.isnan():
                return seq, seq_logprob.detach(), phrase_num, phrase_length, phrase_syn, end-start
            return seq, seq_logprob, phrase_num, phrase_length, phrase_syn, end-start
        elif train_mode == 'UIC_s_NAIC':
            phrase, phrase_num, phrase_length, phrase_syn = self.core_UIC_s_NAIC(batch_size*sample_n, pp_att_feats, p_att_masks)
            seq_logprob = F.log_softmax(self.logit(phrase), dim=-1)
            # seq_logprob, phrase_num, phrase_length, phrase_syn = self.get_NA_phrase_length_syn_logprobs(batch_size*sample_n, pp_att_feats, p_att_masks, output_logsoftmax=output_logsoftmax)
            seq, _ = self.sample_next_word(seq_logprob, sample_method, temperature, train_mode='NAIC')
            for index, n in enumerate(phrase_num):
                seq[index, sum(phrase_length[index]):] = self.pad_idx
                # seq_logprob[index, sum(phrase_length[index]):] = 0
            torch.cuda.synchronize()
            end = time.time()
            if True in seq_logprob.isnan():
                return seq, seq_logprob.detach(), phrase_num, phrase_length, phrase_syn, end-start
            return seq, seq_logprob, phrase_num, phrase_length, phrase_syn, end-start
        elif train_mode == 'UIC_s_SAIC':
            seq, seq_logprob, phrase_num, phrase_length, phrase_syn = self.core_UIC_s_SAIC(batch_size*sample_n, pp_att_feats, p_att_masks, \
                sample_method=sample_method, temperature=temperature, output_logsoftmax=output_logsoftmax)
            torch.cuda.synchronize()
            end = time.time()
            if True in seq_logprob.isnan():
                return seq, seq_logprob.detach(), phrase_num, phrase_length, phrase_syn, end-start
            return seq, seq_logprob, phrase_num, phrase_length, phrase_syn, end-start
        elif train_mode == 'PB_pad':
            # include bos, so the length of seq is seq_length + 1
            B = batch_size*sample_n
            seq = fc_feats.new_full((B, self.seq_length + 1), self.pad_idx, dtype=torch.long)
            seqLogprobs = fc_feats.new_zeros([B, self.seq_length + 1, self.vocab_size + 4], dtype=torch.float)
            phrase_num = fc_feats.new_zeros([B], dtype=torch.int)
            phrase_length = fc_feats.new_zeros([B, self.seq_length + 1], dtype=torch.int)
            current_start = fc_feats.new_zeros([B], dtype=torch.int)
            for t in range(self.seq_length + 1):
                if t == 0:
                    seq[:, 0] = self.bos_idx
                    phrase_num[:] = 1
                    phrase_length[:, 0] = 1
                    unfinished = phrase_num != 0
                    current_start[:] = 1
                
                tmp_phrase_length, phrase_logprobs = self.get_length_phrase_logprobs(pp_att_feats, p_att_masks, seq, phrase_num, phrase_length, output_logsoftmax=output_logsoftmax)

                if t == self.seq_length:
                    break
                
                if phrase_logprobs.shape[1] == 0:
                    break
                
                tmp_phrase, sampleLogprobs = self.sample_next_word(phrase_logprobs, sample_method, temperature, train_mode=train_mode)

                for i in range(0, B):
                    if not unfinished[i]:
                        continue
                    if current_start[i] + tmp_phrase_length[i] >= self.seq_length + 1:
                        tmp_phrase_length[i] = self.seq_length + 1 - current_start[i]
                        phrase_length[i, phrase_num[i] ] = tmp_phrase_length[i]
                        phrase_num[i] += 1
                        seq[i, current_start[i]:current_start[i]+tmp_phrase_length[i] ] = tmp_phrase[i, 0:tmp_phrase_length[i]]
                        seqLogprobs[i, current_start[i]:current_start[i]+tmp_phrase_length[i]] = phrase_logprobs[i, 0:tmp_phrase_length[i]]
                        unfinished[i] = False
                    elif tmp_phrase_length[i] != 0:
                        phrase_length[i, phrase_num[i] ] = tmp_phrase_length[i]
                        phrase_num[i] += 1
                        seq[i, current_start[i]:current_start[i]+tmp_phrase_length[i] ] = tmp_phrase[i, 0:tmp_phrase_length[i]]
                        seqLogprobs[i, current_start[i]:current_start[i]+tmp_phrase_length[i]] = phrase_logprobs[i, 0:tmp_phrase_length[i]]
                    else:
                        unfinished[i] = False
                    
                    current_start[i] = sum(phrase_length[i])
                
                if True not in unfinished:
                    break
            torch.cuda.synchronize()
            end = time.time()
            return seq[:, 1:], seqLogprobs[:, 1:], phrase_num-1, phrase_length[:, 1:], end-start
        
    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)] # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long) for _ in range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.seq_length-1:
                    if t == 0: # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t-1] # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state_table[divm]) # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda
                    
                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if remove_bad_endings and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                        # Impossible to generate remove_bad_endings
                        tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:,t-3:t-1]
                        for i in range(batch_size): # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current  = seq[i][t-1]
                            if t == 3: # initialize
                                trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]: # add to list
                                    trigrams[i][prev_two].append(current)
                                else: # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:,t-2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i,j] += 1
                        # Apply mask to log probs
                        #logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0 # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = (seq[:,t-1] != self.pad_idx) & (seq[:,t-1] != self.eos_idx)
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx) # changed
                    seq[:,t] = it
                    seqLogprobs[:,t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table, 1).reshape(batch_size * group_size, -1)

