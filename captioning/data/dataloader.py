from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from re import X
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
import time
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat

class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        self.pp_mode = getattr(opt, 'preprocess_mode', 'phrase')
        self.train_mode = getattr(opt, 'train_mode', 'AIC')
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 1)
        self.eos_idx = getattr(opt, 'eos_idx', 2)
        # self.len_idx = getattr(opt, 'len_idx', 3)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
            print('vocab[4] is ', self.ix_to_word['4'])
            
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
            # load in the phrase based data if needed
            self.phrase_num, self.phrase_length, self.phrase_syn = None, None, None
            if self.train_mode == 'PB_pad' or self.train_mode == 'NAIC' or self.train_mode == 'SAIC' or self.train_mode == 'UIC' or self.train_mode == 'UIC_ds' or self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
                # self.phrase = self.h5_label_file['phrase'][:]
                self.phrase_num = self.h5_label_file['phrase_num'][:]
                self.phrase_length = self.h5_label_file['phrase_length'][:]
                self.phrase_syn = self.h5_label_file['phrase_label'][:]

        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)
        if self.input_multilabel_dir is not None:
            print('DataLoader loading multilabel file: ', self.input_multilabel_dir)
            self.multilabel_loader = MultilabelLoader(self.input_multilabel_dir, in_memory=self.data_in_memory)

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_captions_and_phrase(self, ix, seq_per_img):
        # to keep the consistency with seq
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            phrase_num = np.zeros([seq_per_img], dtype='int')
            phrase_length = np.zeros([seq_per_img, self.seq_length], dtype='int')
            phrase_syn = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
                phrase_num[q] = self.phrase_num[ixl]
                phrase_length[q] = self.phrase_length[ixl]
                phrase_syn[q] = self.phrase_syn[ixl]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]
            phrase_num = self.phrase_num[ixl: ixl + seq_per_img]
            phrase_length = self.phrase_length[ixl: ixl + seq_per_img]
            phrase_syn = self.phrase_syn[ixl: ixl + seq_per_img]
        
        return seq, phrase_num, phrase_length, phrase_syn

    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []
        phrase_num_batch = []
        phrase_length_batch = []
        phrase_syn_batch = []

        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, \
                tmp_phrase_num, tmp_phrase_length, tmp_phrase_syn, \
                ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True
            
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            if self.pp_mode == 'phrase':
                phrase_num_batch.append(tmp_phrase_num)
                phrase_length_batch.append(tmp_phrase_length)
                phrase_syn_batch.append(tmp_phrase_syn)
            elif self.pp_mode == 'word':
                # for test word mode instead of phrase mode
                word_phrase_num = np.zeros([seq_per_img], dtype = 'int')
                word_phrase_length = np.zeros([seq_per_img, self.seq_length], dtype = 'int')

                for i in range(0, seq_per_img):
                    word_phrase_num[i] = sum(tmp_phrase_length[i, 0:tmp_phrase_num[i]] )
                    word_phrase_length[i, 0:word_phrase_num[i]] = 1
                
                phrase_num_batch.append(word_phrase_num)
                phrase_length_batch.append(word_phrase_length)
            elif self.pp_mode == 'phrase_2':
                phrase_num_2 = np.zeros([seq_per_img], dtype = 'int')
                phrase_length_2 = np.zeros([seq_per_img, self.seq_length], dtype = 'int')

                for i in range(0, seq_per_img):
                    cnt = 0
                    it = 0
                    while(it < tmp_phrase_num[i]):
                        if tmp_phrase_length[i, it] > 1:
                            phrase_length_2[i, cnt] = tmp_phrase_length[i, it]
                            cnt += 1
                            it += 1
                        else:
                            if it + 1 < tmp_phrase_num[i]:
                                phrase_length_2[i, cnt] = tmp_phrase_length[i, it] + tmp_phrase_length[i, it+1]
                                cnt += 1
                                it += 2
                            elif cnt > 0:
                                phrase_length_2[i, cnt - 1] += tmp_phrase_length[i, it]
                                it += 1
                    phrase_num_2[i] = cnt
                
                phrase_num_batch.append(phrase_num_2)
                phrase_length_batch.append(phrase_length_2)
            
            # this part adds bos and eos for seq_label
            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
                tmp_label[:, 0] = self.bos_idx
                tmp_label[:, self.seq_length + 1] = self.eos_idx
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        fc_batch, att_batch, label_batch, phrase_num_batch, phrase_length_batch, phrase_syn_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, label_batch, phrase_num_batch, phrase_length_batch, phrase_syn_batch, gts, infos), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(fc_batch)

        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        
        # generate mask
        if self.train_mode == 'PB_pad' or self.train_mode == 'NAIC' or self.train_mode == 'SAIC' or self.train_mode == 'UIC' or self.train_mode == 'UIC_ds' or self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
            phrase_num = np.stack(phrase_num_batch).astype('int')
            phrase_num = phrase_num.reshape(-1)
            phrase_length = np.vstack(phrase_length_batch).astype('int')
            phrase_syn = np.vstack(phrase_syn_batch).astype('int')
            
            data['phrase_num'] = phrase_num + 1
            max_phrase_num = max(phrase_num) + 2   # bos and eos
            B = data['labels'].shape[0]
            extend_phrase_syn_seq = np.zeros([B, self.seq_length + 2], dtype='long')
            extend_phrase_syn_seq[:, 0] = self.len_idx
            extend_phrase_seq = np.zeros([B, self.seq_length], dtype='long')
            extend_phrase_seq_mask = np.zeros([B, self.seq_length, self.seq_length], dtype='bool')

            if self.train_mode == 'NAIC' or self.train_mode == 'SAIC' or self.train_mode == 'UIC' or self.train_mode == 'UIC_ds' or self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
                data['phrase_length'] = np.zeros([B, self.seq_length + 2], dtype='int')
                data['phrase_length'][:, 0] = 1
                data['phrase_syn'] = np.zeros([B, self.seq_length + 2], dtype='int')
                data['phrase_syn'][:, 0] = self.bos_idx
            else:
                data['phrase_length'] = np.zeros([B, max_phrase_num], dtype='int')
                data['phrase_length'][:, 0] = 1
                data['phrase_syn'] = np.zeros([B, max_phrase_num], dtype='int')
                data['phrase_syn'][:, 0] = self.bos_idx
            
            for ix in range(0, B):
                data['phrase_length'][ix, 1:phrase_num[ix]+1] = phrase_length[ix, 0:phrase_num[ix]]
                data['phrase_syn'][ix, 1:phrase_num[ix]+1] = phrase_syn[ix, 0:phrase_num[ix]]
                data['phrase_syn'][ix, phrase_num[ix]+1] = self.eos_idx
                syn_last = 1
                for j in range(phrase_num[ix]):
                    extend_phrase_syn_seq[ix, syn_last:syn_last+phrase_length[ix, j]] = phrase_syn[ix, j]
                    syn_last += phrase_length[ix, j]
                
                seq_last = 0
                phrase_last = 0
                for j in range(1, data['phrase_num'][ix]):
                    if data['phrase_length'][ix, j] <= data['phrase_length'][ix, j-1]:
                        pre_pad = data['phrase_length'][ix, j-1] - data['phrase_length'][ix, j] # this part should not be copied
                        extend_phrase_seq[ix, phrase_last:phrase_last+data['phrase_length'][ix, j]] = data['labels'][ix, seq_last+pre_pad:seq_last+pre_pad+data['phrase_length'][ix, j]]
                    else:
                        pre_less = data['phrase_length'][ix, j-1] - (data['phrase_length'][ix, j] % data['phrase_length'][ix, j-1])
                        copy_times = data['phrase_length'][ix, j] // data['phrase_length'][ix, j-1]
                        copied = 0
                        for k in range(data['phrase_length'][ix, j-1]):
                            if k < pre_less:
                                extend_phrase_seq[ix, phrase_last+copied:phrase_last+copied+copy_times] = data['labels'][ix, seq_last+k]
                                copied += copy_times
                            else:
                                extend_phrase_seq[ix, phrase_last+copied:phrase_last+copied+copy_times+1] = data['labels'][ix, seq_last+k]
                                copied += (copy_times+1)
                    extend_phrase_seq_mask[ix, phrase_last:, :phrase_last+data['phrase_length'][ix, j]] = True
                    seq_last += data['phrase_length'][ix, j-1]
                    phrase_last += data['phrase_length'][ix, j]
                    
            max_phrase_length = np.zeros([max_phrase_num], dtype='int')
            for ix in range(0, max_phrase_num):
                max_phrase_length[ix] = max([data['phrase_length'][i, ix] for i in range(0, B)])

            phrase_start = np.zeros([max_phrase_num], dtype='int')
            for ix in range(1, max_phrase_num):
                phrase_start[ix] = phrase_start[ix-1] + max_phrase_length[ix-1]
            
            t = torch.Tensor([1])
            phrase = t.new_full([B, sum(max_phrase_length)], self.pad_idx, dtype=torch.long)
            # print(data['phrase_length'])
            # print(phrase.shape)
            # constrcut phrase: B × len
            for i in range(0, B):
                last = 0
                for j in range(0, data['phrase_num'][i]):
                    phrase[i, phrase_start[j]:phrase_start[j]+data['phrase_length'][i, j] ] = torch.from_numpy(data['labels'][i, last:last+data['phrase_length'][i, j] ])
                    # phrase[i, phrase_start[j]+data['phrase_length'][i, j] ] = self.sep_idx
                    last += data['phrase_length'][i, j]
                # phrase[i, phrase_start[data['phrase_num'][i]] ] = self.sep_idx
        
            data['masks'] = phrase != self.pad_idx
            data['phrase_num'] = data['phrase_num'].reshape(len(batch), seq_per_img)
            data['phrase_length'] = data['phrase_length'].reshape(len(batch), seq_per_img, -1)
            data['phrase_syn'] = data['phrase_syn'].reshape(len(batch), seq_per_img, -1)
            data['phrase'] = phrase.reshape(len(batch), seq_per_img, -1)
            data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
            data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)
            data['extend_phrase_syn_seq'] = extend_phrase_syn_seq.reshape(len(batch), seq_per_img, -1)
            data['extend_phrase_seq'] = extend_phrase_seq.reshape(len(batch), seq_per_img, -1)
            data['extend_phrase_seq_mask'] = extend_phrase_seq_mask.reshape(len(batch), seq_per_img, -1)
        else:
            nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
            mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            data['masks'] = mask_batch
            data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
            data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)
            data['phrase'] = None
            data['phrase_num'] = None
            data['phrase_length'] = None
            data['phrase_syn'] = None
            data['extend_phrase_syn_seq'] = None
            data['extend_phrase_seq'] = None
            data['extend_phrase_seq_mask'] = None

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]
        if self.input_multilabel_dir is not None:
            if self.use_att:
                att_feat = self.multilabel_loader.get(cocoid=str(self.info['images'][ix]['id']), label_key='feat')
                # Reshape to K x C
                att_feat = att_feat.reshape(-1, att_feat.shape[-1])
                if self.norm_att_feat:
                    att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
                if self.use_box:
                    """wait implement"""
                    pass
            else:
                att_feat = np.zeros((0,0), dtype='float32')
        else:
            if self.use_att:
                att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
                # Reshape to K x C
                att_feat = att_feat.reshape(-1, att_feat.shape[-1])
                if self.norm_att_feat:
                    att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
                if self.use_box:
                    box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                    # devided by image width and height
                    x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                    h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                    box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                    if self.norm_box_feat:
                        box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                    att_feat = np.hstack([att_feat, box_feat])
                    # sort the features by the size of boxes
                    att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
            else:
                att_feat = np.zeros((0,0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        
        # if hasattr(self, 'h5_label_file'):
        #     seq = self.get_captions(ix, self.seq_per_img)
        # else:
        #     seq = None
        
        if self.train_mode == 'PB_pad' or self.train_mode == 'NAIC' or self.train_mode == 'SAIC' or self.train_mode == 'UIC' or self.train_mode == 'UIC_ds' or self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
            if hasattr(self, 'h5_label_file'):
                seq, phrase_num, phrase_length, phrase_syn = self.get_captions_and_phrase(ix, self.seq_per_img)
            else:
                seq, phrase_num, phrase_length, phrase_syn = None, None, None, None
        else:
            phrase_num, phrase_length, phrase_syn = None, None, None
            if hasattr(self, 'h5_label_file'):
                seq = self.get_captions(ix, self.seq_per_img)
            else:
                seq = None
            
        return (fc_feat,
                att_feat, seq,
                phrase_num, phrase_length, phrase_syn,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])

class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)
        self.KD_file = getattr(opt, 'KD_file', '') # for generate distilled dataset with auto-regressive model

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                if self.KD_file == '':
                    sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
                else:
                    sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=4, # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
