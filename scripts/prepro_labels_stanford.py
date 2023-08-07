"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences
/phrase is (M, max_length, max_length) uint32 array of cutted and encoded labels with spacy, zero padded
/phrase_num stores the num of phrase for each of the M captions
/phrase_length is (M, max_length) uint32 array which stores the lengths of each phrase for each of the M captions

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np

import torch
import torchvision.models as models
import skimage.io
from PIL import Image
# prepare for phrase based dataset
import stanza
import re

VP_IDX=4
NP_IDX=5
CP_IDX=6

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions(imgs, params, wtoi):
    """ 
    encode all captions into one large array, which will be 4-indexed. {0:pad 1:bos 2:eos 3:sep}
    also produces label_start_ix and label_end_ix which store 1-indexed 
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be 1-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i,img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j,s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k,w in enumerate(s):
                if k < max_length:
                    Li[j,k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        
        counter += n
    
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def transform(node):
    # transform parse node to phrase
    tmp = re.split('[()\ ]', str(node))
    # print("tmp: ", tmp)
    word_lst = []
    for x in tmp:
        if x.strip == '' or x.isupper() or x.strip() == '.':
            continue
        word_lst.append(x)
    return " ".join(word_lst)


# def gather_phrase(node, phrase_lst):
#     # return true if current node is or has VP/NP phrase
#     has_vpnp = False
#     if node.children is not None:
#         for child in node.children:
#             if gather_phrase(child, phrase_lst):
#                 has_vpnp = True
    
#     if has_vpnp:
#         # this node include smaller VP/NP phrase
#         return True
#     elif node.label == 'VP' or node.label == 'NP':
#         # this node is smallest VP/NP phrase
#         phrase_lst.append(transform(node))
#         return True
#     else:
#         return False


def gather_phrase_level(node, phrase_lst, label_lst, cur_dep, dest_dep):
    # return True if current node is gathered or has children gathered
    if node.label == 'VP' or node.label == 'NP':
        cur_dep += 1
        # condition 1: if this node is VP/NP and cur_dep == dest_dep , gather it
        if cur_dep == dest_dep:
            phrase_lst.append(transform(node))
            label_lst.append(VP_IDX if node.label == 'VP' else NP_IDX)
            return True

    # try children
    has_gathered = False
    if node.children is not None:
        for child in node.children:
            if gather_phrase_level(child, phrase_lst, label_lst, cur_dep, dest_dep):
                has_gathered = True
    
    # has child gathered, so this node shouldn't be gathered
    if has_gathered:
        return True
    elif node.label == 'VP' or node.label == 'NP':
        # this node is the deepest VP/NP phrase
        phrase_lst.append(transform(node))
        label_lst.append(VP_IDX if node.label == 'VP' else NP_IDX)
        return True
    else:
        return False


def cut_caption(imgs, params, wtoi, itow):

    max_length = params['max_length']
    verbose = params['verbose']
    depth = params['depth']
    cut_json = params['cut_json']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions

    phrase_num = np.zeros(M, dtype='uint32')
    phrase_length_arrays = []
    phrase_label_arrays = []
    caption_cnt = 0
    error_cnt = 0
    cut_info = {}
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    for img in imgs:
        if cut_json != '':
            key = img['cocoid']
            value = []
        for sent in img['sentences']:
            # mention that we should cut overlong part
            # raw = string(sent['raw']).strip(string.punctuation)
            # if caption_cnt < 4000:
            #     caption_cnt += 1
            #     continue
            raw_tokens = sent['tokens']
            raw_tokens = raw_tokens[0:min(max_length, len(raw_tokens))]
            raw = ' '.join(raw_tokens)
            doc = nlp(raw)
            
            phrase_length_arrays_item = np.zeros(max_length, dtype='uint32')
            phrase_label_arrays_item = np.zeros(max_length, dtype='uint32')
            phrase_cnt = 0

            tmp_phrase_num = 0
            tmp_phrase_length = []
            phrase_start = []
            
            try:
                for sentence in doc.sentences:
                    tree = sentence.constituency
                    # print(tree)
                    phrase_lst = []
                    label_lst = []
                    gather_phrase_level(tree, phrase_lst, label_lst, 0, depth)
                    # print(phrase_lst)

                    end = len(raw_tokens)
                    start = 0
                    for id, phrase in enumerate(phrase_lst):
                        phrase = phrase.split()
                        for i in range(start, end):
                            # if some phrase can't match...
                            if len(phrase) > end-i:
                                raise Exception("phrase can't be found")
                            
                            match = True
                            for j in range(len(phrase)):
                                if raw_tokens[i + j] != phrase[j]:
                                    match = False
                                    break
                            
                            if match:
                                tmp_phrase_num += 1
                                phrase_start.append(i)
                                tmp_phrase_length.append(len(phrase))
                                start = i + len(phrase)
                                break

                    end = 0
                    for i in range(tmp_phrase_num):
                        start = phrase_start[i]
                        # deal with those tokens between phrase
                        if start > end:
                            phrase_length_arrays_item[phrase_cnt] = start - end
                            phrase_label_arrays_item[phrase_cnt] = CP_IDX
                            phrase_cnt += 1
                        phrase_length_arrays_item[phrase_cnt] = tmp_phrase_length[i]
                        phrase_label_arrays_item[phrase_cnt] = label_lst[i]
                        phrase_cnt += 1
                        end = start + tmp_phrase_length[i]
                    if len(raw_tokens) > end:
                        phrase_length_arrays_item[phrase_cnt] = len(raw_tokens) - end
                        phrase_label_arrays_item[phrase_cnt] = CP_IDX
                        phrase_cnt += 1
                    
            except Exception as e:
                print(e)
                print("image idx: ", caption_cnt)
                print("error: ", raw)
                print(phrase_lst)
                print(tree)
                error_cnt += 1
                phrase_cnt = len(raw_tokens)
                for i in range(phrase_cnt):
                    phrase_length_arrays_item[i] = 1
                    phrase_label_arrays_item[i] = CP_IDX

            if cut_json != '':
                value_item = {}
                value_item['caption'] = raw
                value_item['phrase_num'] = phrase_cnt
                value_item['phrase_length'] = phrase_length_arrays_item.tolist()
                value_item['phrase_label'] = phrase_label_arrays_item.tolist()
                value.append(value_item)

            phrase_num[caption_cnt] = phrase_cnt
            caption_cnt += 1
            # phrase_arrays.append(phrase_arrays_item)
            phrase_length_arrays.append(phrase_length_arrays_item)
            phrase_label_arrays.append(phrase_label_arrays_item)

            if verbose:
                if caption_cnt <= 50:
                    print(raw)
                    print(phrase_cnt)
                    print(phrase_length_arrays_item)
                    print(phrase_label_arrays_item)
                
                if caption_cnt % 1000 == 0:
                    print(caption_cnt)
                    if cut_json != '':
                        json.dump(cut_info, open(cut_json, 'w'))

        if cut_json != '':
            cut_info[key] = value

    # phrase = np.stack(phrase_arrays)
    phrase_length = np.stack(phrase_length_arrays)
    phrase_label = np.stack(phrase_label_arrays)
    if verbose:
        print("stanza error num: ", error_cnt)
    if cut_json != '':
        json.dump(cut_info, open(cut_json, 'w'))
    return phrase_num, phrase_length, phrase_label


def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    verbose = params['verbose']

    seed(123) # make reproducible
    
    # create the vocab
    input_vocab = params['input_vocab']
    if input_vocab != '':
        print('replace vocab with {}'.format(input_vocab))
        input_vocab = json.load(open(input_vocab, 'r'))
        if 'ix_to_word' in input_vocab:
            itow = input_vocab['ix_to_word']
            wtoi = {w:i for i,w in itow.items()}
        for img in imgs:
            img['final_captions'] = []
            for sent in img['sentences']:
                txt = sent['tokens']
                caption = [w if w in wtoi else 'UNK' for w in txt]
                img['final_captions'].append(caption)
        print(imgs[0]['final_captions'])
    else:
        vocab = build_vocab(imgs, params)
        itow = {i+4:w for i,w in enumerate(vocab)} # a 4-indexed vocab translation table {0:pad 1:bos 2:eos 3:sep}
        wtoi = {w:i+4 for i,w in enumerate(vocab)} # inverse table
    
    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # cut captions with spacy and encode them in large arrays
    phrase_num, phrase_length, phrase_label = cut_caption(imgs, params, wtoi, itow)

    if verbose:
        L_p = sum(phrase_num)
        L_r = sum(label_length)
        print("compressed ratio:{}/{} = {:.3f}".format(L_p, L_r, L_p/L_r))

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    #f_lb.create_dataset("phrase", dtype='uint32', data=phrase)
    f_lb.create_dataset("phrase_num", dtype='uint32', data=phrase_num)
    f_lb.create_dataset("phrase_length", dtype='uint32', data=phrase_length)
    f_lb.create_dataset("phrase_label", dtype='uint32', data=phrase_label)
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(imgs):
        
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)
    
    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_vocab', default='', help='replace vocab with vocab in input vocab file')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--cut_json', default='', help='stanza cut json')
    # options
    parser.add_argument('--depth', default=1, type=int, help='gather phrase which corresponding to this layer')
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--verbose', default=True, type=bool, help='whether or not to print prompt info')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
