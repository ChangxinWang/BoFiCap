import json

import h5py
import re, collections
import argparse
from random import seed


def build_sentence_table(imgs, params):
    input_label_h5 = params['input_h5']
    if input_label_h5 != '':
        h5_label_file = h5py.File(input_label_h5, 'r', driver='core')
        label_start_ix = h5_label_file['label_start_ix'][:]
        label_end_ix = h5_label_file['label_end_ix'][:]
        phrase_num = h5_label_file['phrase_num'][:]
        phrase_length = h5_label_file['phrase_length'][:]
    sentence_table={}
    if input_label_h5 == '':
        for img in imgs:
            for sent in img['sentences']:
                alpha_tokens = []
                for w in sent['tokens']:
                    alpha_tokens.append('@' + w + '@')
                alpha_sent = ' '.join(alpha_tokens)
                sentence_table[alpha_sent] = sentence_table.get(alpha_sent, 0) + 1
    else:
        for ix, img in enumerate(imgs):
            if img['split'] == 'test' or img['split'] == 'val':
                continue
            for j, sent in enumerate(img['sentences']):
                alpha_tokens = []
                cur_idx = label_start_ix[ix] - 1 + j
                cur_phrase_num = phrase_num[cur_idx]
                cur_phrase_length = phrase_length[cur_idx]
                bos = 0
                for k in range(cur_phrase_num):
                    tmp = []
                    for l in range(cur_phrase_length[k]):
                        tmp.append('@' + sent['tokens'][bos + l] + '@')
                    bos += cur_phrase_length[k]
                    alpha_tokens.append(''.join(tmp))
                alpha_sent = ' '.join(alpha_tokens)
                sentence_table[alpha_sent] = sentence_table.get(alpha_sent, 0) + 1
    return sentence_table


def build_phrase_table(sentence_table):
    phrase_table = collections.defaultdict(int)
    for sent, freq in sentence_table.items():
        alpha_tokens = sent.split()
        for i in range(0, len(alpha_tokens)-1):
            phrase_table[alpha_tokens[i], alpha_tokens[i+1]] += freq
    return phrase_table


def merge_phrase_table(pair, t_in):
    t_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for sent in t_in:
        sent_out = p.sub(''.join(pair), sent)
        t_out[sent_out] = t_in[sent]
    return t_out
    

def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    verbose = params['verbose']
    phrase_thr = params['phrase_count_threshold']
    record = {}
    num_merges = 1000
    seed(123) # make reproducible

    sentence_table = build_sentence_table(imgs, params)
    for i in range(0, num_merges):
        phrase_table = build_phrase_table(sentence_table)
        if not phrase_table:
            break
        best = max(phrase_table, key=phrase_table.get)
        if phrase_table[best] < phrase_thr:
            break
        print("{} : {}".format(best, phrase_table[best]))
        # json.dump(record, open('data/phrase_pair.json', 'w'))
        record[str(best)] = phrase_table[best]
        sentence_table = merge_phrase_table(best, sentence_table)
    json.dump(record, open(params['output_json'], 'w'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_h5', default='', help='merge noun phrase')
    parser.add_argument('--input_vocab', default='', help='replace vocab with vocab in input vocab file')
    parser.add_argument('--output_json', default='data/phrase_pair.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--phrase_count_threshold', default=5, type=int, help='only phrases that occur more than this number of times will be put in phrase table')
    parser.add_argument('--verbose', default=True, type=bool, help='whether or not to print prompt info')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)