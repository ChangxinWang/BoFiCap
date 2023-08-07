from cmath import phase
import json
from tabnanny import verbose
import h5py
import re, collections
import argparse
from random import seed

def load_phrase(params):
    input_phrase = params['input_phrase']
    limit = params['limit']
    verbose = params['verbose']
    table = json.load(open(input_phrase, 'r'))
    phrase = []
    for key in table:
        if table[key] >= limit:
            tmp = key.split('\'')
            phrase.append(' '.join( [tmp[1], tmp[3]] )  )
        else:
            break
    if verbose:
        print(phrase)
    return phrase


def merge_phrase(params, phrase):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    verbose = params['verbose']
    input_label_h5 = params['input_h5']

    if input_label_h5 != '':
        h5_label_file = h5py.File(input_label_h5, 'r', driver='core')
        label_start_ix = h5_label_file['label_start_ix'][:]
        label_end_ix = h5_label_file['label_end_ix'][:]
        phrase_num = h5_label_file['phrase_num'][:]
        phrase_length = h5_label_file['phrase_length'][:]
        L = h5_label_file['labels'][:]
        label_length = h5_label_file['label_length'][:]
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

                for k in phrase:
                    bigram = re.escape(k)
                    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
                    alpha_sent = p.sub(''.join(k.split()), alpha_sent)

                tmp = alpha_sent.split()
                phrase_num[cur_idx] = len(tmp)
                for k in range( len(phrase_length[cur_idx]) ):
                    phrase_length[cur_idx][k] = 0
                for k in range(phrase_num[cur_idx]):
                    phrase_length[cur_idx][k] = (len(tmp[k].split('@')) - 1) / 2
                if verbose and ix < 10:
                    print(phrase_num[cur_idx])
                    print(phrase_length[cur_idx])
                    print(tmp)
                    print('--------------------------')
            if verbose and ix % 10000 == 0:
                print(ix)
        if verbose:
            phrase_sum = sum(phrase_num)
            word_sum = sum(sum(phrase_length))
            print("compressed ratio:{}/{}={:.3f}".format(phrase_sum, word_sum, phrase_sum/word_sum))
        f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
        f_lb.create_dataset("phrase_num", dtype='uint32', data=phrase_num)
        f_lb.create_dataset("phrase_length", dtype='uint32', data=phrase_length)
        f_lb.create_dataset("labels", dtype='uint32', data=L)
        f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
        f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
        f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
        f_lb.close()


def main(params):
    seed(123) # make reproducible

    phrase = load_phrase(params)
    merge_phrase(params, phrase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_h5', default='', help='merge noun phrase')
    parser.add_argument('--output_h5', default='', help='output h5 file')
    parser.add_argument('--input_phrase', default='data/phrase_pair.json', help='phrase encoding record')
    
    parser.add_argument('--limit', default=10000, type=int, help='only phrases that occur more than this number of times will be organized to phrase')
    parser.add_argument('--verbose', default=True, type=bool, help='whether or not to print prompt info')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)