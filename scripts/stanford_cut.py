from tkinter.messagebox import NO
import stanza
import re


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


def gather_phrase(node, phrase_lst):
    # return true if current node is or has VP/NP phrase
    has_vpnp = False
    if node.children is not None:
        for child in node.children:
            if gather_phrase(child, phrase_lst):
                has_vpnp = True
    
    if has_vpnp:
        # this node include smaller VP/NP phrase
        return True
    elif node.label == 'VP' or node.label == 'NP':
        # this node is smallest VP/NP phrase
        phrase_lst.append(transform(node))
        return True
    else:
        return False


def gather_phrase_level(node, phrase_lst, cur_dep, dest_dep):
    # return True if current node is gathered or has children gathered
    if node.label == 'VP' or node.label == 'NP':
        cur_dep += 1
        # condition 1: if this node is VP/NP and cur_dep == dest_dep , gather it
        if cur_dep == dest_dep:
            phrase_lst.append(transform(node))
            return True

    # try children
    has_gathered = False
    if node.children is not None:
        for child in node.children:
            if gather_phrase_level(child, phrase_lst, cur_dep, dest_dep):
                has_gathered = True
    
    # has child gathered, so this node shouldn't be gathered
    if has_gathered:
        return True
    elif node.label == 'VP' or node.label == 'NP':
        # this node is the deepest VP/NP phrase
        phrase_lst.append(transform(node))
        return True
    else:
        return False


if __name__ == '__main__':
    # stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    # 'a quick brown fox jumped over the lazy dog'
    annotation = "a man and a woman walking on the dirty road"
    doc = nlp(annotation)
    for sentence in doc.sentences:
        tree = sentence.constituency
        print(tree)
        phrase_lst = []
        gather_phrase_level(tree, phrase_lst, 0, 3)
        print(phrase_lst)

        tmp_phrase_num = 0
        tmp_phrase_length = []
        phrase_start = []
        raw_token_lst = annotation.split()
        start = 0
        end = len(raw_token_lst)

        # find phrase_start in sentence
        for phrase in phrase_lst:
            phrase = phrase.split()
            for i in range(start, end):
                # if some phrase can't match...
                if len(phrase) > end-i:
                    print("error: ", annotation)
                    print(phrase_lst)
                    print(tree)
                    exit()
                
                match = True
                for j in range(len(phrase)):
                    if raw_token_lst[i + j] != phrase[j]:
                        match = False
                        break
                
                if match:
                    tmp_phrase_num += 1
                    phrase_start.append(i)
                    tmp_phrase_length.append(len(phrase))
                    start = i + len(phrase)
                    break
        
        phrase_num = 0
        phrase_length = []
        end = 0
        for i in range(tmp_phrase_num):
            start = phrase_start[i]
            # deal with those tokens between phrase
            if start > end:
                phrase_length.append(start - end)
                phrase_num += 1
            phrase_length.append(tmp_phrase_length[i])
            phrase_num += 1
            end = start + tmp_phrase_length[i]
        if len(raw_token_lst) > end:
            phrase_length.append(len(raw_token_lst) - end)
            phrase_num += 1
        
        print(phrase_num)
        print(phrase_length)
                
