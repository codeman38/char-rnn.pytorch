# https://github.com/spro/char-rnn.pytorch

import string
import random
import time
import math
import torch
import re

# Reading and extracting vocab from data

def read_file(filename, by_word=False):
    print('by_word={}'.format(by_word))
    file = open(filename).read()
    if by_word:
        par_split_re = re.compile(r'\n\n+')
        word_split_re = re.compile(r'[ \n\t]')
        words = []
        for para in par_split_re.split(file):
            words.extend(word_split_re.split(para))
            words.append('\n\n')
        file = words
        vocab = sorted(set(file))
    else:
        vocab = ''.join(sorted(set(file)))
    return file, len(file), vocab

# Turning a string into a tensor

def char_tensor(string, vocab):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = vocab.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

