#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse
import warnings

from helpers import *
from model import *

def generate(decoder, vocab, prime_str='A', predict_len=100, temperature=0.8,
             truncate=0, cuda=False, delim=''):
    return delim.join(generate_yield(decoder, vocab, prime_str, predict_len,
                                     temperature, truncate, cuda))

def generate_yield(decoder, vocab, prime_str='A', predict_len=100,
                    temperature=0.8, truncate=0, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str, vocab).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    for ch in prime_str:
        yield ch

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        with torch.no_grad():
            _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        with torch.no_grad():
            output, hidden = decoder(inp, hidden)
        
        # Normalize outputs to avoid overflow in exponentiation
        output_norm = output.data.view(-1) - output.data.view(-1).max()
        # Sample from the network as a multinomial distribution
        output_dist = output_norm.div(temperature).exp()
        # If truncated, truncate the model to the top K
        if truncate > 0:
            probs, indices = torch.topk(output_dist, truncate)
            output_dist[:] = 0
            output_dist[indices] = probs
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = vocab[top_i]
        yield predicted_char
        inp = Variable(char_tensor([predicted_char], vocab).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--word', action='store_true')
    argparser.add_argument('--truncate', type=int, default=0)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    if args.word:
        del args.word
        args.prime_str = args.prime_str.split(' ')
        delim = ' '
    else:
        delim = ''

    with warnings.catch_warnings():
        warnings.simplefilter('ignore',
                              torch.serialization.SourceChangeWarning)
        try:
            vocab, decoder = torch.load(args.filename)
        except TypeError:
            vocab = string.printable
            decoder = torch.load(args.filename)
    del args.filename
    for ch in generate_yield(decoder, vocab, **vars(args)):
        print(ch, end=delim)
    print()

