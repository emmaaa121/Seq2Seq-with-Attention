#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
utils.py: Utility Functions
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []
    ### YOUR CODE HERE (~6 Lines)
    max_len = len(max(sents, key=len))
    sents_padded = [sent + [pad_token] * (max_len - len(sent) for sent in sents)]
    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = [] #store the tokenized sentences
    sp = spm.SentencePieceProcessor() #tokenization of the sentences
    sp.load('{}.model'.format(source))
    # It loads a pre-trained model specific to either the source or target language, indicated by the source parameter.

    with open(file_path, 'r', encoding='utf8') as f: #read sentences line by line
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data
    

#Differences from the read_corpus function using SentencePiece:

#Tokenization Level: This function tokenizes at the word level rather than the subword level. Word-level tokenization is straightforward but might not handle rare words or morphologically rich languages as effectively as subword tokenization.
#Tool Used for Tokenization: It uses NLTK's word_tokenize instead of SentencePiece, making it more suitable for languages where word boundaries are clearly defined and where handling of rare words or agglutinative language features is not as critical.
#Encoding Consideration: There's no explicit encoding specified for reading the file, which might require adjustments when working with non-ASCII text.
#Dependency on NLTK: This function requires the NLTK library, specifically its word_tokenize method, which might necessitate additional setup or installation compared to the SentencePiece-based function.


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    # This version uses the nltk.word_tokenize method from the Natural Language, Toolkit (NLTK) library, which is more focused on word-level tokenization compared to subword tokenization in the previous read_corpus function.
    data = []
    for line in open(file_path): #Python defaults to the system's default encoding, which might not be UTF-8, potentially leading to issues with non-ASCII characters.
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data
    

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
        #ensure that models do not learn anything from the order of samples.

    for i in range(batch_num):
        #select the indices for the current batch and extract samples
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        #Each batch of examples is sorted in descending order
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
        

        
    

#examples = [
    #("This is a longer sentence.", "This is short."),
    #("Short sentence.", "This is a bit longer."),
    #("This is the longest sentence in this list.", "Short again.")
            #]
#After sorting(based on the length of the source sentences (the first element of each tuple):
    #[
    #("This is the longest sentence in this list.", "Short again."),
    #("This is a longer sentence.", "This is short."),
    #("Short sentence.", "This is a bit longer.")
#]


