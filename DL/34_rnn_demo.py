from enum import unique
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import time
import os

def built_vocab():
    unique_words, all_words = [], []

    data_file = './data/nursery_rhymes.txt'
    if not os.path.exists(data_file):
        raise FileNotFoundError('file not exist')
    
    for line in open(data_file, 'r', encoding='utf-8'):
        line = line.lower().strip()
        if line:
            words = line.split()
            all_words.append(words)
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
    
    unique_words.append('\n')
    unique_words.append(' ')
    
    words_count = len(unique_words)
    word_to_index = {word:i for i, word in enumerate(unique_words)}
    corpus_idx = []
    for words in all_words:
        tmp = [word_to_index[word] for word in words]
        tmp.append(word_to_index['\n'])
        tmp.append(word_to_index[' '])
        corpus_idx.extend(tmp)

    return unique_words, word_to_index, words_count, corpus_idx

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, seq_length) -> None:
        self.corpus_idx = corpus_idx
        self.seq_length = seq_length
        self.word_count = len(corpus_idx)

        self.number = self.word_count//self.seq_length

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.seq_length - 1)
        end = start + self.seq_length

        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start+1:end+1]


if __name__ == '__main__':
    unique_words, word_to_index, words_count, corpus_idx = built_vocab()