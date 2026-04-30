import torch
import torch.nn as nn

def demo01():
    text = 'Some birds are not meant to be caged their feathers are just too bright when fly away'

    words = text.split()
    print(words)

    embed = nn.Embedding(len(words), 4)
    #for i, word in enumerate(words):
    #    word_vector = embed(torch.tensor(i))
    #    print(f'{word}: {word_vector.data}')

    indices = torch.arange(len(words))
    word_vectors = embed(indices)
    for word, word_vector in zip(words, word_vectors):
        print(f'{word}: {word_vector.data}')

if __name__ == '__main__':
    demo01()