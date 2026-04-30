import copy
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules import module

from transformer_35_input import *

def sub_mask(size):
    b = torch.ones(1, size, size, dtype=torch.long)
    upper_tri = torch.triu(b, diagonal=1)

    return 1 - upper_tri

def attention(query, key, value, mask=None, dropout = None):
    dk = query.size(-1)
    scores = torch.matmul(query, torch.transponse(key, -1, -2)) / math.sqrt(dk)

    if math is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    atten_weight = F.softmax(scores, dim = -1)

    if dropout is not None:
        atten_weight = dropout(atten_weight)

    return torch.matmul(atten_weight, value), atten_weight

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim

        assert embed_dim % head == 0
        self.d_k = embed_dim // head

        self.head = head

        self.linears = clones(module = nn.Linear(embed_dim, embed_dim), N=4)

        self.atten_weight = None
        self.dropout_p = nn.Dropout(p=dropout_p)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)

        self.batch_size = query.size(0)
        query, key, value = [model(x).view(self.batch_size, -1, self.head, self.d_k).traspose(1, 2) for model, x in zip(self.linears, (query, key, value))]

        atten_result, atten_weight = attention(query, key, value, mask=mask, dropout=self.dropout_p)

        result = atten_result.transpose(1, 2).contiguous().view(self.batch_size, -1, self.head*self.d_k)

        return self.linears[-1](result)


if __name__ == '__main__':
    print("1_________________________")
    mask = sub_mask(5)
    print(mask)
