import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embed_x = self.embed(x)

        return embed_x * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len = 60):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_p)
        tem_vec = torch.arange(0, max_len).unsqueeze(1)

        div_vec = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        position = tem_vec * div_vec

        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        position_x = x + self.pe[:, :x.size(1)]
        return self.dropout(position_x)

if __name__ == '__main__':
    my_embed = Embeddings(1000, 512)
    x = torch.tensor([[1, 4, 7, 10], [2, 6, 40, 5]])
    embed_result = my_embed(x)
    print(embed_result.shape)

    position_encode = PositionEncoding(512, 0.1)
    result= position_encode(embed_result)
    print(result.shape)