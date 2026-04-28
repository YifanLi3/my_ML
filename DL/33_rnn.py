import torch
import torch.nn as nn

rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
print(rnn)

x = torch.randn(size=(5, 32, 128))

h0 = torch.randn(size=(1, 32, 256))

output, h1 = rnn(x, h0)

print(output.shape)
print(h1.shape)