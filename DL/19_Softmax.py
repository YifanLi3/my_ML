import torch

scores = torch.tensor([[0.2, 0.35, 0.1, 0.45], [0.1, 0.13, 0.05, 2.79]])

proba = torch.softmax(scores, dim=1)
print(proba)

pred = torch.argmax(proba, dim=1)
print(pred)