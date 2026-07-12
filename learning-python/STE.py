#topk for relevancy
import math
import torch
import torch.nn as nn


x = torch.tensor([-1.0, 5.8, 3.0, -3.1, -3.2])

values, _ = torch.topk(x, k=1)
treshold = values[-1]

print(values)
print(treshold)

binary_mask = (x >= treshold).float()

print(binary_mask)
