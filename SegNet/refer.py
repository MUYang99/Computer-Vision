import torch.nn as nn
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

m = nn.LogSoftmax(dim=2)
d = torch.tensor([[[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]],
                  [[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]]])
output = m(d)