import torch
import torch.nn as nn
import torch.nn
from torch.nn import functional as F
import numpy as np

data = np.random.random((8, 16, 25))
data = torch.from_numpy(data).float()
# data.permute((0, 1, 3, 2)).contiguous()
output = F.avg_pool1d(data, data.size()[-1])

pass
