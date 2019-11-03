import os
import random
import numpy as np
import torch


def fix_seed(seed=1119):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def device_cuda(cuda_idx):
    if torch.cuda.is_available():
        return 'cuda:{}'.format(cuda_idx)

    else:
        return "cpu"
