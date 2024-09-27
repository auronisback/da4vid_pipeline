import os
from pathlib import Path

import torch.cuda

RESOURCES_ROOT = f'{str(os.path.dirname(os.path.abspath(__file__)))}/res'
TEST_GPU = 'cuda:0' if torch.cuda.is_available() else 'cpu'
