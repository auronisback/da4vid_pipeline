import os

import torch.cuda

DOTENV_FILE = f'{str(os.path.dirname(os.path.abspath(__file__)))}/../.env'
RESOURCES_ROOT = f'{str(os.path.dirname(os.path.abspath(__file__)))}/res'
TEST_GPU = 'cuda:0' if torch.cuda.is_available() else 'cpu'
