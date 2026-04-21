import torch
import os
import random

import numpy as np
import argparse

from data import MovieLensDataset
from model import SASRec

def set_seed(seed:int)-> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manula_seed_all(seed)


def sample_negative_items(pos_items:torch.tensor,item_num:int,device:torch.device)->torch.tensor:
    neg_items=torch.randint(1,item_num+1,size=pos_items.shape,device=device)
    neg_items[pos_items==0]=0
    return neg_items
    