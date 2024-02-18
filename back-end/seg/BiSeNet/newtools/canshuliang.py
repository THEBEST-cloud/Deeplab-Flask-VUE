import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from newtools.dataset_udd import DatasetTrain,DatasetVal
from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from tqdm import tqdm, trange
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time

from metrics import *
from torchvision  import datasets, transforms


if __name__ == "__main__":
    # NOTE! NOTE! change this to not overwrite all log data when you train the model:
    # network = DeepLabV3(model_id=1, project_dir="E:/master/master1/RSISS/deeplabv3/deeplabv3").cuda()
    # x = Variable(torch.randn(2,3,256,256)).cuda() 
    # print(x.shape)
    # y = network(x)
    # print(y.shape)
    model_id = "terrace"

    num_epochs = 101
    batch_size = 4
    learning_rate = 0.001

    def parse_args():
        parse = argparse.ArgumentParser()
        parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
        parse.add_argument('--port', dest='port', type=int, default=44554,)
        parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
        parse.add_argument('--finetune-from', type=str, default=None,)
        return parse.parse_args()

    args = parse_args()
    cfg = cfg_factory[args.model]
    network = model_factory[cfg.model_type](8)
    network.eval()
    from thop import profile
    input = torch.randn(2,3,512,1024)
    flops, params = profile(network, inputs=(input, ))
    print("flops",flops)
    print("params",params)
    network.cuda()
    imgs = input.cuda()
    start = time.time()
    outputs,*outputs_aux = network(imgs)
    end = time.time()
    print("latency: ", end-start)
    print("fps: ", 1/(end-start))