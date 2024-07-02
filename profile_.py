from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob
import time

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
# from pysot.tracker.tracker_builder import build_tracker
from thop import profile
from thop.utils import clever_format


# load config
cfg.merge_from_file('experiments/config.yaml')
# cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
cfg.CUDA = False
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# create model
model = ModelBuilder()
model.eval().to(device)

xs = 287
zs = 127

x = torch.randn(1, 3, xs, xs)
z = torch.randn(1, 3, zs, zs)

x = x.to(device)
z = z.to(device)
inputs=(x, z)

# macs1, params1 = profile(model, inputs=inputs,
#                              custom_ops=None, verbose=False)
# macs, params = clever_format([macs1, params1], "%.3f")
# print('overall macs is ', macs)
# print('overall params is ', params)

zf = model.template(z)
T_w = 50  # warmup
T_t = 500  # test
with torch.no_grad():
    for i in range(T_w):
        oup = model(x,z)
    t_s = time.time()
    for i in range(T_t):
        oup = model(x,z)
    torch.cuda.synchronize()
    t_e = time.time()
    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))

print("Done")


print("Done")