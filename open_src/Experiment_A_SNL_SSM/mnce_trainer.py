"""Model training and online NLL adaptation for the existing model."""

import argparse
import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from device import get_device
from trainer import build_model, gaussian_nll_loss

try:
    from torch.func import functional_call
except ImportError:  # PyTorch 1.x compatibility
    from torch.nn.utils.stateless import functional_call


