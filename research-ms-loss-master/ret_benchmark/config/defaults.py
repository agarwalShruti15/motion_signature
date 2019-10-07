# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN
from .model_path import MODEL_PATH

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "vgg16_bn"

_C.MODEL.PRETRAIN = 'none'
_C.MODEL.PRETRIANED_PATH = MODEL_PATH

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "linear_norm"
_C.MODEL.HEAD.DIM = 10

_C.MODEL.WEIGHT = ""

# Checkpoint save dir
_C.SAVE_DIR = 'output'

# Loss
_C.LOSSES = CN()
_C.LOSSES.NAME = 'ce_loss'

# ms loss
_C.LOSSES.MULTI_SIMILARITY_LOSS = CN()
_C.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS = 2.0
_C.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG = 40.0
_C.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING = True

# Data option
_C.DATA = CN()
_C.DATA.TRAIN_IMG_SOURCE = '/data/home/shruti/voxceleb/fabnet/vox2_mp4/train_classification.txt'
_C.DATA.TEST_IMG_SOURCE = '/data/home/shruti/voxceleb/fabnet/vox2_mp4/test_classification.txt'
_C.DATA.TRAIN_BATCHSIZE = 70
_C.DATA.TEST_BATCHSIZE = 256
_C.DATA.NUM_WORKERS = 10
_C.DATA.NUM_INSTANCES = 5

# Input option
_C.INPUT = CN()

# INPUT CONFIG
_C.INPUT.MODE = 'BGR'
_C.INPUT.PIXEL_MEAN = '/data/home/shruti/voxceleb/fabnet/vox2_mp4/mean.npy'
_C.INPUT.PIXEL_STD = '/data/home/shruti/voxceleb/fabnet/vox2_mp4/std.npy'

_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.ORIGIN_SIZE = 256
_C.INPUT.CROP_SCALE = [0.16, 1]
_C.INPUT.CROP_SIZE = 227

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.IS_FINETURN = False
_C.SOLVER.FINETURN_MODE_PATH = ''
_C.SOLVER.MAX_ITERS = 4000
_C.SOLVER.STEPS = [1000, 2000, 3000]
_C.SOLVER.OPTIMIZER_NAME = 'SGD'
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 200
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.CHECKPOINT_PERIOD = 200
_C.SOLVER.RNG_SEED = 1

# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = 20
_C.LOGGER.STREAM = 'stdout'

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.VERBOSE = 200
_C.VALIDATION.IS_VALIDATION = True
