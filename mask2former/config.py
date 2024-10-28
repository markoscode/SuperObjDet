# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    
    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    #SWIN SNNET CONFIG Parameters
    cfg.SWIN_SNNET = CN()
    cfg.SWIN_SNNET.LAYER_TYPE = "fc"

    # TINY Parameters
    cfg.SWIN_SNNET.TINY = CN()
    cfg.SWIN_SNNET.TINY.PRETRAIN_IMG_SIZE = 224
    cfg.SWIN_SNNET.TINY.PATCH_SIZE = 4
    cfg.SWIN_SNNET.TINY.EMBED_DIM = 96
    cfg.SWIN_SNNET.TINY.DEPTHS = [2, 2, 6, 2]
    cfg.SWIN_SNNET.TINY.NUM_HEADS = [3, 6, 12, 24]
    cfg.SWIN_SNNET.TINY.WINDOW_SIZE = 7
    cfg.SWIN_SNNET.TINY.MLP_RATIO = 4.0
    cfg.SWIN_SNNET.TINY.QKV_BIAS = True
    cfg.SWIN_SNNET.TINY.QK_SCALE = None
    cfg.SWIN_SNNET.TINY.DROP_RATE = 0.0
    cfg.SWIN_SNNET.TINY.ATTN_DROP_RATE = 0.0
    cfg.SWIN_SNNET.TINY.DROP_PATH_RATE = 0.3
    cfg.SWIN_SNNET.TINY.APE = False
    cfg.SWIN_SNNET.TINY.PATCH_NORM = True
    cfg.SWIN_SNNET.TINY.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SWIN_SNNET.TINY.USE_CHECKPOINT = False
    cfg.SWIN_SNNET.TINY.WEIGHTS= "swin_tiny_patch4_window7_224.pkl"

    # SMALL Parameters
    cfg.SWIN_SNNET.SMALL = CN()
    cfg.SWIN_SNNET.SMALL.PRETRAIN_IMG_SIZE = 224
    cfg.SWIN_SNNET.SMALL.PATCH_SIZE = 4
    cfg.SWIN_SNNET.SMALL.EMBED_DIM = 96
    cfg.SWIN_SNNET.SMALL.DEPTHS = [2, 2, 6, 2]
    cfg.SWIN_SNNET.SMALL.NUM_HEADS = [3, 6, 12, 24]
    cfg.SWIN_SNNET.SMALL.WINDOW_SIZE = 7
    cfg.SWIN_SNNET.SMALL.MLP_RATIO = 4.0
    cfg.SWIN_SNNET.SMALL.QKV_BIAS = True
    cfg.SWIN_SNNET.SMALL.QK_SCALE = None
    cfg.SWIN_SNNET.SMALL.DROP_RATE = 0.0
    cfg.SWIN_SNNET.SMALL.ATTN_DROP_RATE = 0.0
    cfg.SWIN_SNNET.SMALL.DROP_PATH_RATE = 0.3
    cfg.SWIN_SNNET.SMALL.APE = False
    cfg.SWIN_SNNET.SMALL.PATCH_NORM = True
    cfg.SWIN_SNNET.SMALL.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SWIN_SNNET.SMALL.USE_CHECKPOINT = False
    cfg.SWIN_SNNET.SMALL.WEIGHTS= "swin_small_patch4_window7_224.pkl"

    # BASE Parameters
    cfg.SWIN_SNNET.BASE = CN()
    cfg.SWIN_SNNET.BASE.PRETRAIN_IMG_SIZE = 224
    cfg.SWIN_SNNET.BASE.PATCH_SIZE = 4
    cfg.SWIN_SNNET.BASE.EMBED_DIM = 96
    cfg.SWIN_SNNET.BASE.DEPTHS = [2, 2, 6, 2]
    cfg.SWIN_SNNET.BASE.NUM_HEADS = [3, 6, 12, 24]
    cfg.SWIN_SNNET.BASE.WINDOW_SIZE = 7
    cfg.SWIN_SNNET.BASE.MLP_RATIO = 4.0
    cfg.SWIN_SNNET.BASE.QKV_BIAS = True
    cfg.SWIN_SNNET.BASE.QK_SCALE = None
    cfg.SWIN_SNNET.BASE.DROP_RATE = 0.0
    cfg.SWIN_SNNET.BASE.ATTN_DROP_RATE = 0.0
    cfg.SWIN_SNNET.BASE.DROP_PATH_RATE = 0.3
    cfg.SWIN_SNNET.BASE.APE = False
    cfg.SWIN_SNNET.BASE.PATCH_NORM = True
    cfg.SWIN_SNNET.BASE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SWIN_SNNET.BASE.USE_CHECKPOINT = False
    cfg.SWIN_SNNET.BASE.WEIGHTS= "swin_base_patch4_window12_384.pkl"

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
