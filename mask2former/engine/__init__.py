# Copyright (c) Facebook, Inc. and its affiliates.

# For inference, we only need defaults - skip training imports for compatibility
# with older detectron2 versions (v0.6)
_INFERENCE_ONLY = True

if not _INFERENCE_ONLY:
    from .launch import *
    from .train_loop import *
    from .hooks import *
    __all__ = [k for k in globals().keys() if not k.startswith("_")]
else:
    __all__ = []

# Always import defaults (contains DefaultPredictor for inference)
from .defaults import *