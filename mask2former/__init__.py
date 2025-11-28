# Copyright (c) Facebook, Inc. and its affiliates.

# Pillow removed Image.LINEAR in v10; add alias before Detectron2 imports rely on it.
try:
    from PIL import Image as _Image

    if not hasattr(_Image, "LINEAR"):
        if hasattr(_Image, "Resampling"):
            _Image.LINEAR = _Image.Resampling.BILINEAR  # type: ignore[attr-defined]
        else:
            _Image.LINEAR = _Image.BILINEAR  # type: ignore[attr-defined]
except Exception:
    pass

# NumPy >=1.24 removed `np.int`; older Detectron2 code still references it.
try:
    import numpy as _np

    if not hasattr(_np, "int"):
        _np.int = int  # type: ignore[attr-defined]
except Exception:
    pass

from . import data  # register all new datasets
from . import modeling

# Detectron2<0.7 calls a helper that now errors under NumPy>=2.0 when copy=False.
try:
    from PIL import Image as _PatchedImage  # ensures reference even if earlier import failed
    import numpy as _PatchedNP
    from detectron2.evaluation import sem_seg_evaluation as _sem_seg_eval

    if hasattr(_sem_seg_eval, "load_image_into_numpy_array"):
        def _patched_load_image_into_numpy_array(filename, copy=False, dtype=None):
            pil_image = _PatchedImage.open(filename)
            try:
                array = _PatchedNP.asarray(pil_image, dtype=dtype)
                if copy:
                    array = array.copy()
            finally:
                pil_image.close()
            return array

        _sem_seg_eval.load_image_into_numpy_array = _patched_load_image_into_numpy_array
except Exception:
    pass

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
