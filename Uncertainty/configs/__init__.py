def get_cfg():
    """Get the config object.
    In this function, we add new keys that are not in the default one.
    """
    from detectron2.config.defaults import _C
    from detectron2.config import CfgNode as CN

    # add more custom terms
    _C = _C.clone()

    # =============== Augmentation ===============
    _C.INPUT.ROTATE = CN()
    _C.INPUT.ROTATE.ENABLED = True
    _C.INPUT.ROTATE.ROTATE_P = 1.0
    _C.INPUT.ROTATE.HFLIP_P = 0.5
    _C.INPUT.CROP_AND_RESIZE = CN()
    _C.INPUT.CROP_AND_RESIZE.ENABLED = True
    _C.INPUT.CROP_AND_RESIZE.CROP_LENGTH = (8, 16)  # (half height, half width)
    _C.INPUT.CROP_AND_RESIZE.DROP_BOX_THRES = (5, 8)  # (height, width)
    _C.INPUT.CROP_AND_RESIZE.CROP_AND_RESIZE_P = 0.5
    _C.INPUT.SCALE_TRANSFORM = CN()
    _C.INPUT.SCALE_TRANSFORM.ENABLED = True
    _C.INPUT.SCALE_TRANSFORM.SCALE_RANGE = (0.8, 1.2)
    _C.INPUT.SCALE_TRANSFORM.SCALE_P = 0.5
    _C.INPUT.JITTER = CN()
    _C.INPUT.JITTER.ENABLED = True
    _C.INPUT.JITTER.MEAN = 0.0
    _C.INPUT.JITTER.STD = 0.003
    _C.INPUT.JITTER.JITTER_P = 0.5
    _C.INPUT.FIRST_REFL = CN()
    _C.INPUT.FIRST_REFL.ENABLED = True
    _C.INPUT.FIRST_REFL.JITTER_P = 0.5

    _C.MODEL.BACKBONE.NUM_BLOCKS_PER_DOWN = (2, 2, 2, 2)
    _C.MODEL.BACKBONE.DIM_MULTS = (1, 2, 4, 8)
    _C.MODEL.HAS_GLASS = False
    _C.MODEL.HAS_SEM_SEG = False
    _C.MODEL.HAS_OBJ_DET = False

    _C.DATASETS.BASE_PATH = ""

    _C.SOLVER.NAME = "SGD"
    _C.VIS_SEM_SEG_THRESHOLD = 0.5
    _C.MODEL.SEM_SEG_HEAD.PANORAMIC = False
    _C.MODEL.TWO_STAGE = False
    _C.MODEL.CIRCULAR_DEPTH = True
    _C.MODEL.CIRCULAR_SEG_OBJ = False
    _C.MODEL.DROPOUT_RATE = 0.1
    _C.MODEL.BACKBONE.STEM_OUT_CHANNELS = 64

    return _C
