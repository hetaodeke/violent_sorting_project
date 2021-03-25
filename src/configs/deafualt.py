from fvcore.common.config import CfgNode

_C = CfgNode()

_C.DATA = CfgNode()
_C.DATA.MEAN = [0.45, 0.45, 0.45]
_C.DATA.STD = [0.225, 0.225, 0.225]
_C.DATA.BGR = False
_C.DATA.RANDOM_FLIP = True
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]
_C.DATA.TRAIN_USE_COLOR_AUGMENTATION = False
_C.DATA.TRAIN_PCA_JITTER_ONLY = True
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
_C.DATA.TEST_FORCE_FLIP = False
_C.DATA.IMG_PROC_BACKEND = "cv2"

_C.MODEL = CfgNode()
_C.MODEL.FC_INIT_STD = 0.01
_C.MODEL.DROPCONNECT_RATE = 0.0

_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]




def get_cfg():
    return _C.clone()