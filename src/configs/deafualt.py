from fvcore.common.config import CfgNode


_C = CfgNode()

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.DATASET = "ImgData"
_C.TRAIN.BATCH_SIZE = 2

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.NUM_FRAMES = 10
_C.DATA.SAMPLING_RATE = 2
_C.DATA.RAIN_JITTER_SCALES = [256, 320]
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.TEST_CROP_SIZE = 224
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]
_C.DATA.DETECTION_SCORE_THRESH = 0.8
_C.DATA.TRAIN_PREDICT_BOX_LISTS = ["my_dataset_train_copy.csv",]
_C.DATA.TEST_PREDICT_BOX_LISTS = ["person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"]
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
_C.DATA.FRAME_LIST_DIR = ("data/my_dataset/frame_lists/")
_C.DATA.TRAIN_LISTS = ["train.csv"] 
_C.DATA.TEST_LISTS = ["val.csv"]
_C.DATA.FRAME_DIR = "data/my_dataset/rawframes/"
_C.DATA.TRAIN_GT_BOX_LISTS = ["my_dataset_train_copy.csv"]
_C.DATA.ANNOTATION_DIR = ("data/my_dataset/annotations/")
_C.DATA.FULL_TEST_ON_VAL = False

# -----------------------------------------------------------------------------
# Detection options
# -----------------------------------------------------------------------------
_C.DETECTION = CfgNode()
_C.DETECTION.ENABLE = True
_C.DETECTION.ALIGNED = True
# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

_C.MODEL.FC_INIT_STD = 0.01
_C.MODEL.DROPCONNECT_RATE = 0.0
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.ARCH = "slowfast"
_C.MODEL.MODEL_NAME = "SlowFast"
_C.MODEL.LOSS_FUNC = "cross_entropy"
_C.MODEL.DROPOUT_RATE = 0.5
_C.MODEL.HEAD_ACT = "softmax"
# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()
_C.SLOWFAST.ALPHA = 4
_C.SLOWFAST.BETA_INV = 8
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2
_C.SLOWFAST.FUSION_KERNEL_SZ = 7
# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
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

#_C.DATAIf true, initialize the gamma of the final BN of each block to zero.
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
_C.NONLOCAL.POOL = [[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]]

# -----------------------------------------------------------------------------
# BatchNormal options
# -----------------------------------------------------------------------------
_C.BN = CfgNode()
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_BATCHES_PRECISE = 200
# -----------------------------------------------------------------------------
# BatchNormal options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.LR_POLICY = "cosine"
_C.SOLVER.STEPS = [0, 10, 15, 20]
_C.SOLVER.LRS = [1, 0.1, 0.01, 0.001]
_C.SOLVER.MAX_EPOCH = 20
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-7
_C.SOLVER.WARMUP_EPOCHS = 5.0
_C.SOLVER.WARMUP_START_LR = 0.000125
_C.SOLVER.OPTIMIZING_METHOD = "sgd"
# -----------------------------------------------------------------------------
# Test options
# -----------------------------------------------------------------------------
_C.TEST = CfgNode()
_C.TEST.ENABLE = True
_C.TEST.DATASET = "ImgData"
_C.TEST.BATCH_SIZE = 8
# -----------------------------------------------------------------------------
# DataLoader options
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 2
_C.DATA_LOADER.PIN_MEMORY = True
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

def get_cfg():
    return _C.clone()