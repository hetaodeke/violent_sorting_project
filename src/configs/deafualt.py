from fvcore.common.config import CfgNode


_C = CfgNode()

# -----------------------------------------------------------------------------
# Train options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.DATASET = "ImgData"
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.IS_VALIDATION = True

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False
# Directory path of frames.
_C.DATA.FRAME_DIR = "/home/andy/Dataset/my_dataset/rawframes/"
# _C.DATA.FRAME_DIR = "/home/andy/Dataset/ava/rawframes/"

# Directory path for files of frame lists.
_C.DATA.FRAME_LIST_DIR = (
    "/home/andy/Dataset/my_dataset/frame_lists/"
)
# _C.DATA.FRAME_LIST_DIR = (
#     "/home/andy/Dataset/ava/frame_list/"
# )
# Directory path for annotation files.
_C.DATA.ANNOTATION_DIR = (
    "/home/andy/Dataset/my_dataset/annotations/"
)
# _C.DATA.ANNOTATION_DIR = (
#     "/home/andy/Dataset/ava/annotations/"
# )
# Filenames of training samples list files.
_C.DATA.TRAIN_LISTS = ["train.csv"]
# _C.DATA.TRAIN_LISTS = ["train_extract.csv"]

# Filenames of test samples list files.
_C.DATA.TEST_LISTS = ["val.csv"]
# _C.DATA.TEST_LISTS = ["val_extract.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.DATA.TRAIN_GT_BOX_LISTS = ["my_dataset_train.csv"]
# _C.DATA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.1_extract.csv"]

# Filenames of box list files for train.
_C.DATA.TRAIN_PREDICT_BOX_LISTS = ["my_dataset_train_predicted_boxes.csv"]
# _C.DATA.TRAIN_PREDICT_BOX_LISTS = ["ava_train_predicted_boxes_extract.csv"]

# Filenames of box list files for test.
_C.DATA.TEST_PREDICT_BOX_LISTS = ["my_dataset_val_predicted_boxes.csv"]
# _C.DATA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes_extract.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.DATA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.DATA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.DATA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.DATA.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.DATA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.DATA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
# _C.DATA.LABEL_MAP_FILE = "ava_action_list_v2.1_for_activitynet_2018.pbtxt"
_C.DATA.LABEL_MAP_FILE = "my_dataset_action_list.pbtxt"

# The name of the file to the ava exclusion.
_C.DATA.EXCLUSION_FILE = "my_dataset_val_excluded_timestamps.csv"
# _C.DATA.EXCLUSION_FILE = "ava_train_excluded_timestamps_v2.1_extract.csv"

# The name of the file to the ava groundtruth.
_C.DATA.GROUNDTRUTH_FILE = "my_dataset_val.csv"
# _C.DATA.GROUNDTRUTH_FILE = "ava_val_v2.1_extract.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.DATA.IMG_PROC_BACKEND = "cv2"

# ---------------------------------------------------------------------------- #
# Skeleton Dataset options
# ---------------------------------------------------------------------------- #
_C.SKELETON = CfgNode()

_C.SKELETON.TRAIN_DATA_PATH = "data/ucf101_skeleton/train_data.npy"

_C.SKELETON.TEST_DATA_PATH = "data/ucf101_skeleton/val_data.npy"

_C.SKELETON.TRAIN_LABEL_FILE = "data/ucf101_skeleton/train_label.pkl"

_C.SKELETON.TEST_LABEL_FILE = "data/ucf101_skeleton/val_label.pkl"

_C.SKELETON.PERSON_NUM_OUT = 5

_C.SKELETON.NUM_PATHWAYS = 2

_C.SKELETON.MAX_FRAME = 150

_C.SKELETON.DEBUG = False

_C.SKELETON.RANDOM_CHOOSE = True

_C.SKELETON.RANDOM_MOVE = True

_C.SKELETON.WINDOW_SIZE = 150

_C.SKELETON.MMAP = True

_C.SKELETON.IN_CHANNELS = 3

_C.SKELETON.EDGE_IMPORTANCE_WEIGHT = True

_C.SKELETON.GRAPH_ARGS = CfgNode()

_C.SKELETON.GRAPH_ARGS.LAYOUT = "openpose"

_C.SKELETON.GRAPH_ARGS.STRATEGY = "spatial"


# -----------------------------------------------------------------------------
# Detection options
# -----------------------------------------------------------------------------
_C.DETECTION = CfgNode()
_C.DETECTION.ENABLE = True
_C.DETECTION.ALIGNED = True
# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7
# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

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
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow", "x3d"]
_C.MODEL.DOUBLE_PATHWAY_ARCH = ["slowfast", "sf_gcn"]
_C.MODEL.MULTI_PATHWAY_ARCH = [""]
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
_C.BN.NORM_TYPE = "batchnorm"

# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.LR_DECAY = 0.1
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
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

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
_C.CUDNN_DETERMINISTIC = False

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

def get_cfg():
    return _C.clone()