import pathlib

USE_IMGNET = True

BUFFER_SIZE = 400
BATCH_SIZE = 64
# Each image is 224x224 in size

size = 224
size = 256
IMG_WIDTH = size
IMG_HEIGHT = size
IMG_SIZE = size
INPUT_SIZE = (size, size, size)

IMG_CH = 3
OUT_CH = 1
OUTPUT_CLASSES = 2

NB_MIX = 7

#PATH = pathlib.Path("../Datasets/datasets_21110115")
PATH = pathlib.Path("/datasets")
TR_SAT_PATH = PATH / "sat"
TR_MAP_PATH = PATH / "map"
assert PATH.exists() and TR_SAT_PATH.exists() and TR_MAP_PATH.exists()

VA_SAT_PATH = PATH / "valid/sat"
VA_MAP_PATH = PATH / "valid/map"
assert VA_SAT_PATH.exists() and VA_MAP_PATH.exists()

EPOCHS = 20
VAL_SUBSPLITS = 5
TRAIN_LENGTH = len(list(TR_SAT_PATH.glob("*.png")))
VAR_LENGTH = len(list(VA_SAT_PATH.glob("*.png")))

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = VAR_LENGTH // BATCH_SIZE // VAL_SUBSPLITS
