import pathlib

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

NB_MIX = 4

# basepath = pathlib.Path("../Datasets/datasets_21110115")
basepath = pathlib.Path("/datasets")
TR_SAT_PATH = basepath / "sat"
TR_MAP_PATH = basepath / "map"

VA_SAT_PATH = basepath / "valid/sat"
VA_MAP_PATH = basepath / "valid/map"

EPOCHS = 20
_train_length = len(list(TR_SAT_PATH.glob("*.png")))
_var_length = len(list(VA_SAT_PATH.glob("*.png")))

STEPS_PER_EPOCH = _train_length // BATCH_SIZE

EPOCHS = 20
__val_subsplits = 5
VALIDATION_STEPS = _var_length // BATCH_SIZE // __val_subsplits


# Config for saving path
MODEL_SAVE_PATH = pathlib.Path("./savedmodel")
CHECKPOINT_PATH = pathlib.Path("./checkpoint")
LOG_PATH = pathlib.Path("./log")
