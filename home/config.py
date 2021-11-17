import pathlib
from datetime import datetime

BUFFER_SIZE = 400
BATCH_SIZE = 64
# Each image is 224x224 in size

# size = 224
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
# basepath = pathlib.Path("/Volumes/RX3070/hagadir/dataset")
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
# VALIDATION_STEPS = _var_length // BATCH_SIZE // __val_subsplits
VALIDATION_STEPS = 5


# util func
def _mkdir_ifne(path: pathlib.Path):
    if not path.exists():
        path.mkdir(parents=True)


# Config for saving path
date = datetime.now().strftime("%y%m%d%H%M")
RES_BASE = pathlib.Path("../../results") / date
MODEL_SAVE_PATH = RES_BASE / pathlib.Path("./savedmodels")
CHECKPOINT_PATH = RES_BASE / pathlib.Path("./checkpoints")
LOG_PATH = RES_BASE / pathlib.Path("./logs")

_mkdir_ifne(RES_BASE)
_mkdir_ifne(MODEL_SAVE_PATH)
_mkdir_ifne(CHECKPOINT_PATH)
_mkdir_ifne(LOG_PATH)
