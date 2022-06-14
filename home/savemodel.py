import functools
from model import residual_unet

input_shape = (256, 256, 3)
base_model_provider = functools.partial(
    residual_unet.unet,
    input_shape=input_shape,
    initial_channels=32,
    name="baseunet",
    parallel_dilated=True,
    nb_layer=4,
)


def savemodel(name):
    path = f"/Volumes/GoogleDrive/マイドライブ/卒研/experiments_splitted/checkpoints/{name}/{name}"
    model = base_model_provider()
    model.load_weights(path)
    model.save(f"/Users/hagayuya/GradRes/モデル実験/trained_model/{name}")


savemodel("ibr05_base_dense")
savemodel("ibr05_base_sparse")
savemodel("ibr10_base_dense")
savemodel("ibr10_base_sparse")
