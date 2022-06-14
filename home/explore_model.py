import os
import pathlib
from typing import Literal

import fire
import keras
import numpy as np
import pandas as pd
from skimage import filters
from skimage import io as skio
from skimage import morphology, transform
from tqdm import tqdm

from model import losses, metrics, residual_unet


def save_results_to_csv(
    domain: Literal["mass_roads", "ibaraki"],
    dataset: Literal["train", "test"],
    model_root: str,
    model_name: str,
    csv_path: str,
    nb_bins: int = 15,
    epochs: int = 100,
):
    keras.backend.clear_session()

    # ready loss and metrics
    dice_loss = losses.DICELoss(name="dice")
    iou_coef = metrics.iou_coef

    # Choose Dataset
    if domain == "ibaraki":
        dsroot = pathlib.Path("../../DataSets/ibarakimap_s17/") / dataset
        map_paths = sorted(list((dsroot / "map").glob("*.jpg")))
        sat_paths = sorted(list((dsroot / "sat").glob("*.jpg")))

    elif domain == "mass_roads":
        dsroot = pathlib.Path("/Users/hagayuya/DataSets/mass_roads9/train") / dataset
        map_paths = sorted(list((dsroot / "map").glob("*.png")))
        sat_paths = sorted(list((dsroot / "sat").glob("*.png")))
    else:
        raise ValueError

    assert dsroot.exists()

    # Load Pretrained Weights
    model = residual_unet.unet(
        (256, 256, 3),
        name="unet",
        parallel_dilated=True,
        nb_layer=4,
    )
    path = os.path.join(model_root, model_name)
    print(path)
    model_path = os.path.join(path, model_name) + f"_{epochs:03d}"
    model.load_weights(model_path)

    # Calc metrics
    result_list = []
    for idx, map_path in enumerate(tqdm(map_paths)):

        map_im = skio.imread(map_path, as_gray=True)
        thr = filters.threshold_otsu(map_im)
        map_im = map_im > thr
        map_im = map_im[..., None].astype("float32")
        map_im = transform.resize(map_im, (256, 256))
        map_sum = morphology.skeletonize(map_im).sum()

        sat_path = sat_paths[idx]
        sat_im = skio.imread(sat_path) / 255.0
        sat_im = transform.resize(sat_im, (256, 256))

        pred = model.predict(sat_im[None])[0]

        loss = dice_loss(map_im[None], pred[None])
        iou = iou_coef(map_im[None], pred[None])
        result_list.append((idx, map_sum, loss, iou))

    # make Histgram
    result_array = np.array(result_list)
    nums, edges = np.histogram(result_array[..., 1], bins=nb_bins)

    # Bin IDX
    result_df = pd.DataFrame(result_array, columns=("idx", "map_sum", "loss", "iou"))
    result_df["bin_idx"] = None
    for edge_id in range(len(nums)):
        sums_array = result_array[..., 1]
        result_df["bin_idx"][(sums_array >= edges[edge_id])] = edge_id

    result_df.to_csv(csv_path, index=False)


def main(dataset, nb_bins: int = 10):
    # * Transfered Models
    epochs = (50, 100)
    model_root = "../../experiments_autolr/checkpoints"
    results_root = pathlib.Path("../../モデル実験/transfered/")
    domain = "ibaraki"
    model_names = (
        "DICE-DA_GEO-IM_1.0-E100_base1.0",
        "DICE-DA_GEO-IM_1.0-E100_base0.5",
        "DICE-DA_GEO-IM_1.0-E100_base_nopret",
        "DICE-DA_GEO-IM_0.5-E100_base1.0",
        "DICE-DA_GEO-IM_0.5-E100_base0.5",
        "DICE-DA_GEO-IM_0.5-E100_base_nopret",
    )
    for model_name in model_names:
        results_dir = results_root / model_name 
        results_dir.mkdir(parents=True, exist_ok=True)
        for epoch in epochs:
            csv_path = str(results_dir / f"{dataset}_{epoch:03d}.csv")
            save_results_to_csv(
                domain=domain,
                dataset=dataset,
                model_root=model_root,
                model_name=model_name,
                csv_path=csv_path,
                nb_bins=nb_bins,
                epochs=epoch,
            )

    # # * Base Models
    # max_epochs = 50
    # model_root = "../../experiments_bases/checkpoints"
    # results_root = pathlib.Path("../../モデル実験/bases/")
    # domain = "mass_roads"
    # model_names = (
    #     "DICE-DA_GEO_CH-MR9_0.5-E50.csv",
    #     "DICE-DA_GEO_CH-MR9_1.0-E50.csv",
    # )
    # for model_name in model_names:
    #     results_dir = results_root / model_name / "csv"
    #     results_dir.mkdir(parents=True, exist_ok=True)
    #     for epochs in range(10, max_epochs + 1, 10):
    #         csv_path = str(results_dir / f"{epochs:03d}.csv")
    #         save_results_to_csv(
    #             domain=domain,
    #             dataset=dataset,
    #             model_root=model_root,
    #             model_name=model_name,
    #             csv_path=csv_path,
    #             nb_bins=nb_bins,
    #             epochs=epochs,
    #         )


if __name__ == "__main__":
    fire.Fire(main)
