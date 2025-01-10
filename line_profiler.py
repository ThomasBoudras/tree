import os
from pathlib import Path
from src.datamodules.datasets.timeseries_sentinel_dataset import sentinelDataset  # Assure-toi que le chemin est correct
from src.utils.dataset_utils import BottomLeftCrop
from torchvision.transforms import v2
from line_profiler import LineProfiler
import numpy as np

# Configuration des chemins et des paramètres
data_dir = "/work/data"
normalization_save_path = "/work/work/tree/normalization_values_sentinel_1_2/"
patch_size_input = 128
patch_size_target = 32
input_mean = np.load(os.path.join(normalization_save_path, "mean.npy"))
input_std = np.load(os.path.join(normalization_save_path, "std.npy"))
transform_input = v2.Compose([v2.ToTensor(), BottomLeftCrop(patch_size_input), v2.Normalize(mean=input_mean, std=input_std)])  # does not scale by 255
transform_target = v2.Compose([v2.ToTensor(), BottomLeftCrop(patch_size_target)]) 
config = {
    "input_path": f"{data_dir}/sentinel/timeseries/one_month_image",
    "target_path": f"{data_dir}/lidar/france",
    "geometries_path": f"{data_dir}/utils/geometries_120_days_cleaned.geojson",
    "min_year": 2021,
    "max_year": 2023,
    "input_resolution": 10,
    "target_resolution": 2.5,
    "nb_timeseries_image": 8,
    "duplication_level_noise": 80,
    "replace_nan_by_zero_in_target": True,
}

            
# Initialisation du dataset
def init_dataset():
    dataset = sentinelDataset(
        input_path=Path(config["input_path"]),
        target_path=Path(config["target_path"]),
        geometries_path=Path(config["geometries_path"]),
        replace_nan_by_zero_in_target=config["replace_nan_by_zero_in_target"],
        min_year=config["min_year"],
        max_year=config["max_year"],
        target_unit="m",  # Assumes target is in meters
        target_resolution=config["target_resolution"],
        input_resolution=config["input_resolution"],
        nb_timeseries_image=config["nb_timeseries_image"],
        duplication_level_noise=config["duplication_level_noise"],
        transform_input=transform_input,
        transform_target=transform_target,
        split="train",
        max_bounds_size=325
    )
    return dataset


# Profiling de la méthode __getitem__
def profile_getitem(dataset, indices):
    profiler = LineProfiler()
    profiler.add_function(dataset.__getitem__)
    print(f"Profiling __getitem__ for dataset indices: {indices}...")
    for idx in indices:
        print(f"Profiling index {idx}...")
        dataset[idx]
    profiler.print_stats(output_unit=1)

if __name__ == "__main__":
    # Initialisation et profiling
    dataset = init_dataset()
    print(f"len dataset {len(dataset)}")
    indices_to_test = [1, 2, 3, 4]  # Indices à tester
    profile_getitem(dataset, indices_to_test)
