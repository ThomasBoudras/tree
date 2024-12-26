import geopandas as gpd
import numpy as np
import os
import torch
from torch.utils.data import  Dataset
from pathlib import Path
from torchvision.transforms import v2

from src.utils.dataset_utils import get_window, found_nearest_date

class sentinelDataset(Dataset):
    def __init__(
        self,
        input_sources: list[str],
        input_path : Path,
        target_path : Path,
        target_unit : str,
        geometries_path : Path,
        replace_nan_by_zero_in_target: bool,
        min_year: int,
        max_year: int,
        target_resolution: float,
        input_resolution : int,
        max_bounds_size,
        split: str = None,
        transform_input : v2 = None,
        transform_target: v2 = None,
    ):
        self.input_sources = input_sources
        self.input_path = input_path
        self.target_path = target_path
        self.split = split
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.min_year = min_year
        self.max_year = max_year
        self.target_unit = target_unit
        self.target_resolution = target_resolution
        self.input_resolution = input_resolution
        self.max_bounds_size = max_bounds_size
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.geometries = (
            gpd.read_file(geometries_path)
            .query("split == @split and @min_year <= lidar_year <= @max_year")
            .reset_index(drop=True)
)     

    def __len__(self):
        return self.geometries.shape[0]
    
    def update_transforms(self, transform_input : v2 = None, transform_target: v2 = None):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __getitem__(self, ix):

        row_gdf = self.geometries.loc[ix]

        #Load target
        target_year = row_gdf["lidar_year"]
        bounds = list(row_gdf["geometry"].bounds)
        bounds[2] = min(bounds[0] + self.max_bounds_size, bounds[2])
        bounds[3] =  min(bounds[1] + self.max_bounds_size, bounds[3])


        vrt_path  = os.path.join(self.target_path, str(target_year), "lidar.vrt")


        target = get_window(
            image_path=vrt_path,
            bounds=bounds,
            resolution=self.target_resolution
        )
        target = target.astype(np.float32).transpose(1, 2, 0)

        if self.replace_nan_by_zero_in_target:
            target[np.isnan(target)] = 0

        scaling_factor = {"m": 1, "dm": 10, "cm": 100}
        target = target / scaling_factor[self.target_unit]

        # Load input
        lidar_month = row_gdf["lidar_acquisition_date"][:6]

        input = []
        for source in self.input_sources :
            vrt_path = os.path.join(self.input_path, f"lidar_date_{lidar_month}" , source, f"{source}_EPSG2154.vrt")

            input_source = get_window(
                image_path=vrt_path,
                bounds=bounds,
                resolution=(self.input_resolution)
                )

            input_source = input_source.astype(np.float32).transpose(1, 2, 0)
            input_source[~np.isfinite(input_source)] = 0
            input.append(input_source)
        input = np.concatenate(input, axis=2)

        if self.transform_input:
            input = self.transform_input(input)

        if self.transform_target:
            target = self.transform_target(target)

        bounds = torch.tensor(bounds)
        target_year = torch.tensor(target_year)

        meta_data = {"bounds" : bounds, "years" : target_year}
        return input, target, meta_data
    
    def custom_collate_fn(self, batch):
        input = [item[0] for item in batch]  
        target = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(input, dim=0)
        batch_target = torch.stack(target, dim=0)

        batch_meta_data = {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data

