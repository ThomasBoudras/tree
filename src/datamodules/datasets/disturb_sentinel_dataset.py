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
        source,
        input_path : Path,
        target_path : Path,
        target_unit : str,
        patch_size,
        replace_nan_by_zero_in_target: bool,
        year1: int,
        year2: int,
        resolution: float,
        change_threshold,
        geometries_path,
        transform_input : v2 = None,
        transform_target: v2 = None,
    ):
        self.source = source
        self.input_path = input_path
        self.target_path = target_path
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.year1 = year1
        self.year2 = year2
        self.target_unit = target_unit
        self.patch_size = patch_size
        self.resolution = resolution
        self.change_threshold = change_threshold
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.geometries = (
            gpd.read_file(geometries_path)
            .query("split == @split and @min_year <= lidar_year <= @max_year")
            .reset_index(drop=True)
            )   

    def __len__(self):
        return 
    
    def update_transforms(self, transform_input : v2 = None, transform_target: v2 = None):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __getitem__(self, ix):
        
        row_gdf = self.geometries.loc[ix]

        bounds = row_gdf["geometry"].bounds
        date_year1 = row_gdf["date_year1"]
        date_year2 = row_gdf["date_year2"]


        def _get_target_input_by_year(year, date) :
            vrt_path  = os.path.join(self.target_path, str(year), "lidar.vrt")

            targets = get_window(
                image_path=vrt_path,
                bounds=bounds,
                resolution=self.target_resolution
            )
            targets = targets.astype(np.float32).transpose(1, 2, 0)

            
            if self.replace_nan_by_zero_in_target:
                targets[np.isnan(targets)] = 0

            scaling_factor = {"m": 1, "dm": 10, "cm": 100}
            targets = targets / scaling_factor[self.target_unit]


            # Load input
            inputs, _ = found_nearest_date(date=date, min_year=self.year1, max_year=self.year2)

            inputs = []
            for source in self.input_sources :
                vrt_path = os.path.join(self.input_path, str(year), f"{date}_plus_minus_30_days", source, f"{source}_EPSG2154.vrt")

                input_source = get_window(
                    image_path=vrt_path,
                    bounds=bounds,
                    resolution=(self.target_resolution*self.zoom_factor)
                    )

                input_source = input_source.astype(np.float32).transpose(1, 2, 0)
                input_source[np.isneginf(input_source)] = 0
                inputs.append(input_source)
            inputs = np.concatenate(inputs, axis=2)

            return targets, inputs
        
        targets_year1, inputs_year1 = _get_target_input_by_year(self.year1, date_year1)
        targets_year2, inputs_year2 = _get_target_input_by_year(self.year2, date_year2)

        
        relative_diff = np.abs(targets_year1- targets_year2)

        targets = relative_diff > self.change_threshold

        if self.transform_input:
            inputs_year1 = self.transform_input(inputs_year1)
            inputs_year2 = self.transform_input(inputs_year2)

        if self.transform_target:
            target = self.transform_target(target)

        meta_data = {"bounds" : bounds}

        return (inputs_year1, inputs_year2), targets, meta_data
    
    def custom_collate_fn(self, batch):
        input = [item[0] for item in batch]  
        target = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(input, dim=0)
        batch_target = torch.stack(target, dim=0)

        batch_meta_data = {key: [d[key] for d in meta_data] for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data

