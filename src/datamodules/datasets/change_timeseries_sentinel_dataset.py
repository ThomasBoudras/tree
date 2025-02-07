import geopandas as gpd
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import  Dataset
from pathlib import Path
from torchvision.transforms import v2

from src.utils.dataset_utils import get_window, get_patch_bounds
import json
from datetime import datetime

class sentinelDataset(Dataset):
    def __init__(
        self,
        input_path : Path,
        target_path : Path,
        replace_nan_by_zero_in_target: bool,
        target_resolution: float,
        input_resolution : int,
        nb_timeseries_image,
        patch_size_input, 
        duplication_level_noise,
        method_change, 
        geometries_path,
        proportion_change, 
        split = "train",
        transform_input : v2 = None,
        transform_target: v2 = None,
    ):
        self.input_path = input_path
        self.target_path = target_path
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.target_resolution = target_resolution
        self.input_resolution = input_resolution
        self.nb_timeseries_image = nb_timeseries_image
        self.patch_size_input = patch_size_input
        self.max_bounds_size = (patch_size_input+1)*input_resolution # we take a +1 pixel security
        self.duplication_level_noise = duplication_level_noise
        self.method_change = method_change

        self.geometries_path = geometries_path
        self.proportion_change = proportion_change
        self.split = split
        self.geometries = self.get_split_geometries(geometries_path, split, proportion_change)
        
        self.transform_input = transform_input
        self.transform_target = transform_target
    
    def __len__(self):
        return len(self.geometries)
    
    def update_transforms(self, transform_input : v2 = None, transform_target: v2 = None):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def get_split_geometries(self, geometries_path, split, proportion_change) :
        geometries = gpd.read_file(geometries_path)
        geometries[f"vrt_list_1"] = geometries['vrt_list_1'].apply(json.loads) 
        geometries[f"vrt_list_2"] = geometries['vrt_list_2'].apply(json.loads) 

        geometries = geometries[geometries['split'] == split ].reset_index(drop=True) 
        geometries_with_changes = geometries[geometries['has_changes'] == True].reset_index(drop=True)
        geometries_without_changes = geometries[geometries['has_changes'] == False].reset_index(drop=True)
        
        initial_proportion_change = len(geometries_with_changes)/len(geometries)
        if initial_proportion_change < proportion_change :
            num_no_change_to_keep = int(len(geometries_with_changes)/proportion_change - len(geometries_with_changes))
            geometries_without_changes= geometries_without_changes.sample(n=num_no_change_to_keep, random_state=42)

        if initial_proportion_change > proportion_change :
            num_change_to_keep = int(len(geometries_without_changes)/(1-proportion_change) - len(geometries_without_changes))
            geometries_with_changes= geometries_with_changes.sample(n=num_change_to_keep, random_state=42)

        return pd.concat([geometries_with_changes, geometries_without_changes], ignore_index=True)


    def get_date_from_vrt_name(self, vrt_name) :
        return datetime.strptime(vrt_name.split('_')[1].split('.')[0], "%Y%m%d")
    
    def get_n_closest_dates(
        self,
        tuples_list,
        reference_date,
    ) :
        """
        Returns the n tuples whose first date is closest to the reference date,
        sorted by ascending order of the first date in each tuple.
        """
        
        ref_date = datetime.strptime(reference_date, "%Y%m%d")

        n = min(self.nb_timeseries_image, len(tuples_list))

        # Sort the list by proximity to the reference date
        sorted_by_proximity = sorted(
            tuples_list,
            key=lambda x: abs(datetime.strptime(x[0].split('_')[1].split('.')[0], "%Y%m%d") - ref_date)
        )
        
        # Take the n closest elements and sort them by ascending order of date
        closest_n = sorted(
        sorted_by_proximity[:n],
        key=lambda x: datetime.strptime(x[0].split('_')[1].split('.')[0], "%Y%m%d")
        )
        
        return closest_n
    
    
    def custom_collate_fn(self, batch):
        input = [item[0] for item in batch]  
        target = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(input, dim=0)
        batch_target = torch.stack(target, dim=0)

        batch_meta_data = {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data


    def _get_target_input_by_year(self, bounds, row, year_number) :
        lidar_date = row[f"lidar_acquisition_date_{year_number}"]
        if len(lidar_date) == 6 : #If you only have the month, place yourself in the middle of the month
            lidar_date = lidar_date + "15"
        vrt_list = row[f"vrt_list_{year_number}"]
        target_unit = row[f"unit_lidar_{year_number}"]
        area = row[f"area"]

        vrt_path  = os.path.join(self.target_path, area, f"lidar_{lidar_date[:4]}/lidar_masked/full.vrt")

        target = get_window(
            image_path=vrt_path,
            bounds=bounds,
            resolution=self.target_resolution
        )
        target = target.astype(np.float32).transpose(1, 2, 0)

        if self.replace_nan_by_zero_in_target:
            target[np.isnan(target)] = 0

        scaling_factor = {"m": 1, "dm": 10, "cm": 100}
        target = target / scaling_factor[target_unit]


        input = []
        input_date = []
        s1_vrt_list_path = os.path.join(self.input_path, f"year_{year_number}/lidar_date_{lidar_date[:6]}/s1/vrt_files")
        s2_vrt_list_path = os.path.join(self.input_path, f"year_{year_number}/lidar_date_{lidar_date[:6]}/s2/vrt_files")
        
        vrt_list = self.get_n_closest_dates(vrt_list, lidar_date)


        for s2_s1_vrt in vrt_list[:self.nb_timeseries_image] :
            s2_vrt = os.path.join(s2_vrt_list_path, s2_s1_vrt[0])
            s1_vrt = os.path.join(s1_vrt_list_path, s2_s1_vrt[1])
            s2_image = get_window(
                image_path=s2_vrt,
                bounds=bounds,
                resolution=self.input_resolution
            )
            s1_image = get_window(
                image_path=s1_vrt,
                bounds=bounds,
                resolution=self.input_resolution
            )
            if len(s1_image.shape) == 1 or len(s2_image.shape) == 1 :
                print(f"row problématique {row}")
                print(f"original bounds {row['geometry'].bounds}, max_bounds_soze {self.max_bounds_size}, patch size input  {self.patch_size_input}, input_resolution {self.input_resolution}")
            image = np.concatenate((s2_image,s1_image), axis=0)
            image = image.astype(np.float32).transpose(1, 2, 0)
            image[~np.isfinite(image)] = 0
            input.append(image)
            input_date.append(int(self.get_date_from_vrt_name(s2_s1_vrt[0]).timetuple().tm_yday))

        while len(input) < self.nb_timeseries_image :
            idx = np.random.randint(0, len(input))
            image_added = input[idx] 
            date_added = input_date[idx]

            if self.duplication_level_noise : 
                noise = np.random.normal(0, self.duplication_level_noise, image_added.shape).astype(np.float32)
                image_added = image_added + noise 

            input.insert(idx + 1, image_added)
            input_date.insert(idx + 1, date_added)

        return input, input_date, target

    def __getitem__(self, ix):
        
        row_gdf = self.geometries.loc[ix]

        bounds = list(row_gdf["geometry"].bounds)
        position = row_gdf["relative_position"]
        rotation = row_gdf["rotation"]
        mirror = row_gdf["mirror"]
        bounds = get_patch_bounds(bounds, self.max_bounds_size, position)

        input_year_1, input_dates_year_1, target_year_1 = self._get_target_input_by_year(bounds, row = row_gdf, year_number=1)
        input_year_2, input_dates_year_2, target_year_2 = self._get_target_input_by_year(bounds, row = row_gdf, year_number=2)

        if self.transform_input:
            input_year_1 = torch.stack([self.transform_input(image) for image in input_year_1], dim=0)
            input_year_2 = torch.stack([self.transform_input(image) for image in input_year_2], dim=0)
        else : 
            input_year_1 = torch.from_numpy(np.stack(input_year_1, axis=0).mean(axis=0))
            input_year_2 = torch.from_numpy(np.stack(input_year_2, axis=0).mean(axis=0))

        if self.transform_target:
            target_year_1 = self.transform_target(target_year_1)
            target_year_2 = self.transform_target(target_year_2)
        else :
            target_year_1 = torch.from_numpy(target_year_1)
            target_year_2 = torch.from_numpy(target_year_2)

        input = torch.cat([input_year_1, input_year_2], dim = 0)
        target = self.method_change(target_year_1, target_year_2)

        if rotation > 0 :
            k = int(rotation//90)
            input = torch.rot90(input, k=k, dims=(-1, -2))
            target = torch.rot90(target, k=k, dims=(-1, -2))
        
        if mirror == "vertical" :
            input = torch.flip(input, dims=[-1])  
            target = torch.flip(target, dims=[-1])  
        
        if mirror == "horizontal" :
            input = torch.flip(input, dims=[-2])  
            target = torch.flip(target, dims=[-2])  


        bounds = torch.tensor(bounds)
        input_dates = input_dates_year_1 + input_dates_year_2  
        input_date = torch.from_numpy(np.stack(input_dates, axis=0))  

        meta_data = {"bounds" : bounds, "dates" : input_date}

        return input, target, meta_data
    
