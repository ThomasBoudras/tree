import geopandas as gpd
import numpy as np
import os
import torch
from torch.utils.data import  Dataset
from pathlib import Path
from torchvision.transforms import v2
import json
from datetime import datetime


from src.utils.dataset_utils import get_window, found_nearest_date

class sentinelDataset(Dataset):
    def __init__(
        self,
        input_path : Path,
        target_path : Path,
        target_unit : str,
        geometries_path : Path,
        replace_nan_by_zero_in_target: bool,
        min_year: int,
        max_year: int,
        target_resolution: float,
        input_resolution : int,
        nb_timeseries_image,
        patch_size_input, 
        duplication_level_noise,
        split: str = None,
        transform_input : v2 = None,
        transform_target: v2 = None,
    ):
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
        self.nb_timeseries_image = nb_timeseries_image
        self.duplication_level_noise = duplication_level_noise
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.patch_size_input = patch_size_input
        self.max_bounds_size = patch_size_input*input_resolution + 10
        
        self.geometries_path = geometries_path
        self.geometries = gpd.read_file(self.geometries_path)
        self.geometries['vrt_list_timeseries'] = self.geometries['vrt_list_timeseries'].apply(json.loads) 
        self.geometries = self.geometries.query("split == @split and @min_year <= lidar_year <= @max_year").reset_index(drop=True)

        

    def __len__(self):
        return self.geometries.shape[0]
    
    def update_transforms(self, transform_input : v2 = None, transform_target: v2 = None):
        self.transform_input = transform_input
        self.transform_target = transform_target


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

        lidar_date = row_gdf["lidar_acquisition_date"]
        vrt_list = row_gdf["vrt_list_timeseries"]
        input = []
        input_date = []
        s1_vrt_list_path = os.path.join(self.input_path, f"lidar_date_{lidar_date[:6]}/s1/vrt_files")
        s2_vrt_list_path = os.path.join(self.input_path, f"lidar_date_{lidar_date[:6]}/s2/vrt_files")
        
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
        
        if self.transform_input:
            input = torch.stack([self.transform_input(image) for image in input], dim=0)
        else : 
            input = np.stack(input, axis=0).mean(axis=0)

        if self.transform_target:
            target = self.transform_target(target)

        input_date = torch.from_numpy(np.stack(input_date, axis=0))
        bounds = torch.tensor(bounds)
        target_year = torch.tensor(target_year)

        meta_data = {"bounds" : bounds, "years" : target_year, "dates" : input_date}

        return input, target, meta_data
    
    
