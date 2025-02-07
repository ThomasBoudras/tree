# Dataset
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
import os
from src.utils.dataset_utils import get_window
import torch 
import logging

def get_grid(patch_size, crop_size, global_bounds):
    patch_bounds = []
    
    min_x = global_bounds[0]
    min_y = global_bounds[1]
    max_x = global_bounds[2]
    max_y = global_bounds[3]

    x_start = min_x - crop_size
    y_start = min_y - crop_size

    while (x_start <= max_x) and (y_start <= max_y):
        x_stop = x_start + patch_size
        y_stop = y_start + patch_size

        # Save pixel_patch_bounds
        patch_bounds.append((x_start, y_start, x_stop, y_stop))

        if y_stop - 2 * crop_size <= max_y:
            y_start = y_stop - 2 * crop_size
        else:
            y_start = min_y
            x_start = x_stop - 2 * crop_size
    return patch_bounds


def adjust_bounds_to_1point5_res(transform_c, transform_f, bounds):
    # Adjust bounds to avoid rounding issues when loading from input tif/vrt
    c = transform_c
    f = transform_f
    # Extend bounds if necessary
    if (bounds[0] - c) % 3:  # extend xmin on the left
        bounds[0] = bounds[0] - 1
        if (bounds[0] - c) % 3:
            bounds[0] = bounds[0] - 1
    if (bounds[2] - c) % 3:  # extend xmax on the right
        bounds[2] = bounds[2] + 1
        if (bounds[2] - c) % 3:
            bounds[2] = bounds[2] + 1

    if (f - bounds[1]) % 3:  # extend ymin on the bottom
        bounds[1] = bounds[1] - 1
        if (f - bounds[1]) % 3:
            bounds[1] = bounds[1] - 1
    if (f - bounds[3]) % 3:  # extend ymax on the top
        bounds[3] = bounds[3] + 1
        if (f - bounds[3]) % 3:
            bounds[3] = bounds[3] + 1
    return bounds




class GridDataset(Dataset):
    """Get patches on a grid"""

    def __init__(
        self, 
        bounds, 
        vrt_path_base, 
        patch_size, 
        crop_size, 
        transform=None, 
        input_resolution=None, 
        target_resolution = None,
        aoi_gdf=None,
        reference_year= None,
        nb_timeseries_image =None,
        duplication_level_noise = None,
    ):
        s2_vrt_path = next((os.path.join(root, file) for root, _, files in os.walk(os.path.join(vrt_path_base, f"s2")) for file in files if file.endswith('.vrt')), None)

        with rasterio.open(s2_vrt_path) as src:
            transform_c = src.transform.c
            transform_f = src.transform.f
            if input_resolution is None:
                self.input_resolution = src.transform.a
            else:
                self.input_resolution = input_resolution

        if bounds is not None:
            bounds = adjust_bounds_to_1point5_res(transform_c, transform_f, bounds)

        self.bounds = bounds
        self.vrt_path_base = vrt_path_base
        self.patch_size = patch_size
        self.crop_size = crop_size  # equivalent to stride of 2*crop_size
        # tansform refers here to the transformation of the input, not the projection
        self.transform = transform
        self.target_resolution = target_resolution

        reference_date = {"2022" : "20220215", "2023": "20230915"}
        self.reference_date = reference_date[reference_year]
        self.nb_timeseries_image = nb_timeseries_image
        self.duplication_level_noise = duplication_level_noise

        if aoi_gdf is not None:
            bounds_list = list(aoi_gdf.bounds.itertuples(index=False, name=None))
            bounds_list = [
                adjust_bounds_to_1point5_res(transform_c, transform_f, list(bounds))
                for bounds in tqdm(bounds_list, desc="Processing bounds")
            ]
        else:
            bounds_list = [self.bounds]
        
        self.patch_bounds = []
        # Iterate through all geometries
        for bounds in tqdm(bounds_list, desc="Processing geometries"):
            self.patch_bounds.extend(
                get_grid(
                    patch_size,
                    crop_size,
                    global_bounds=bounds
                )
            )

    def __len__(self):
        return len(self.patch_bounds)

    def __getitem__(self, idx):        
        bounds = self.patch_bounds[idx]

        input = []
        input_date = []
        s1_vrt_list_path = os.path.join(self.vrt_path_base, f"s1/vrt_files")
        s2_vrt_list_path = os.path.join(self.vrt_path_base, f"s2/vrt_files")

        vrt_list = self.get_correct_vrt(bounds)   
        vrt_list = self.get_n_closest_dates(vrt_list)


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
        
        if self.transform:
            input = torch.stack([self.transform(image) for image in input], dim=0)
        else : 
            input = np.stack(input, axis=0).mean(axis=0)

        input_date = torch.from_numpy(np.stack(input_date, axis=0))
        bounds = torch.tensor(bounds)
        target_nb_pixel = int((bounds[3] - bounds[1])/self.target_resolution)

        meta_data = {"bounds" : bounds, "dates" : input_date, "target_nb_pixel" : target_nb_pixel}
        return input, meta_data

    
    def get_n_closest_dates(
        self,
        tuples_list,
        ) :
        """
        Returns the n tuples whose first date is closest to the reference date,
        sorted by ascending order of the first date in each tuple.
        """
        ref_date = datetime.strptime(self.reference_date, "%Y%m%d")

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

    def get_date_from_vrt_name(self, vrt_name) :
        return datetime.strptime(vrt_name.split('_')[1].split('.')[0], "%Y%m%d")
            
    def sort_by_proximity(self, target_file, file_list):
        target_date = self.get_date_from_vrt_name(target_file)
        return sorted(file_list, key=lambda file: abs((self.get_date_from_vrt_name(file) - target_date).days))

    def get_correct_vrt(self, bounds):

        list_correct_vrt = []

        s1_vrt_list_path = os.path.join(self.vrt_path_base, f"s1/vrt_files")
        s2_vrt_list_path = os.path.join(self.vrt_path_base, f"s2/vrt_files")
        s1_vrt_list = [file for file in os.listdir(s1_vrt_list_path) if file.endswith('.vrt')]
        s2_vrt_list = sorted([file for file in os.listdir(s2_vrt_list_path) if file.endswith('.vrt')])

        for s2_vrt in s2_vrt_list :
            try :
                s2_image = get_window(
                    image_path=os.path.join(s2_vrt_list_path, s2_vrt),
                    bounds=bounds,
                    )
            except Exception as e:
                logging.info(f"Problem with {os.path.join(s2_vrt_list_path, s2_vrt)} : {e}")
                s2_image = np.array([])

            if s2_image.size == 0 or not np.isfinite(s2_image).any() :
                continue

            sorted_s1_list = self.sort_by_proximity(s2_vrt, s1_vrt_list) #We are looking for the nearest tensor s1 in terms of date 
            for s1_vrt in sorted_s1_list :
                try:
                    s1_image = get_window(
                        image_path=os.path.join(s1_vrt_list_path, s1_vrt),  
                        bounds=bounds,
                        )
                except Exception as e:
                    logging.info(f"Problem with {os.path.join(s1_vrt_list_path, s1_vrt)} : {e}")
                    s1_image = np.array([])
                
                if len(s1_image) > 0  and np.isfinite(s1_image).any():
                    list_correct_vrt.append([s2_vrt, s1_vrt])
                    break
    
        return list_correct_vrt


