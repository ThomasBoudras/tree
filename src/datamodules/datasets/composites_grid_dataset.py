# Dataset
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from src.utils.dataset_utils import get_window

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
        resolution=None, 
        aoi_gdf=None
    ):
        s2_vrt_path = os.path.join(vrt_path_base, f"s2/s2_EPSG2154.vrt")
        with rasterio.open(s2_vrt_path) as src:
            transform_c = src.transform.c
            transform_f = src.transform.f
        if bounds is not None:
            bounds = adjust_bounds_to_1point5_res(transform_c, transform_f, bounds)
        self.bounds = bounds
        self.vrt_path_base = vrt_path_base
        self.patch_size = patch_size
        self.crop_size = crop_size  # equivalent to stride of 2*crop_size
        # tansform refers here to the transformation of the input, not the projection
        self.transform = transform
        with rasterio.open(s2_vrt_path) as src:
            if resolution is None:
                self.resolution = src.transform.a
            else:
                self.resolution = resolution

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
        for source in ["s2", "s1"] :
            source_vrt_path = os.path.join(self.vrt_path_base, source, f"{source}_EPSG2154.vrt")
            
            input_source = get_window(source_vrt_path, bounds=bounds, resolution=self.resolution)
            input_source = input_source.astype(np.float32).transpose(1, 2, 0)
            input_source[np.isneginf(input_source)] = 0
            input.append(input_source)
        input = np.concatenate(input, axis=2)

        if self.transform:
            input = self.transform(input)
        
        meta_data =  {}
        return input, meta_data
