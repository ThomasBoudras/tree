# Dataset
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset
from tqdm import tqdm
import os

def get_grid(patch_size, crop_size, width, height, left=0, top=0):
    """Get a grid of patches, with a stride corresponding to patch_size-2*crop_size up and down
    If images from each plot on the grid are cropped on each side, then they can be concatenated.
    NB: starts at 0 from the top left. width and height are in number of pixels.
    max_row corresponds to "bottom".
    """

    pixel_patch_bounds = []
    col_start = left
    row_start = top
    max_row = top + height
    max_col = left + width

    while (row_start <= max_row) and (col_start <= max_col):
        row_stop = row_start + patch_size
        col_stop = col_start + patch_size
        # Save pixel_patch_bounds
        pixel_patch_bounds.append((col_start, col_stop, row_start, row_stop))

        if col_stop - 2 * crop_size <= max_col:
            col_start = col_stop - 2 * crop_size
        else:
            col_start = left
            row_start = row_stop - 2 * crop_size
    return pixel_patch_bounds


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
            self.pixel_patch_bounds = []
            # Iterate through all geometries
            for bounds in tqdm(bounds_list, desc="Processing geometries"):
                window = from_bounds(*bounds, src.transform)
                # Convert geographic bounds to pixel coordinates within the VRT
                left = (bounds[0] - src.transform.c) / self.resolution
                top = (src.transform.f - bounds[3]) / self.resolution
                # check these are integers to avoid shifts of one pixel due to rounding
                if ((left - np.floor(left)) > 0) or ((top - np.floor(top)) > 0):
                    raise ValueError("bounds lead to rounding issues")

                # Shift bounds to compensate cropping
                left -= crop_size
                top -= crop_size
                self.pixel_patch_bounds.extend(
                    get_grid(
                        patch_size,
                        crop_size,
                        window.width,
                        window.height,
                        left=left,
                        top=top,
                    )
                )

    def __len__(self):
        return len(self.pixel_patch_bounds)

    def __getitem__(self, idx):

        input = []
        for source in ["s2", "s1"] :
            source_vrt_path = os.path.join(self.vrt_path_base, source, f"{source}_EPSG2154.vrt")
            with rasterio.open(source_vrt_path) as src:
                window = rasterio.windows.Window(
                    col_off=self.pixel_patch_bounds[idx][0],
                    row_off=self.pixel_patch_bounds[idx][2],
                    width=self.patch_size,
                    height=self.patch_size,
                )

                input_source = src.read(window=window).transpose(1, 2, 0).astype(np.float32)
            
            input_source[np.isneginf(input_source)] = 0
            input.append(input_source)
        input = np.concatenate(input, axis=2)

        if self.transform:
            input = self.transform(input)

        return input
