import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from datetime import datetime, timedelta
import torchvision.transforms.functional as F
from torch.utils.data import Sampler

def get_window(
    image_path,
    geometry=None,
    bounds=None,
    resolution=None,
    return_profile=False,
    resampling_method=None,
):
    """Retrieve a window from an image, within given bounds or within the bounds of a geometry"""
    with rasterio.open(image_path) as src:
        profile = src.profile
        if bounds is None:
            if geometry is not None:
                bounds = geometry.bounds
            else:
                bounds = src.bounds

        window = from_bounds(*bounds, transform=src.transform)
        transform = src.window_transform(window)

        init_resolution = profile["transform"].a

        if (resolution is not None) and (init_resolution != resolution):
            if resampling_method == "max_pooling" or (resampling_method is None and init_resolution < resolution):
                # Downsample with max pooling
                data = src.read(window=window)
                # Calculate the target shape
                target_width = int((bounds[2] - bounds[0]) / resolution)
                target_height = int((bounds[3] - bounds[1]) / resolution)

                # Resize the data to the target shape, using max pooling within each block
                def max_pooling_resize(image, target_width, target_height):
                    output = np.zeros((image.shape[0], target_height, target_width))
                    scale_x = image.shape[2] / target_width
                    scale_y = image.shape[1] / target_height
                    for i in range(target_height):
                        for j in range(target_width):
                            # XXX could replace int by rounding
                            x_start = int(j * scale_x)
                            x_end = int((j + 1) * scale_x)
                            y_start = int(i * scale_y)
                            y_end = int((i + 1) * scale_y)
                            output[:, i, j] = np.max(image[:, y_start:y_end, x_start:x_end])
                    return output

                data = max_pooling_resize(data, target_width, target_height)
            elif resampling_method == "bilinear" or (resampling_method is None and init_resolution > resolution):
                scale_factor = init_resolution / resolution
                data = src.read(
                    out_shape=(
                        src.count,
                        int(np.round(window.height * scale_factor, 0)),
                        int(np.round(window.width * scale_factor, 0)),
                    ),
                    resampling=Resampling.bilinear,
                    window=window,
                )
            else:
                raise ValueError(f"{resampling_method} is not a valid resampling method")

            # Update the transform for the new resolution
            transform = Affine(
                resolution,
                transform.b,
                transform.c,
                transform.d,
                -resolution,
                transform.f,
            )
        else:
            data = src.read(window=window)

    if return_profile:
        new_profile = profile.copy()
        if len(data.shape) == 3:
            count = data.shape[0]
            height = data.shape[1]
            width = data.shape[2]
        else:
            count = 1
            height = data.shape[0]
            width = data.shape[1]
        new_profile.update(
            {
                "transform": transform,
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": count,
            }
        )
        return data, new_profile
    else:
        return data
    

class BottomLeftCrop:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        return F.crop(img, 0, 0, self.patch_size, self.patch_size)
        

def found_nearest_date(date, min_year, max_year):
    current_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    
    possible_final_dates = {
        2021 : ["05-01", "07-01", "09-01", "11-01"],
        2022 : ["01-01", "03-01", "05-01", "07-01", "09-01", "11-01"],
        2023 : ["01-01", "03-01", "05-01"],
        }
    
    final_date = None
    final_year = None
    min_deltatime = timedelta.max  

    for year in range(min_year, max_year + 1):
        for possible_date in possible_final_dates[year]:
            possible_final_date = datetime.strptime(f"{year}-{possible_date}", "%Y-%m-%d")
            delta_time = abs(current_date - possible_final_date)
            if delta_time < min_deltatime:
                final_date = possible_date
                final_year = year
                min_deltatime = delta_time

    return final_date, final_year
    

class SubsetSampler(Sampler):
    def __init__(self, data_source, num_samples, shuffle):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle :
            indices = np.random.choice(len(self.data_source), self.num_samples, replace=False)
        else :
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self):
        return self.num_samples
