import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
import geopandas as gpd

from src.utils.dataset_utils import get_window


def get_vegetation_and_forest_mask(
    forest_mask_gdf,
    classification_raster_path,
    bounds,
    classes_to_keep,
    resolution=1,
    resampling_method="bilinear",
):
    raster_bounds = box(*bounds)
    classification, profile = get_window(
        classification_raster_path,
        geometry=None,
        bounds=bounds,
        resolution=resolution,
        return_profile=True,
        resampling_method=resampling_method,
    )
    classification = classification.squeeze()
    # Create a mask for pixels with value 5 in the classification raster
    classif_mask = classification == classes_to_keep[0]
    if len(classes_to_keep) > 1:
        for aclass in classes_to_keep[1::]:
            classif_mask = classif_mask | (classification == aclass)
    
    clipped_gdf = gpd.clip(forest_mask_gdf, raster_bounds)     
    
    geometries = [(geom, 1) for geom in clipped_gdf.geometry]
    if len(geometries):
        mask_geometries = rasterize(
            geometries,
            out_shape=classification.shape,
            transform=profile["transform"],
            fill=0,
            default_value=1,
            dtype=np.uint8,
        ).astype(bool)
    else:
        mask_geometries = np.zeros_like(classif_mask, dtype=bool)

    # Combine the two masks
    final_mask = classif_mask | mask_geometries
    return final_mask, profile
