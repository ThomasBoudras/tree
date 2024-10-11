import torch
import numpy as np
from src.utils.module_utils import get_vegetation_and_forest_mask
from src.utils.dataset_utils import BottomLeftCrop
import geopandas as gpd
import os

class OpenCanopyMetrics:
    def __init__(
        self,
        tree_cover_threshold = 2,
        forest_mask_path = "./",
        classification_path = "./",
        classes_to_keep = [5],
        target_patch_size = 512,
    ):
        super().__init__()
        self.tree_cover_threshold = tree_cover_threshold
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path)
        self.classification_path = classification_path
        self.classes_to_keep = classes_to_keep
        self.crop = BottomLeftCrop(patch_size=target_patch_size)

        
        
    def __call__(self, pred, target, meta_data):
        bounds_batch = meta_data["bounds"]
        year_batch = meta_data["years"]

        vegetation_mask = []
        for i, bounds in enumerate(bounds_batch) :
            year = year_batch[i]
            classification_path = os.path.join(self.classification_path, str(year), "lidar_classification.vrt")
            mask, _ = get_vegetation_and_forest_mask(
            forest_mask_gdf=self.forest_mask_gdf,
            classification_raster_path = classification_path,
            bounds = bounds,
            classes_to_keep=self.classes_to_keep,
            resolution = 1,
            resampling_method="bilinear",
            )
            mask = np.expand_dims(mask, axis=0)
            mask = self.crop(torch.from_numpy(mask)).numpy()
            vegetation_mask.append(torch.tensor(mask))

        vegetation_mask = torch.stack(vegetation_mask)

        pred = pred[vegetation_mask]
        target = target[vegetation_mask]

        #Mean_error
        mean_error = torch.mean(pred - target)
        # Absolute Error
        absolute_error = torch.abs(pred - target)
        # Mean Absolute Error (MAE)
        mae = torch.mean(absolute_error)
        # Standard Deviation of the Error
        std_e = torch.std(pred - target)
        # Standard Deviation of the Absolute Error
        std_ae = torch.std(absolute_error)
        # Normalized Mean Absolute Error (nMAE)
        nmae = torch.mean(absolute_error / (1 + torch.abs(target)))
        # Compute the RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        # Metrics for whiskers plot
        percentiles = torch.tensor([0, 25, 50, 75, 100]).to(target.device)
        percentiles_values = torch.quantile(pred - target, percentiles / 100)

        # Compute intersection and union for IoU, assuming binary classification
        threshold = torch.tensor(self.tree_cover_threshold).to(target.device)
        tree_cover_true = target >= threshold
        tree_cover_pred = pred >= threshold
        intersection = torch.logical_and(tree_cover_pred, tree_cover_true).sum()
        union = torch.logical_or(tree_cover_pred, tree_cover_true).sum()
        iou = intersection.float() / (union.float() + 1.0)
        
        self.res={}
        self.res["mean_absolute_error"] = mae.item()
        self.res["mean_error"] = mean_error.item()
        self.res["std_error"] = std_e.item()
        self.res["std_absolute_error"] = std_ae.item()
        self.res["root_mean_squared_error"] = rmse.item()
        self.res["normalized_mean_absolute_error"] = nmae.item()
        self.res["min_error"] = percentiles_values[0].item()
        self.res["max_error"] = percentiles_values[4].item()
        self.res["median_error"] = percentiles_values[2].item()
        self.res["first_quartile_error"] = percentiles_values[1].item()
        self.res["third_quartile_error"] = percentiles_values[3].item()
        self.res["tree_cover_iou"] = iou.item()

        return self.res
 