import torch
import numpy as np
from src.utils.module_utils import get_vegetation_and_forest_mask
from src.utils.dataset_utils import BottomLeftCrop
import geopandas as gpd
import os
from torchmetrics import Metric


class masked_method_metrics(Metric):
    def __init__(
        self,
        metric_calculator,
        forest_mask_path="./",
        classification_path="./",
        classes_to_keep=[5],
        target_patch_size=512,
        bins=[]
    ):
        super().__init__()
        self.metric_calculator = metric_calculator
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path)
        self.classification_path = classification_path
        self.classes_to_keep = classes_to_keep
        self.crop = BottomLeftCrop(patch_size=target_patch_size)
        self.bins = bins
        self.accumulated_metrics = {}

    def update(self, pred, target, meta_data):
        bounds_batch = meta_data["bounds"]
        year_batch = meta_data["years"]

        vegetation_mask = []
        for i, bounds in enumerate(bounds_batch):
            year = year_batch[i]
            classification_path = os.path.join(self.classification_path, str(year), "lidar_classification.vrt")
            mask, _ = get_vegetation_and_forest_mask(
                forest_mask_gdf=self.forest_mask_gdf,
                classification_raster_path=classification_path,
                bounds=bounds,
                classes_to_keep=self.classes_to_keep,
                resolution=1,
                resampling_method="bilinear",
            )
            mask = torch.from_numpy(np.expand_dims(mask, axis=0)).to(target.device)
            mask = self.crop(mask)
            vegetation_mask.append(mask)

        vegetation_mask = torch.stack(vegetation_mask).to(target.device)
        bin_indices_target = torch.from_numpy(np.digitize(target.cpu().numpy(), self.bins)).to(target.device)
        
        for i in range(1, len(self.bins)):
            bin_mask = (bin_indices_target == i) & vegetation_mask
            self.metric_calculator.update(pred.to(target.device), target.to(target.device), bin_mask, self.accumulated_metrics, i)

        full_mask = (bin_indices_target != 0) & (bin_indices_target != len(self.bins)) & vegetation_mask
        self.metric_calculator.update(pred.to(target.device), target.to(target.device), full_mask, self.accumulated_metrics, "full")

    def compute(self):
        final_results = {}
        for bin in range(1, len(self.bins)):
            self.metric_calculator.final_compute(self.accumulated_metrics, final_results, bin)
        self.metric_calculator.final_compute(self.accumulated_metrics, final_results, "full")
        return final_results

    def reset(self):
        self.accumulated_metrics = {}





class classic_metrics :
    def __init__(self, tree_cover_threshold):
        self.tree_cover_threshold =  tree_cover_threshold
        self.metrics_accumulated = ["sum_error", "sum_absolute_error", "sum_relative_error", "sum_squarred_error", "intersection", "union", "nb_values"]
   
        
    def update(self, pred, target, mask, accumulated_metrics, bin) :

        if not bin in accumulated_metrics :
            accumulated_metrics[bin] = {}
        for metric in self.metrics_accumulated :
            if not metric in accumulated_metrics :
                accumulated_metrics[bin][metric] = 0
              
        masked_target = target[mask]
        masked_pred = pred[mask]
        if len(masked_pred) > 0 :
            error = masked_pred - masked_target
            accumulated_metrics[bin][f"sum_error"] += torch.sum(error)      
            accumulated_metrics[bin][f"sum_squarred_error"] += torch.sum(error ** 2)
            
            absolute_error = torch.abs(error)
            accumulated_metrics[bin][f"sum_absolute_error"] += torch.sum(absolute_error)
            accumulated_metrics[bin][f"sum_relative_error"] += torch.sum(absolute_error / (1 + torch.abs(masked_target)))
        
            threshold = torch.tensor(self.tree_cover_threshold).to(target.device)
            tree_cover_true = target >= threshold
            tree_cover_pred = pred >= threshold
            accumulated_metrics[bin][f"intersection"] += torch.logical_and(tree_cover_pred, tree_cover_true).sum()
            accumulated_metrics[bin][f"union"] += torch.logical_or(tree_cover_pred, tree_cover_true).sum()

            accumulated_metrics[bin][f"nb_values"] += len(masked_pred) 


            try :
                # Metrics for whiskers plot
                percentiles = torch.tensor([0, 25, 50, 75, 100]).to(masked_target.device)
                percentiles_values = torch.quantile(masked_pred - masked_target, percentiles / 100)
            except Exception as e :
                print(f"metrics_ type : {bin} target {masked_target}, pred {masked_pred}")
                raise e
        

    def final_compute(self, accumulated_metrics, final_results, bin):
        
        final_results[bin] = {}
        
        sum_absolute_error = accumulated_metrics[bin]["sum_absolute_error"]
        sum_error = accumulated_metrics[bin]["sum_error"]
        sum_squared_error = accumulated_metrics[bin]["sum_squarred_error"]
        sum_relative_error = accumulated_metrics[bin]["sum_relative_error"]
        nb_values = accumulated_metrics[bin]["nb_values"]
        intersection = accumulated_metrics[bin]["intersection"]
        union = accumulated_metrics[bin]["union"]

        if nb_values > 0 :
            # MAE - Mean Absolute Error
            mae = sum_absolute_error / nb_values
            final_results[bin][f"MAE"] = mae

            # std_ae - Standard Absolute Error
            final_results[bin][f"std_ae"] = torch.sqrt((sum_squared_error / nb_values) - mae ** 2)

            # RMSE - Root Mean Square Error
            final_results[bin][f"RMSE"] = torch.sqrt(sum_squared_error / nb_values)

            #Bias - Mean Error
            bias = sum_error / nb_values
            final_results[bin][f"Bias"] = bias

            # std_e - Standard Error
            final_results[bin][f"std_e"] = torch.sqrt((sum_squared_error / nb_values) - bias ** 2)


            # nMAE - Normalized MAE 
            final_results[bin][f"nMAE"] = sum_relative_error / nb_values

            # TreeCov - Treecover IoU
            final_results[bin][f"TreeCov"] = intersection / (union + 1)

            final_results[bin]["nb_values"] = nb_values
        else :
            final_results[bin][f"MAE"] = torch.nan
            final_results[bin][f"std_ae"] = torch.nan
            final_results[bin][f"RMSE"] = torch.nan
            final_results[bin][f"Bias"] = torch.nan
            final_results[bin][f"std_e"] = torch.nan
            final_results[bin][f"nMAE"] = torch.nan
            final_results[bin][f"TreeCov"] = torch.nan
            final_results[bin]["nb_values"] = 0

 
class OpenCanopyMetrics:
    def __init__(self, *args, **kwargs):
        pass