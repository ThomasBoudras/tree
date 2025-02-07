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
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path) if forest_mask_path is not None else None
        self.classification_path = classification_path
        self.classes_to_keep = classes_to_keep
        self.crop = BottomLeftCrop(patch_size=target_patch_size)
        self.bins = bins
        self.accumulated_metrics = {}

    def update(self, pred, target, meta_data):
        
        
       
        if self.classification_path is not None:
            bounds_batch = meta_data["bounds"]
            year_batch = meta_data["years"]
            vegetation_mask = []
            for i, bounds in enumerate(bounds_batch):
                year = year_batch[i].item()
                classification_path = os.path.join(self.classification_path, str(year), "lidar_classification.vrt")
                mask, _ = get_vegetation_and_forest_mask(
                    forest_mask_gdf=self.forest_mask_gdf,
                    classification_raster_path=classification_path,
                    bounds=bounds.tolist(),
                    classes_to_keep=self.classes_to_keep,
                    resolution=1,
                    resampling_method="bilinear",
                )
                mask = torch.from_numpy(np.expand_dims(mask, axis=0)).to(target.device)
                mask = self.crop(mask)
                vegetation_mask.append(mask)
        else : 
            vegetation_mask = [torch.ones_like(pred[0], dtype=torch.bool) for _ in range(len(pred))]
        
        vegetation_mask = torch.stack(vegetation_mask).to(target.device)
        nan_mask = torch.isnan(target)

        if self.bins is not None :    
            bin_indices_target = torch.from_numpy(np.digitize(target.cpu().numpy(), self.bins)).to(target.device)
            
            for i in range(1, len(self.bins)):
                bin_name = f"{self.bins[i-1]}-{self.bins[i]}"
                bin_mask = (bin_indices_target == i) & vegetation_mask & ~nan_mask
                self.metric_calculator.update(pred.to(target.device), target.to(target.device), mask=bin_mask, mask_MAE=None, accumulated_metrics = self.accumulated_metrics, bin_name=bin_name)

            full_mask = (bin_indices_target != 0) & (bin_indices_target != len(self.bins)) & vegetation_mask
            full_mask_MAE = (bin_indices_target > 1) & (bin_indices_target != len(self.bins)) & vegetation_mask
            self.metric_calculator.update(pred.to(target.device), target.to(target.device), mask=full_mask, mask_MAE=full_mask_MAE, accumulated_metrics = self.accumulated_metrics, bin_name = "full")
        else :
            full_mask = vegetation_mask & ~nan_mask
            self.metric_calculator.update(pred.to(target.device), target.to(target.device), mask=full_mask, mask_MAE=None, accumulated_metrics = self.accumulated_metrics, bin_name = "full")

    def compute(self):
        final_results = {}
        if self.bins is not None :
            for i in range(1, len(self.bins)):
                bin_name = f"{self.bins[i-1]}-{self.bins[i]}"
                self.metric_calculator.final_compute(self.accumulated_metrics, final_results, bin_name)
        
        self.metric_calculator.final_compute(self.accumulated_metrics, final_results, "full")
        
        return final_results

    def reset(self):
        self.accumulated_metrics = {}



class height_metrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, tree_cover_threshold, bins):
        self.tree_cover_threshold =  tree_cover_threshold
        self.bins = bins
        self.metrics_accumulated = ["sum_error", "sum_absolute_error", "sum_relative_error", "sum_squared_error", "intersection", "union", "nb_values"]
        
    def update(self, pred, target, mask, accumulated_metrics, bin_name, mask_MAE=None) :

        if not bin_name in accumulated_metrics :
            accumulated_metrics[bin_name] = {}
        for metric in self.metrics_accumulated :
            if not metric in accumulated_metrics[bin_name] :
                accumulated_metrics[bin_name][metric] = 0
              
        masked_target = target[mask]
        masked_pred = pred[mask]
        if len(masked_pred) > 0 :
            error = masked_pred - masked_target
            accumulated_metrics[bin_name][f"sum_error"] += torch.sum(error)      
            accumulated_metrics[bin_name][f"sum_squared_error"] += torch.sum(error ** 2)
            
            absolute_error = torch.abs(error)
            accumulated_metrics[bin_name][f"sum_absolute_error"] += torch.sum(absolute_error)
            
            if bin_name != "full" :
                accumulated_metrics[bin_name][f"sum_relative_error"] += torch.sum(absolute_error / (1 + torch.abs(masked_target)))
            else :
                error_mae = pred[mask_MAE] - target[mask_MAE]
                accumulated_metrics[bin_name][f"sum_relative_error"] += torch.sum(torch.abs(error_mae) / (1 + torch.abs(target[mask_MAE])))

            threshold = torch.tensor(self.tree_cover_threshold).to(target.device)
            tree_cover_true = target >= threshold
            tree_cover_pred = pred >= threshold
            accumulated_metrics[bin_name][f"intersection"] += torch.logical_and(tree_cover_pred, tree_cover_true).sum()
            accumulated_metrics[bin_name][f"union"] += torch.logical_or(tree_cover_pred, tree_cover_true).sum()

            accumulated_metrics[bin_name][f"nb_values"] += len(masked_pred) 

    def final_compute(self, accumulated_metrics, final_results, bin_name):
        
        final_results[bin_name] = {}
        sum_absolute_error = accumulated_metrics[bin_name]["sum_absolute_error"]
        sum_error = accumulated_metrics[bin_name]["sum_error"]
        sum_squared_error = accumulated_metrics[bin_name]["sum_squared_error"]
        sum_relative_error = accumulated_metrics[bin_name]["sum_relative_error"]
        nb_values = accumulated_metrics[bin_name]["nb_values"]
        intersection = accumulated_metrics[bin_name]["intersection"]
        union = accumulated_metrics[bin_name]["union"]

        if nb_values > 0 :
            # MAE - Mean Absolute Error
            mae = sum_absolute_error / nb_values
            final_results[bin_name][f"MAE"] = mae

            # std_ae - Standard Absolute Error
            final_results[bin_name][f"std_ae"] = torch.sqrt((sum_squared_error / nb_values) - mae ** 2)

            # RMSE - Root Mean Square Error
            final_results[bin_name][f"RMSE"] = torch.sqrt(sum_squared_error / nb_values)

            #Bias - Mean Error
            bias = sum_error / nb_values
            final_results[bin_name][f"Bias"] = bias

            # std_e - Standard Error
            final_results[bin_name][f"std_e"] = torch.sqrt((sum_squared_error / nb_values) - bias ** 2)


            # nMAE - Normalized MAE 
            if bin_name != "full" :
                final_results[bin_name][f"nMAE"] = sum_relative_error / nb_values
            else :
                final_results[bin_name][f"nMAE"] = sum_relative_error / (nb_values - accumulated_metrics[f"{self.bins[0]}-{self.bins[1]}"]["nb_values"])

            # TreeCov - Treecover IoU
            final_results[bin_name][f"TreeCov"] = intersection / (union + 1)

            final_results[bin_name]["nb_values"] = nb_values
        else :
            final_results[bin_name][f"MAE"] = torch.nan
            final_results[bin_name][f"std_ae"] = torch.nan
            final_results[bin_name][f"RMSE"] = torch.nan
            final_results[bin_name][f"Bias"] = torch.nan
            final_results[bin_name][f"std_e"] = torch.nan
            final_results[bin_name][f"nMAE"] = torch.nan
            final_results[bin_name][f"TreeCov"] = torch.nan
            final_results[bin_name]["nb_values"] = 0
 


class change_metrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, threshold_percentage):
        self.threshold_percentage =  threshold_percentage
        self.metrics_accumulated = ["continuous_intersection", "continuous_pred_sum", "discrete_intersection", "discrete_pred_sum", "target_sum", "sum_bce_per_element", "nb_values"]
        
    def update(self, pred, target, mask, accumulated_metrics, bin_name, mask_MAE=None) :

        if not bin_name in accumulated_metrics :
            accumulated_metrics[bin_name] = {}
        for metric in self.metrics_accumulated :
            if not metric in accumulated_metrics[bin_name] :
                accumulated_metrics[bin_name][metric] = 0
              
        masked_target = target[mask].float()
        masked_pred = pred[mask]
        if len(masked_pred) > 0 :

            accumulated_metrics[bin_name][f"continuous_intersection"] += (masked_pred * masked_target).sum()  # TP
            accumulated_metrics[bin_name][f"continuous_pred_sum"] += masked_pred.sum()  # TP + FP
            
            discrete_masked_pred  = (masked_pred > self.threshold_percentage).float()  
            accumulated_metrics[bin_name][f"discrete_intersection"] += (discrete_masked_pred * masked_target).sum()  # TP
            accumulated_metrics[bin_name][f"discrete_pred_sum"] += discrete_masked_pred.sum()

            accumulated_metrics[bin_name][f"target_sum"] += masked_target.sum()  # TP + FN
            

            log_pred = torch.log(masked_pred)               
            log_1_minus_pred = torch.log(1 - masked_pred)    
            bce_per_element = - (masked_target * log_pred + (1 - masked_target) * log_1_minus_pred)
            accumulated_metrics[bin_name][f"sum_bce_per_element"] += bce_per_element.sum()

            accumulated_metrics[bin_name][f"nb_values"] += len(masked_pred) 



    def final_compute(self, accumulated_metrics, final_results, bin_name):
    
        final_results[bin_name] = {}
        continuous_intersection = accumulated_metrics[bin_name]["continuous_intersection"]
        continuous_pred_sum = accumulated_metrics[bin_name]["continuous_pred_sum"]
        discrete_intersection = accumulated_metrics[bin_name]["discrete_intersection"]
        discrete_pred_sum = accumulated_metrics[bin_name]["discrete_pred_sum"]
        target_sum = accumulated_metrics[bin_name]["target_sum"]
        sum_bce_per_element = accumulated_metrics[bin_name]["sum_bce_per_element"]
        nb_values = accumulated_metrics[bin_name]["nb_values"]


        if nb_values > 0 :
            final_results[bin_name][f"continuous_recall"] = continuous_intersection / target_sum if target_sum > 0 else torch.nan 
            final_results[bin_name][f"continuous_precision"] = continuous_intersection/continuous_pred_sum if continuous_pred_sum > 0 else torch.nan 
            final_results[bin_name][f"continuous_f1_score"] = 2*continuous_intersection/(continuous_pred_sum+target_sum) if continuous_pred_sum+target_sum > 0 else torch.nan
            

            final_results[bin_name][f"discrete_recall"] = discrete_intersection/target_sum if target_sum > 0 else torch.nan 
            final_results[bin_name][f"discrete_precision"] = discrete_intersection /discrete_pred_sum if discrete_pred_sum > 0 else torch.nan 
            final_results[bin_name][f"discrete_f1_score"] = 2*discrete_intersection/(discrete_pred_sum+target_sum) if discrete_pred_sum+target_sum > 0 else torch.nan
            
            final_results[bin_name]["bce_loss"] = sum_bce_per_element / nb_values 

            final_results[bin_name]["nb_values"] = nb_values


        else :
            final_results[bin_name][f"continuous_recall"] = torch.nan
            final_results[bin_name][f"continuous_precision"] = torch.nan
            final_results[bin_name][f"continuous_f1_score"] = torch.nan
            final_results[bin_name][f"discrete_recall"] = torch.nan
            final_results[bin_name][f"discrete_precision"] = torch.nan
            final_results[bin_name][f"discrete_f1_score"] = torch.nan
            final_results[bin_name][f"bce_loss"] = torch.nan
            final_results[bin_name]["nb_values"] = 0





class difference_metrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, delta_change):
        self.delta_change = delta_change
        self.metrics_accumulated = ["sum_error", "sum_absolute_error", "sum_squared_error", "change_intersection", "change_pred_sum", "change_target_sum", "nb_values"]
        
    def update(self, pred, target, mask, accumulated_metrics, bin_name, mask_MAE=None) :

        if not bin_name in accumulated_metrics :
            accumulated_metrics[bin_name] = {}
        for metric in self.metrics_accumulated :
            if not metric in accumulated_metrics[bin_name] :
                accumulated_metrics[bin_name][metric] = 0
              
        masked_target = target[mask]
        masked_pred = pred[mask]
        if len(masked_pred) > 0 :
            error = masked_pred - masked_target
            accumulated_metrics[bin_name][f"sum_error"] += torch.sum(error)      
            accumulated_metrics[bin_name][f"sum_squared_error"] += torch.sum(error ** 2)
            
            absolute_error = torch.abs(error)
            accumulated_metrics[bin_name][f"sum_absolute_error"] += torch.sum(absolute_error)
            
            changes_pred = masked_pred < self.delta_change
            changes_target = masked_target < self.delta_change
            accumulated_metrics[bin_name][f"change_intersection"] += (changes_pred * changes_target).sum()  # TP
            accumulated_metrics[bin_name][f"change_pred_sum"] += changes_pred.sum()  # TP + FP
            accumulated_metrics[bin_name][f"change_target_sum"] += changes_target.sum()  # TP + FN
            accumulated_metrics[bin_name][f"nb_values"] += len(masked_pred) 

    def final_compute(self, accumulated_metrics, final_results, bin_name):
        
        final_results[bin_name] = {}
        sum_error = accumulated_metrics[bin_name]["sum_error"]
        sum_squared_error = accumulated_metrics[bin_name]["sum_squared_error"]
        sum_absolute_error = accumulated_metrics[bin_name]["sum_absolute_error"]
        change_intersection = accumulated_metrics[bin_name]["change_intersection"]
        change_pred_sum = accumulated_metrics[bin_name]["change_pred_sum"]
        change_target_sum = accumulated_metrics[bin_name]["change_target_sum"]
        nb_values = accumulated_metrics[bin_name]["nb_values"]

        if nb_values > 0 :
            # MAE - Mean Absolute Error
            mae = sum_absolute_error / nb_values
            final_results[bin_name][f"MAE"] = mae

            # std_ae - Standard Absolute Error
            final_results[bin_name][f"std_ae"] = torch.sqrt((sum_squared_error / nb_values) - mae ** 2)

            # RMSE - Root Mean Square Error
            final_results[bin_name][f"RMSE"] = torch.sqrt(sum_squared_error / nb_values)

            #Bias - Mean Error
            bias = sum_error / nb_values
            final_results[bin_name][f"Bias"] = bias

            # std_e - Standard Error
            final_results[bin_name][f"std_e"] = torch.sqrt((sum_squared_error / nb_values) - bias ** 2)

            final_results[bin_name][f"recall"] = change_intersection / change_target_sum if change_target_sum > 0 else torch.nan 
            final_results[bin_name][f"precision"] = change_intersection/change_pred_sum if change_pred_sum > 0 else torch.nan 
            final_results[bin_name][f"f1_score"] = 2*change_intersection/(change_pred_sum + change_target_sum) if change_pred_sum + change_target_sum > 0 else torch.nan


            final_results[bin_name]["nb_values"] = nb_values
        else :
            final_results[bin_name][f"MAE"] = torch.nan
            final_results[bin_name][f"std_ae"] = torch.nan
            final_results[bin_name][f"RMSE"] = torch.nan
            final_results[bin_name][f"Bias"] = torch.nan
            final_results[bin_name][f"std_e"] = torch.nan
            final_results[bin_name][f"recall"] = torch.nan
            final_results[bin_name][f"precision"] = torch.nan
            final_results[bin_name][f"f1_score"] = torch.nan
            final_results[bin_name]["nb_values"] = 0
 