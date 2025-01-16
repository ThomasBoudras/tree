
import os
import shutil
from datetime import datetime

import geopandas as gpd
import hydra
import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal
from pyproj import Transformer
from pytorch_lightning import Trainer
from shapely.geometry import box
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import glob
from rasterio.transform import from_bounds

from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)


# from torchvision import transforms
from tqdm import tqdm



def predict_on_aoi(config: DictConfig) -> None:
   
    from src.utils import utils

    log = utils.get_logger(__name__)

    save_dir = os.path.expanduser(config.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config.aoi_name is not None:
        aoi_name = config.aoi_name
    else:
        if config.aoi_bounds is not None:
            aoi_name = (
                str(config.aoi_bounds[0])
                + "_"
                + str(config.aoi_bounds[1])
                + "_"
                + str(config.aoi_bounds[2])
                + "_"
                + str(config.aoi_bounds[3])
            )
        else:
            raise ValueError("An aoi_name must be provided if aoi_bounds are null.")

    if (config.aoi_bounds is not None) and (config.aoi_path is not None):
        raise ValueError("Cannot provide both aoi_bounds and aoi_path, one of them must be null.")

    if config.run_year is not None:
        # FIXME why does this take out "_" put "" in config
        run_year = str(config.run_year)
    else:
        run_year = "test"

    predictions_dir = os.path.join(save_dir, aoi_name, run_year)
    predictions_dir_tmp = os.path.join(save_dir, aoi_name, run_year, "tmp")
    predictions_dir_data = os.path.join(save_dir, aoi_name, run_year, "data")

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if not os.path.exists(predictions_dir_tmp):
        os.makedirs(predictions_dir_tmp)

    if not os.path.exists(predictions_dir_data):
        os.makedirs(predictions_dir_data)

    # save the config next to the data
    OmegaConf.save(config, os.path.join(predictions_dir, "predict_on_aoi_config.yaml"))

    if config.aoi_bounds is not None:
        # Reproject bounds to 2154 if necessary
        if config.aoi_crs == 4326:
            transformer = Transformer.from_crs(config.aoi_crs, "epsg:2154")
            # Transform the min and max points of the bounds
            # FIXME this only works for 4326 input, inversion lon/lat and x/y in transform?
            min_point_2154 = transformer.transform(config.aoi_bounds[1], config.aoi_bounds[0])
            max_point_2154 = transformer.transform(config.aoi_bounds[3], config.aoi_bounds[2])
            bounds = [
                int(min_point_2154[0]),
                int(min_point_2154[1]),
                int(max_point_2154[0]),
                int(max_point_2154[1]),
            ]
        elif config.aoi_crs == 2154:
            # Round
            bounds = list(np.array(config.aoi_bounds).astype(int))
        else:
            raise NotImplementedError(
                f"aoi crs {config.aoi_crs} is not supported. Only crs 4326 and 2154 are supported."
            )
        aoi_gdf = None
    else:
        # Apply prediction on each geometry of the provided geodataframe
        aoi_path = config.aoi_path
        if aoi_path.endswith("parquet"):
            aoi_gdf = gpd.read_parquet(aoi_path)
        else:
            aoi_gdf = gpd.read_file(aoi_path)

        aoi_gdf = aoi_gdf.query("(split==@config.split) & (lidar_year==@config.lidar_year)")
        # Save it for reproducibility
        aoi_gdf.to_file(
            os.path.join(predictions_dir, "geometries.geojson"),
            driver="GeoJSON",
        )

        bounds = None


    # Init lightning model
    log.info(f"Instantiating model <{config.model.instance._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model.instance)
    model.predictions_save_dir = predictions_dir_tmp

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger)


    # Train the model
    if config.get("ckpt_path"):
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            ckpt_path = None
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None

    
    # For future versions hardcode the normalization constatns in the config
    normalization_constants_path = os.path.expanduser(config.normalization_constants_path)
    input_mean = np.load(os.path.join(normalization_constants_path, "mean.npy"))
    input_std = np.load(os.path.join(normalization_constants_path, "std.npy"))


    transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Normalize(input_mean, input_std),
        ]
    )

    # Adjust bounds in order to have exact match between coordinates and pixels
    # eg (bounds[0]-src.transform.c) should be a multiple of 3 for resolution 1.5, done in GridDataset

    # Preprocess aoi_gdf to merge geometries that form rectangles
    # !!!WARNING!!! this results in a potentially larger predict area than demanded,
    # as we're taking bounds of adjacent geometries to form new (larger) geometries
    # (geometries may be adjacent but not form a full rectangle)

    if aoi_gdf is not None:
        # Step 1: Dissolve all geometries into a single MultiPolygon
        dissolved = aoi_gdf.unary_union
        # Step 2: Create a new list of rectangles from the dissolved geometries
        merged_polygons = []

        if dissolved.geom_type == "Polygon":
            # If it's a single Polygon, create a bounding box
            minx, miny, maxx, maxy = dissolved.bounds
            new_box = box(minx, miny, maxx, maxy)
            if dissolved.equals(new_box):
                merged_polygons.append(new_box)
        elif dissolved.geom_type == "MultiPolygon":
            # If it's a MultiPolygon, iterate through each Polygon
            for geom in dissolved.geoms:
                if geom.is_valid:
                    minx, miny, maxx, maxy = geom.bounds
                    new_box = box(minx, miny, maxx, maxy)
                    merged_polygons.append(new_box)
                else:
                    for part in geom:
                        minx, miny, maxx, maxy = part.bounds
                        new_box = box(minx, miny, maxx, maxy)
                        merged_polygons.append(new_box)

        # Step 3: Convert merged polygons back into a GeoDataFrame
        aoi_gdf = gpd.GeoDataFrame(geometry=merged_polygons)

    # Predict then recombine on each geometry
    if bounds is not None:
        n_geom = 1
    else:
        n_geom = aoi_gdf.shape[0]

    for i in tqdm(range(n_geom), desc="Predicting on each aggegrated tile"):
        if aoi_gdf is None:
            row_gdf = None
        else:
            row_gdf = aoi_gdf.iloc[[i]]

        log.info(f"Instantiating dataset <{config.dataset._target_}>")
        grid_dataset = hydra.utils.instantiate(config.dataset, bounds = bounds, transform=transform, aoi_gdf=row_gdf)

        bounds = grid_dataset.bounds

        predict_dataloader = DataLoader(
            grid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )

        # Predict
        trainer.predict(model, predict_dataloader)

        # Transform npy predictions to tif files
        patch_bounds = grid_dataset.patch_bounds
        # FIXME handle non int bounds for aoi

        predictions_files = [
            os.path.join(predictions_dir_tmp, x)
            for x in os.listdir(predictions_dir_tmp)
            if x.endswith(".npy")
        ]

        # XXX The following could be parallelized / included in the prediction step (especially for small patch size),
        # to make it faster
        # XXX gdal can handle a limited number of files to concat (need to increase file limit / it is slow)
        # Create a callback called after predict to reconcat predictions -> need to save params for that
        for pred_file in tqdm(predictions_files, desc="Cropping and saving predictions for each batch"):
            # Create tmp dir for each batch
            predictions_dir_tmp_batch = os.path.join(predictions_dir_tmp, "tmp")
            os.makedirs(predictions_dir_tmp_batch, exist_ok=True)
            # Each pred_file stores one batch of predictions
            batch_idx = int(os.path.basename(pred_file).split(".")[0])
            batch_pred = np.load(pred_file)
            for ix, pred in enumerate(batch_pred):
                ix_grid = config.batch_size * batch_idx + ix
                sample_bounds = patch_bounds[ix_grid]
                output_path = os.path.join(predictions_dir_tmp_batch, str(ix_grid) + ".tif")
                
                pixel_crop_size = int(config.crop_size / config.target_resolution)

                sample_bounds = (
                    sample_bounds[0] + config.crop_size,
                    sample_bounds[1] + config.crop_size,
                    sample_bounds[2] - config.crop_size,
                    sample_bounds[3] - config.crop_size
                    )
                
                print(f"initial pred {pred}")
                # clip negative values to 0
                pred = np.clip(pred, a_min=0, a_max=None)
                print(f"2 pred {pred}")
                pred = pred.squeeze().astype(np.uint16)
                print(f"3 pred {pred}")
                pred = pred[pixel_crop_size : -pixel_crop_size, pixel_crop_size : -pixel_crop_size]
                print(f"4 pred {pred}")

                height, width = pred.shape

                transform = from_bounds(*sample_bounds, width, height)
                print(f"bounds {sample_bounds}")

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=pred.dtype,
                    crs="EPSG:2154",  # Remplacer par le CRS approprié si nécessaire
                    transform=transform,
                ) as dst:
                    dst.write(pred, 1)
                        
            # XXX to increase speed of the second concat, would be faster to save without compression at this stage
            print(f"jpg saved in {os.path.join(predictions_dir_tmp, f'{config.run_name}_{config.run_year}_{batch_idx}_pred.jp2')}")
            concat_tif_to_jp2(
                predictions_dir_tmp_batch,
                os.path.join(predictions_dir_tmp, f"{config.run_name}_{config.run_year}_{batch_idx}_pred.jp2"),
                pattern=".tif",
            )
            # Remove npy file
            os.remove(pred_file)
            shutil.rmtree(predictions_dir_tmp_batch)

        concat_tif_to_jp2(
            predictions_dir_tmp, os.path.join(predictions_dir_data, f"{config.run_name}_{config.run_year}_{i}_pred.jp2"), pattern=".jp2"
        )

        # remove old predictions_dir
        shutil.rmtree(predictions_dir_tmp)
        os.makedirs(predictions_dir_tmp)

    # Create vrt with predictions
    vrt_path = os.path.join(predictions_dir_data, f"{config.run_name}_{config.run_year}_pred.vrt")
    files_list = [
        os.path.join(predictions_dir_data, x) for x in os.listdir(predictions_dir_data) if x.endswith(".jp2")
    ]
    create_virtual_dataset(files_list, vrt_path)

    if bounds is not None:
        # Translate to bounds (the grid might be bigger than bounds)
        # The projWin parameter specifies the spatial extent in the format (minX, maxY, maxX, minY)
        aoi_vrt_path = os.path.join(predictions_dir, f"{config.run_name}_{config.run_year}_pred_aoi.vrt")
        ds = gdal.Translate(aoi_vrt_path, vrt_path, projWin=(bounds[0], bounds[3], bounds[2], bounds[1]))
        # Close the dataset to flush to disk
        with rasterio.open(aoi_vrt_path, "r") as src:
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)
            print("Transform:", src.transform)
        ds = None
        if ds is None:
            print(f"Prediction vrt dataset successfully created within bounds at {aoi_vrt_path}")
        return bounds



def concat_tif_to_jp2(folder_path, output_path, pattern=".jp2"):
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", "10000")

    # Create one tif from all files
    # Step 1: List all input TIFF files
    input_tif_files = glob.glob(folder_path + "/*" + pattern)
    # Step 2: Open the input files using gdal.Open
    input_datasets = [gdal.Open(tif_file) for tif_file in input_tif_files]
    # Step 3: Create a virtual raster to merge the images
    vrt = gdal.BuildVRT(os.path.join(folder_path, "merged.vrt"), input_datasets)
    # Step 4: Use gdal.Warp to write the virtual raster to a new TIFF file
    # Define the options for the JPEG2000 driver
    options = [
        "QUALITY=100",  # Maximum quality for JPEG2000
    ]
    gdal.Warp(output_path, vrt, format="JPEG", creationOptions=options)

    # Close the input datasets
    for dataset in input_datasets:
        dataset = None
    if dataset:  # just to avoid linter error, dataset must be used somewhere
        print("")

    vrt = None


def create_virtual_dataset(files_list, output_path):
    # Create a VRT
    dataset = gdal.BuildVRT(output_path, files_list)
    # Save the VRT to a file
    dataset.FlushCache()  # Ensure all data is written
    dataset = None  # Close the dataset
    print(f"Virtual dataset successfully created at {output_path}")
