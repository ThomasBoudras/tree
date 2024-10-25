import rasterio
import numpy as np
import os
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from tqdm import tqdm
from shapely.geometry import mapping
import random
import rasterio
from rasterio.windows import from_bounds

def main(geometries_path, 
         year1,
         year2,
         folders_original_tif_year1,
         folders_original_tif_year2,
         path_output_dataset, 
         split_percentage, 
         nb_samples_limit, 
         ):
    
    
    original_tif_list_year1 = [
        os.path.join(folders_original_tif_year1, file)
        for file in os.listdir(folders_original_tif_year1)
        if file.endswith('.tif')]
    
    original_tif_list_year2 = [
        os.path.join(folders_original_tif_year2, file)
        for file in os.listdir(folders_original_tif_year2)
        if file.endswith('.tif')]



    geometries =  gpd.read_file(geometries_path)
    path_inter = path_output_dataset.split('.')[0] + "_inter.geojson"
    initialize_geojson(path_inter)
    
    for row, row_gdf in tqdm(geometries.iterrows()) : 
        bounds = row_gdf["geometry"].bounds
        lidar_file_name_year1 = get_lidar_date(year=year1, initial_bounds=bounds, tif_list=original_tif_list_year1)
        lidar_file_name_year2 = get_lidar_date(year=year2, initial_bounds=bounds, tif_list=original_tif_list_year2)

        tif_containing_box.split('_')[-1].split(".")[0]
        
        if lidar_file_name_year1 and lidar_file_name_year2 :
            split = row_gdf["split"]
            add_feature_to_geojson(
                path_inter,
                year1,
                year2,
                lidar_file_name_year1,
                lidar_file_name_year2,
                split,
                bounds
                )

        if nb_samples_limit and row > nb_samples_limit:
            break
    
    close_geojson(path_inter)

    geometries = gpd.read_file(path_inter)

    nb_samples = len(geometries)
    n_train, n_val, n_test = (int(p * nb_samples) for p in split_percentage)
    n_train += int(nb_samples - sum((n_train, n_val, n_test)))

    split_list = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    random.shuffle(split_list)
    
    for idx, feature in tqdm(geometries.iterrows()) :
        split = split_list[idx]
        lidar_file_name_year1, lidar_file_name_year2, bounds = feature[f"lidar_file_name_{year1}"], feature[f"lidar_file_name_{year2}"], feature["geometry"].bounds
        add_feature_to_geojson(
            path_output_dataset,
            year1,
            year2,
            lidar_file_name_year1,
            lidar_file_name_year2,
            split,
            bounds
            )

    print(f"GeoJSON dataset créé et sauvegardé dans {path_output_dataset}")



def check_all_zeros_in_bounds(tif_path, bounds):
    # Ouvrir le fichier TIFF
    with rasterio.open(tif_path) as dataset:
        # Extraire la fenêtre de lecture à partir des bounds
        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], dataset.transform)

        # Lire les données dans la fenêtre définie par les bounds
        data = dataset.read(1, window=window)  # On suppose que la bande 1 est utilisée ici

        # Vérifier si tous les pixels sont à 0
        if (data == 0).all():
            return True
        else:
            return False


def get_lidar_date(year, initial_bounds, tif_list):
    # Créer un polygone "box" pour les bounds initiaux
    initial_box = box(*initial_bounds)  # bounds sous la forme (xmin, ymin, xmax, ymax)

    tif_containing_box = None

    for tif_path in tif_list:
        with rasterio.open(tif_path) as tif:
            bounds = tif.bounds
            tif_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            if tif_box.contains(initial_box):
                window = from_bounds(initial_bounds[0], initial_bounds[1], initial_bounds[2], initial_bounds[3], tif.transform)

                # Lire les données dans la fenêtre définie par les bounds
                data = tif.read(1, window=window)  # On suppose que la bande 1 est utilisée ici

                if (data != 0).any() :
                    tif_containing_box = tif_path
                    break
    
    return tif_containing_box


def create_polygon(bounds):
    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    polygon = [
        [xmin, ymin],  # coin inférieur gauche
        [xmin, ymax],  # coin supérieur gauche
        [xmax, ymax],  # coin supérieur droit
        [xmax, ymin],  # coin inférieur droit
        [xmin, ymin]   # retour au point de départ
    ]
    return Polygon(polygon)

def create_feature(year1, year2, lidar_file_name_year1, lidar_file_name_year2, split, bounds):
    polygon = create_polygon(bounds)
    return {
        "type": "Feature",
        "properties": {
            f"lidar_file_name_{year1}": lidar_file_name_year1,
            f"lidar_file_name_{year2}": lidar_file_name_year2,
            "split": split
        },
        "geometry": mapping(polygon)  # Conversion du polygone en format GeoJSON compatible
        }


def initialize_geojson(path):
    geojson_header = '''
{
"type": "FeatureCollection", 
"crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
"features": []
}
'''
    # Écrire directement la chaîne de caractères formatée dans le fichier
    with open(path, 'w') as f:
        f.write(geojson_header)

# Fonction pour ajouter une feature au GeoJSON ligne par ligne
def add_feature_to_geojson(
    path,
    year1,
    year2,
    lidar_file_name_year1,
    lidar_file_name_year2,
    split,
    bounds,
    ):

    feature = create_feature(
        year1,
        year2,
        lidar_file_name_year1,
        lidar_file_name_year2,
        split,
        bounds,
        )

    with open(path, 'r+') as f:
        # Aller à la fin du fichier pour ajouter la nouvelle feature
        f.seek(0, 2)  # Déplacer le curseur à la fin du fichier
        
        # Reculer de deux caractères pour enlever "]}\n" et ajouter une virgule
        f.seek(f.tell() - 4, 0)
        f.write('\n')  # Ajouter une virgule pour séparer les features

        # Écrire la nouvelle feature
        json.dump(feature, f)
        f.write(',\n]\n}')  # Terminer la structure du GeoJSON


def close_geojson(path):
    with open(path, 'r+') as f:
        f.seek(0, 2)  # Déplacer le curseur à la fin du fichier
        
        # Reculer de deux caractères pour enlever "]}\n" et ajouter une virgule
        f.seek(f.tell() - 5, 0)
        f.write(' ')  # supprimer la dernière virgule pour séparer les features

if __name__ == "__main__":
    
    # path_output_dataset= "/home/projects/ku_00196/people/thobou/data/utils/geometries_inter_2021_2022.geojson"
    path_output_dataset= "/home/thom_bdrs/travail/copenhague/data/utils/geometries_inter_2021_2022.geojson"

    initialize_geojson(path_output_dataset)
    # main(
    #     geometries_path="/home/projects/ku_00196/people/thobou/data/utils/geometries.geojson",
    #     year1=2021,
    #     year2=2022,
    #     folders_original_tif_year1="/home/projects/ku_00196/people/thobou/data/lidar/2021/lidar",
    #     folders_original_tif_year2= "/home/projects/ku_00196/people/thobou/data/lidar/2022/lidar",
    #     path_output_dataset=path_output_dataset,
    #     split_percentage=[0.8,0.1,0.1],
    #     nb_samples_limit= None,
    # )
    main(
        geometries_path="/home/thom_bdrs/travail/copenhague/data/utils/geometries.geojson",
        year1=2021,
        year2=2022,
        folders_original_tif_year1="/home/thom_bdrs/travail/copenhague/data/lidar/lidar_2021",
        folders_original_tif_year2= "/home/thom_bdrs/travail/copenhague/data/lidar/lidar_2022",
        path_output_dataset=path_output_dataset,
        split_percentage=[0.8,0.1,0.1],
        nb_samples_limit= 3000,
    )
    close_geojson(path_output_dataset)

