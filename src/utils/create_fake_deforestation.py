import torch
from dataset_utils import get_window
import rasterio
from rasterio.transform import from_bounds

import rasterio
import numpy as np
tile_shape = 1000
half_deforested_shape = 20
max_size_tree = 10
tif_path = "/home/thom_bdrs/travail/copenhague/data/lidar/lidar_2021/compressed_lidar_20210405374464.tif"


def main():
    # Open the TIF file and calculate bounds
    with rasterio.open(tif_path) as tif:
        tif_bounds = tif.bounds
        # Calculate real bounds correctly
        real_bounds = (
            tif_bounds.left + 10000,
            tif_bounds.bottom,
            tif_bounds.left + 10000 + tile_shape,
            tif_bounds.bottom + tile_shape
        )

    # Extract tensor data for the given bounds
    tif_tensor = get_window(tif_path, bounds=real_bounds).astype(np.float32)  # Expecting shape (1, H, W)
    tif_tensor = torch.from_numpy(tif_tensor)
    # Filter indices where the value is greater than 50
    indices = torch.nonzero(tif_tensor > 50, as_tuple=False)

    # If there are points found, choose a random one
    if len(indices) > 0:
        while True:
            # Select a random index from the valid indices
            random_index = torch.randint(len(indices), (1,)).item()
            _, x, y= indices[random_index]

            # Retrieve the value from the (1, H, W) tensor
            valeur_point = tif_tensor[0, x, y]
            print(f"Selected point coordinates: {(x, y)}, value: {valeur_point}")
            
            # Clamping coordinates for sub-region extraction
            x_min = torch.clamp(x - half_deforested_shape, min=0)
            x_max = torch.clamp(x + half_deforested_shape, max=tif_tensor.shape[1])
            y_min = torch.clamp(y - half_deforested_shape, min=0)
            y_max = torch.clamp(y + half_deforested_shape, max=tif_tensor.shape[2])

            # Extract the zone tensor around the point (x, y)
            zone_tensor = tif_tensor[0, x_min:x_max, y_min:y_max]
            # Get the top 10 values in the flattened zone and their coordinates
            top_values, top_indices = torch.topk(zone_tensor.reshape(-1), 10)
            top_coords = torch.stack([top_indices // zone_tensor.size(1), top_indices % zone_tensor.size(1)], dim=1)
            center_of_mass = top_coords.float().mean(dim=0)

            # Calculate the Euclidean distance of each point to the center of mass
            distances = torch.norm(top_coords.float() - center_of_mass, dim=1)

            # Find the closest point to the center of mass
            central_point_index = torch.argmin(distances)
            central_point = top_coords[central_point_index]
            print(f"cime : valeur {zone_tensor[central_point[0], central_point[1]] }, coord {central_point}")

            # Define directions for border point search
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            border_points = []

            for direction in directions:
                # Ensure `search_min` is defined to get a border point in the given direction
                border_point = search_min(zone_tensor, central_point, direction)
                if border_point:
                    border_points.append(border_point)
                else:
                    break
            print(f"Nombre de points ok : {len(border_points)}")
            if len(border_points) == 8:
                # Clear the triangular region between border points and the center point
                for i, border_point in enumerate(border_points):
                    a = border_point
                    b = border_points[i % 8 - 1]
                    zone_tensor = points_dans_triangle(a, b, central_point,zone_tensor, 1000)
                    
                break
            

            # Update the tif_tensor with modified zone_tensor
            tif_tensor[0, x_min:x_max, y_min:y_max]= zone_tensor

    else : 
        raise Exception("No non-zero value ")
    

    output_path = tif_path.split('.')[0] + "_modified.tif"
    save_modified_tif(tif_path, tif_tensor,output_path, bounds=real_bounds)
    print(f"Fake data save in {output_path}")
        
    
def search_min(tensor, point_origin, direction):

    x, y = point_origin

    x_limit, y_limit = tensor.shape

    # Parcourir jusqu'à un maximum de 5 pixels ou atteindre la limite du tensor
    for _ in range(1, max_size_tree):
    
        x = x + direction[0]
        y = y + direction[1]
        print(f"x {x}, y {y}, direction {direction}")
        #sortie de la fenêtre
        if (x >= x_limit) or x < 0 or y < 0 or y >= y_limit:
            return None

        # Condition 1 : rencontrer un pixel de valeur 0
        if tensor[x, y] < 50:
            return (x, y)

        if y + direction[1] < y_limit and y + direction[1] >= 0 and x + direction[0] < x_limit and x + direction[0] >= 0 :
            prev_value = tensor[x- direction[0], y - direction[1]]
            current_value = tensor[x, y]
            next_value = tensor[x + direction[0], y + direction[1]]
            if current_value < prev_value and current_value < next_value:
                return (x,y)
    return None


def points_dans_triangle(p1, p2, p3, tensor, valeur):
    points = np.array([p1, p2, p3])

    sorted_points = points[points[:, 1].argsort()]

    direction_a = (sorted_points[1,0] - sorted_points[0,0])/(sorted_points[1,1] - sorted_points[0,1])
    direction_b = (sorted_points[2,0] - sorted_points[0,0])/(sorted_points[2,1] - sorted_points[0,1])
    for i in range(sorted_points[0, 1], sorted_points[1, 1] + 1):
        limit_a = int(direction_a * (i -sorted_points[0, 1]) + sorted_points[0,0])
        limit_b = int(direction_b * (i- sorted_points[0, 1]) + sorted_points[0,0])
        for j in range(min(limit_a, limit_b) , max(limit_a, limit_b)) :
            tensor[j, i] = valeur
    

    direction_a = (sorted_points[2,0] - sorted_points[1,0])/(sorted_points[2,1] - sorted_points[1,1])
    for i in range(sorted_points[1, 1], sorted_points[2, 1] + 1):
        limit_a = int(direction_a * (i -sorted_points[1, 1]) + sorted_points[1,0])
        limit_b = int(direction_b * (sorted_points[0,1] - sorted_points[1,1] + i - sorted_points[1, 1]) + sorted_points[0,0])
        for j in range(min(limit_a, limit_b) , max(limit_a, limit_b)) :
            tensor[j, i] = valeur

    return tensor



def save_modified_tif(tif_path, modified_tensor, output_path, bounds):
    # Open the original TIF to get metadata
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()  # Copy metadata from the original file
    
    # Convert the tensor to a NumPy array compatible with rasterio
    numpy_array = modified_tensor.numpy().astype('float32')  # Ensure it’s compatible with rasterio

    # Update metadata to match our tensor's shape and data type
    meta.update({
        "count": numpy_array.shape[0],  # Number of bands
        "height": numpy_array.shape[1],
        "width": numpy_array.shape[2],
        "dtype": 'float32',  # dtype should match rasterio’s accepted formats
        "transform": from_bounds(*bounds, numpy_array.shape[2], numpy_array.shape[1])  # Apply bounds
    })
    
    # Write the modified tensor as a TIF file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(numpy_array)  # Write NumPy array, not the tensor directly


if __name__ == "__main__":
    main()
