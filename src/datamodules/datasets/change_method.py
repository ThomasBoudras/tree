import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion

class threshold_method :
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, image_1, image_2):
        difference = image_2 - image_1
        return difference < self.threshold
    
class percentage_method :
    def __init__(self, threshold, min_height_tree):
        self.threshold =threshold
        self.min_height_tree = min_height_tree

    def __call__(self, image_1, image_2):
        changes =  (image_2 / (image_1 + 1e-6)) < self.threshold
        #avoid the noise of tree less than 2 meters
        changes[image_1 < self.min_height_tree] = False
        return changes


class tree_cover_method:
    def __init__(self, threshold):
        self.threshold =threshold

    def __call__(self, image_1, image_2):
        binary_map_1 = image_1 > self.threshold
        binary_map_2 = image_2 > self.threshold
        return torch.logical_and(binary_map_1, ~binary_map_2)
    

class difference_method:
    def __init__(self, min_difference, max_difference):
        self.min_difference = min_difference
        self.max_difference = max_difference

    def __call__(self, image_1, image_2):
        difference  = image_2 - image_1
        difference = torch.clip(difference, min=self.min_difference, max=self.max_difference)
        return difference