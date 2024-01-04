# Dataloader for VOC dataset

import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List
from typing import Tuple
from PIL import Image

from . import image
from ..model import anchor

@dataclass
class Bbox:
    class_id: int
    class_name: str
    corners: np.ndarray
    
    def __repr__(self):
        return "[class=%s (%f,%f,%f,%f)]" % (self.class_name, self.corners[0], self.corners[1], self.corners[2], self.corners[3])

    def __str__(self):
        return repr(self)

@dataclass
class TrainingSample:
    anchors: np.ndarray
    anchors_valid_map: np.ndarray
    gt_rpn_map: np.ndarray                
    gt_rpn_object_indices: List[Tuple[int, int, int]] 
    gt_rpn_background_indices: List[Tuple[int, int, int]] 
    gt_bboxes: List[Bbox]               
    image_data: np.ndarray                
    image: Image                  
    filepath: str                      

class VOCDataset:
    """
    A VOC dataset iterator for a particular split (train, val)
    """
    
    # Classification information
    class_num = 21 # 20 classification classes + 1 background
    class_id_to_name = {
        0:  "background",
        1:  "aeroplane",
        2:  "bicycle",
        3:  "bird",
        4:  "boat",
        5:  "bottle",
        6:  "bus",
        7:  "car",
        8:  "cat",
        9:  "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor"
    }
    
    def __init__(self, split, image_preprocess_params, get_backbone_shape, feature_pixels=16, dir="data", augment=True, shuffle=True):
        """
        Args:
            split: str, datast split to load (train, val)
            image_preprocess_params: dataset.image.PreprocessParams, image preporcessing parameters to apply when loading images
            get_backbone_shape: function to compute feature map shape from input image shape
            feature_pixels: int, distance in pixels between anchors 
            dir: str, root directory of dataset
            augment: bool, whether to randomly agument (horizontally) flip images during iteraction with 50% probability
            shuffle: bool, whether to shuffle the dataset each time it is iterated
        """
        
        if not os.path.exists(dir):
            raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
        self._split = split
        self._dir = dir
        self._class_id_to_name = VOCDataset.class_id_to_name
        self._class_name_to_id = {class_name: class_id for (class_id, class_name) in self.class_id_to_name.items()}
        self._class_num = VOCDataset.class_num
        self._image_paths = self._get_image_paths()
        self._sample_num = len(self._image_paths)
        self._gt_bboxes_by_image_path = self._get_bboxes(image_paths=self._image_paths)
        self._i = 0
        self._iterable_filepaths = self._image_paths.copy()
        self._image_preprocess_params = image_preprocess_params
        self._get_backbone_shape = get_backbone_shape
        self._feature_pixels = feature_pixels
        self._augment = augment
        self._shuffle = shuffle
        
    def __iter__(self):
        
        self._i = 0
        if self._shuffle:
            random.shuffle(self._iterable_filepaths)
        return self
    
    def __next__(self):
        
        if self._i >= len(self._iterable_filepaths):
            raise StopIteration
        
        # Next file to load
        filepath = self._iterable_filepaths[self._i]
        self._i += 1
        
        # Augment
        is_augment = random.randint(0, 1) != 0 if self._augment else 0
        if is_augment:
            augment_params = image.AugmentParams(
                augment = True,
                horizontal_flip = random.randint(0, 1) != 0,
                vertical_flip = random.randint(0, 1) != 0,
                hsv = random.randint(0, 1) != 0,
                saturation = random.randint(0, 1) != 0,
                hue = random.randint(0, 1) != 0,
                contrast = random.randint(0, 1) != 0,
                brightness = random.randint(0, 1) != 0,
                # add_noise = random.randint(0, 1) != 0
                add_noise = False
            )
        else:
            augment_params = image.AugmentParams(
                augment = False,
                horizontal_flip = False,
                vertical_flip = False,
                hsv = False,
                saturation = False,
                hue = False,
                contrast = False,
                brightness = False,
                add_noise = False
            )
        
        # Load next training sample
        return self._generate_training_sample(filepath=filepath, augment_params=augment_params)
    
    def _generate_training_sample(self, filepath, augment_params):
        
        # Load and preprocess the image
        scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(path=filepath, preprocessing=self._image_preprocess_params, augmenting=augment_params, min_dimension_pixels=600)
        _, original_height, original_width = original_shape
        
        # Scale ground truth bboxes to new image size
        scaled_gt_bboxes = []
        for bbox in self._gt_bboxes_by_image_path[filepath]:
            
            corners = bbox.corners
            if augment_params.augment:
                if augment_params.horizontal_flip:
                    corners = np.array([
                    corners[0],
                    original_width - 1 - corners[3],
                    corners[2],
                    original_width - 1 - corners[1]
                    ])
                if augment_params.vertical_flip:
                    corners = np.array([
                        original_height - 1 -corners[2],
                        corners[1],
                        original_height - 1 -corners[0],
                        corners[3]
                    ])
                    
            scaled_bbox = Bbox(
                class_id=bbox.class_id,
                class_name=bbox.class_name,
                corners=corners * scale_factor
            )
            scaled_gt_bboxes.append(scaled_bbox)

        # Generate anchor maps and RPN truth map
        anchors, anchors_vaild_map = anchor.generate_anchors(image_shape=scaled_image_data.shape,
                                                             feature_map_shape=self._get_backbone_shape(scaled_image_data.shape),
                                                             feature_pixels=self._feature_pixels)
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchor.generate_rpn_map(anchors=anchors,
                                                                                               anchors_vaild_map=anchors_vaild_map,
                                                                                               gt_bboxes = scaled_gt_bboxes)

        # Return sample
        return TrainingSample(
            anchors=anchors,
            anchors_valid_map=anchors_vaild_map,
            gt_rpn_map=gt_rpn_map,
            gt_rpn_object_indices=gt_rpn_object_indices,
            gt_rpn_background_indices=gt_rpn_background_indices,
            gt_bboxes=scaled_gt_bboxes,
            image_data=scaled_image_data,
            image=scaled_image,
            filepath=filepath
        )
        
    def _get_image_paths(self):
        
        image_list_file = os.path.join(self._dir, self._split + ".txt")
        with open(image_list_file) as fp:
            ids = [line.strip() for line in fp.readlines()]
        image_paths = [os.path.join(self._dir, "JPEGImages", id) + ".jpg" for id in ids]
        return image_paths
    
    def _get_bboxes(self, image_paths):
        
        gt_bboxes_by_image_path = {}
        
        for image_path in image_paths:
            id = os.path.splitext(os.path.basename(image_path))[0]
            annotation_path = os.path.join(self._dir, "Annotations", id) + ".xml"
            
            # Use the ElementTree to parse XML annotations
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            # Extract bbox and class information
            bboxes = []
            for obj in root.findall("object"):
                # Extract class information
                class_name = obj.find("name").text
                class_id = self._class_name_to_id[class_name]
                
                # Extract bounding box information
                bndbox = obj.find("bndbox")
                # Convert to 0-based pixel coordinates
                x_min = int(bndbox.find("xmin").text) - 1
                y_min = int(bndbox.find("ymin").text) - 1
                x_max = int(bndbox.find("xmax").text) - 1
                y_max = int(bndbox.find("ymax").text) - 1
                corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
                
                # Save bbox and class information
                bbox = Bbox(class_id=class_id, class_name=class_name, corners=corners)
                bboxes.append(bbox)
        
            gt_bboxes_by_image_path[image_path] = bboxes
        
        return gt_bboxes_by_image_path