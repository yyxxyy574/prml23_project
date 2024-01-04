import numpy as np
import imageio
from PIL import Image
import random
import cv2
import xml.etree.ElementTree as ET
import time
from skimage.util import random_noise
import copy

import sys
sys.path.append("/home/stu7/prml23_project/detection/code")
import visualize

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
class_name_to_id = {class_name: class_id for (class_id, class_name) in class_id_to_name.items()}

def horizontal_flip(data, image, scored_bboxes_and_names):
    
    original_width, original_height = image.width, image.height
    
    image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    
    scored_bboxes_and_names_au = []
    for i in range(len(scored_bboxes_and_names)):
        scored_bbox_and_name = scored_bboxes_and_names[i]
        corners = scored_bbox_and_name[0: 4]
        
        corners = np.array([
            corners[0],
            original_width - 1 - corners[3],
            corners[2],
            original_width - 1 - corners[1]
        ])
        
        scored_bbox_and_name[0: 4] = corners
        
        scored_bboxes_and_names_au.append(scored_bbox_and_name)
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/horizontal_flip.jpg",
                              image=image,
                              scored_bboxes_and_names=scored_bboxes_and_names_au,
                              class_id_to_name=class_id_to_name)

def vertical_flip(data, image, scored_bboxes_and_names):
    
    original_width, original_height = image.width, image.height
    
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            
    scored_bboxes_and_names_au = []
    for i in range(len(scored_bboxes_and_names)):
        scored_bbox_and_name = scored_bboxes_and_names[i]
        corners = scored_bbox_and_name[0: 4]
        
        corners = np.array([
            original_height - 1 -corners[2],
            corners[1],
            original_height - 1 -corners[0],
            corners[3]
        ])
        
        scored_bbox_and_name[0: 4] = corners
        
        scored_bboxes_and_names_au.append(scored_bbox_and_name)
        
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/vertical_flip.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names_au,
                            class_id_to_name=class_id_to_name)

def hsv(data, image, scored_bboxes_and_names):
    # Generate random enhancement amplitude
    r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1
    # Change RGB to HSV
    hue, sat, val = cv2.split(cv2.cvtColor(data, cv2.COLOR_RGB2HSV))
    dtype = data.dtype
    # Generate look-table table
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    # Perform color enhancement
    data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # Change HSV to RGB
    data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
    image = Image.fromarray(data, mode = "RGB")
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/hsv.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)
    
def saturation(data, image, scored_bboxes_and_names):
    data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    r = np.random.uniform(0.95, 1.05)
    data[..., 1] = np.clip(data[..., 1] * r, 0, 255)
    data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
    image = Image.fromarray(data, mode = "RGB")
            
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/saturation.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)

def hue(data, image, scored_bboxes_and_names):
    data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    r = np.random.uniform(-0.05, 0.05)
    data[..., 0] = np.mod(data[..., 0] + r * 180, 180)
    data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
    image = Image.fromarray(data, mode = "RGB")
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/hue.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)

def contrast(data, image, scored_bboxes_and_names):
    
    mean = data.mean(axis=0).mean(axis=0)
    r = np.random.uniform(0.9, 1.1)
    data = np.clip((data- mean) * r + mean, 0, 255).astype(np.uint8)
    image = Image.fromarray(data, mode = "RGB")
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/contrast.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)
    
    
def brightness(data, image, scored_bboxes_and_names):
    
    r = np.random.uniform(-0.1, 0.1)
    data = np.clip(data + r * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(data, mode = "RGB")
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/brightness.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)
    
def add_noise(data, image, scored_bboxes_and_names):
    
    data = random_noise(data, mode="gaussian", seed=int(time.time()), clip=True) * 255
    image = Image.fromarray(data, mode = "RGB")
    
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/add_noise.jpg",
                            image=image,
                            scored_bboxes_and_names=scored_bboxes_and_names,
                            class_id_to_name=class_id_to_name)
    
if __name__ == "__main__":
    data = imageio.imread("/home/stu7/prml23_project/augment_test/2007_000559.jpg", pilmode = "RGB")
    image = Image.fromarray(data, mode = "RGB")
    
    # Use the ElementTree to parse XML annotations
    tree = ET.parse("/home/stu7/prml23_project/augment_test/2007_000559.xml")
    root = tree.getroot()
    
    # Extract bbox and class information
    scored_bboxes_and_names = []
    for obj in root.findall("object"):
        # Extract class information
        class_name = obj.find("name").text
        class_id = class_name_to_id[class_name]
        
        # Extract bounding box information
        bndbox = obj.find("bndbox")
        # Convert to 0-based pixel coordinates
        x_min = int(bndbox.find("xmin").text) - 1
        y_min = int(bndbox.find("ymin").text) - 1
        x_max = int(bndbox.find("xmax").text) - 1
        y_max = int(bndbox.find("ymax").text) - 1
        corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
        # Score
        corners = np.append(corners, 1)
        # ID
        corners = np.append(corners, class_id)
        
        scored_bboxes_and_names.append(corners)
       
    visualize.draw_gt_detections(save_path="/home/stu7/prml23_project/augment_test/no_augment.jpg",
                                image=image,
                                scored_bboxes_and_names=scored_bboxes_and_names,
                                class_id_to_name=class_id_to_name)
    horizontal_flip(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    vertical_flip(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    hsv(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    saturation(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    hue(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    contrast(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    brightness(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))
    # add_noise(data=copy.deepcopy(data), image=copy.deepcopy(image), scored_bboxes_and_names=copy.deepcopy(scored_bboxes_and_names))