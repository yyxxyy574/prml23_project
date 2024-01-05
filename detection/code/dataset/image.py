import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List
import imageio
from PIL import Image
import random
import cv2
import time
from skimage.util import random_noise

class ChannelOrder(Enum):
    RGB = "RGB"
    BGR = "BGR"
    
@dataclass
class PreprocessParams:
    """
    Image preprocessing parameters
    """
    channel_order: ChannelOrder
    scaling: float
    means: List[float]
    stds: List[float]
    
@dataclass
class AugmentParams:
    """
    data augment parameters
    """
    augment: bool
    horizontal_flip: bool
    vertical_flip: bool
    hsv: bool
    saturation: bool
    hue: bool
    contrast: bool
    brightness: bool
    add_noise: bool
    
def compute_scale_factor(original_width, original_height, min_dimension_pixels):
    if not min_dimension_pixels:
        return 1.0
    if original_width > original_height:
        scale_factor = min_dimension_pixels / original_height
    else:
        scale_factor = min_dimension_pixels / original_width
    return scale_factor

def preprocess_vgg16(image_data, preprocessing):
    
    if preprocessing.channel_order == ChannelOrder.RGB:
        pass                                        # already in RGB order
    elif preprocessing.channel_order == ChannelOrder.BGR:
        image_data = image_data[:, :, :: -1]         # RGB -> BGR
    else:
        raise ValueError("Invalid ChannelOrder value: %s" % str(preprocessing.channel_order))
    image_data[:, :, 0] *= preprocessing.scaling
    image_data[:, :, 1] *= preprocessing.scaling
    image_data[:, :, 2] *= preprocessing.scaling
    image_data[:, :, 0] = (image_data[:, :, 0] - preprocessing.means[0]) / preprocessing.stds[0]
    image_data[:, :, 1] = (image_data[:, :, 1] - preprocessing.means[1]) / preprocessing.stds[1]
    image_data[:, :, 2] = (image_data[:, :, 2] - preprocessing.means[2]) / preprocessing.stds[2]
    image_data = image_data.transpose([2, 0, 1]) # (height,width,3) -> (3,height,width)
    return image_data.copy() # copy required to eliminate negative stride

def load_image(path, preprocessing, augmenting, min_dimension_pixels=None):
    """
    Load, preprocess and augment an image:
    1. Standardizing image pixels to ImageNet dataset-level statistics
    2. Ensuring channel order matches what the model's backbone's feature extractor requires.
    3. Resizing the image so that the minimum dimension is a defined size, as recommended by Faster R-CNN.

    Args:
        path: str, path to load image from
        preprocessing: PreprocessParams, image pre-process parameters governing channel order and normalization
        augmenting: AugmentParams, image augment parameters
        min_dimension_pixels: int, if not None, specifies the size in pixels of the smaller side of the image, the other side is scaled proportionally

    Returns:
        image_data: np.ndarry, Image pixels as float32, shaped as (channels, height, width)
        image: PIL.Image, image after processing
        (image_data.shape[0], original_height, original_width): Tuple(int, int, int) scaling factor applied to the image dimensions and the original image shape
    """
    
    data = imageio.imread(path, pilmode = "RGB")
    image = Image.fromarray(data, mode = "RGB")
    original_width, original_height = image.width, image.height
    
    # Augementing with give parameters
    if augmenting.augment:
        if augmenting.horizontal_flip:
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        if augmenting.vertical_flip:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        if augmenting.hsv:
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
        if augmenting.saturation:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
            r = np.random.uniform(0.95, 1.05)
            data[..., 1] = np.clip(data[..., 1] * r, 0, 255)
            data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
            image = Image.fromarray(data, mode = "RGB")
        if augmenting.hue:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
            r = np.random.uniform(-0.05, 0.05)
            data[..., 0] = np.mod(data[..., 0] + r * 180, 180)
            data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
            image = Image.fromarray(data, mode = "RGB")
        if augmenting.contrast:
            mean = data.mean(axis=0).mean(axis=0)
            r = np.random.uniform(0.9, 1.1)
            data = np.clip((data- mean) * r + mean, 0, 255).astype(np.uint8)
            image = Image.fromarray(data, mode = "RGB")
        if augmenting.brightness:
            r = np.random.uniform(-0.1, 0.1)
            data = np.clip(data + r * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(data, mode = "RGB")
        if augmenting.add_noise:
            data = random_noise(data, mode="gaussian", seed=int(time.time()), clip=True) * 255
            image = Image.fromarray(data, mode = "RGB")
            
    # Resizing the image so that the minimum dimension is a defined size
    if min_dimension_pixels is not None:
        scale_factor = compute_scale_factor(original_width=image.width, original_height=image.height, min_dimension_pixels=min_dimension_pixels)
        width = int(image.width * scale_factor)
        height = int(image.height * scale_factor)
        image = image.resize((width, height), resample=Image.BILINEAR)
    else:
        scale_factor = 1.0
    
    # Get processed image information
    image_data = np.array(image).astype(np.float32)
    image_data = preprocess_vgg16(image_data=image_data, preprocessing=preprocessing)
    
    return image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)