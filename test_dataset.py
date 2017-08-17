import tensorflow as tf
import numpy as np
import os
from dataset.dataset_raw import RawDataSet

dataset_params = {
"path" : "./training_data/",
"thread_num" : 1,
}
common_params = {
"image_width" : 512,
"image_height" : 512,
"batch_size" : 2,
}

while (True):
    dataset = RawDataSet(common_params, dataset_params)
    input_images, output_images = dataset.batch()
    print input_images.shape
    break
