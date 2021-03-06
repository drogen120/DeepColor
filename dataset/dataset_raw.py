import os
import math
import random
import numpy as np
from PIL import Image
from Queue import Queue
from threading import Thread
import tifffile as tiff 

from dataset import DataSet

class RawDataSet(DataSet):

    def __init__(self, common_params, dataset_params):
        self.data_path = str(dataset_params['path'])
        self.width = int(common_params['image_width'])
        self.height = int(common_params['image_height'])
        self.batch_size = int(common_params['batch_size'])
        self.thread_num = int(dataset_params['thread_num'])

        self.record_queue = Queue(maxsize=1000)
        self.image_inout_queue = Queue(maxsize=30)

        self.record_list = []
        self.listfile_name = "filelist.txt"
        self.input_folder = "input/"
        self.output_folder = "output/"

        with open(self.data_path + self.listfile_name, 'r') as filelist_file:
            for line in filelist_file:
                line = line.strip()
                self.record_list.append(line)

        self.record_index = 0
        self.record_numbers = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_numbers / self.batch_size)

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        while True:
            if self.record_index % self.record_numbers == 0:
                random.shuffle(self.record_list)
                self.record_index = 0
            self.record_queue.put(self.record_list[self.record_index])
            self.record_index += 1

    def record_process(self, record):
        input_image = Image.open(self.data_path + self.input_folder + record)
        output_image = Image.open(self.data_path + self.output_folder + record)
        # output_image = tiff.imread(self.data_path + self.output_folder + record) 

        input_image = input_image.resize((self.width, self.height))
        output_image = output_image.resize((self.width, self.height))

        input_image_array = np.array(input_image)
        # print np.amax(input_image_array)
        output_image_array = np.array(output_image)
        # print np.amax(output_image_array)

        return [input_image_array, output_image_array]

    def record_customer(self):
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_inout_queue.put(out)

    def batch(self):
        input_images_list = []
        output_images_list = []
        for i in range(self.batch_size):
            [input_image, output_image] = self.image_inout_queue.get()
            # print input_image.shape
            input_images_list.append(input_image)
            # print output_image.shape
            output_images_list.append(output_image)
        input_images = np.asarray(input_images_list, dtype=np.float32)
        input_images = input_images/255.0
        output_images = np.asarray(output_images_list, dtype=np.float32)
        output_images = output_images/255.0
        return input_images, output_images
