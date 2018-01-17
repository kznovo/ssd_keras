import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300,300,3)
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)
model = SSD(input_shape, num_classes=NUM_CLASSES)
model.load_weights('../weights_SSD300.hdf5') 
vid_test = VideoTest(class_names, model, input_shape)
vid_test.run('test.mp4')
