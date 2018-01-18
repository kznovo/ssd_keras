import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300,300,3)
class_names = ["background", "stop_signs"];
NUM_CLASSES = len(class_names)
model = SSD(input_shape, num_classes=NUM_CLASSES)
model.load_weights('../checkpoints/weights.20-3.64.hdf5') 
vid_test = VideoTest(class_names, model, input_shape)
vid_test.run('videoplayback.mp4')
