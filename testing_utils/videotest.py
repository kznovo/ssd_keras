""" A class for testing a SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import datetime
import subprocess
import pickle
import numpy as np
import pandas as pd
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer

import sys
sys.path.append("..")
from ssd_utils import BBoxUtility


class VideoTest(object):
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
        
        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col) 
        
    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0.6):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        # Compute aspect ratio of video     
        vidw = vid.get(3)
        vidh = vid.get(4)
        vidar = vidw/vidh
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(0, start_frame)
            
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        
        
        # time as name constant
        now = datetime.datetime.now()
        time_as_name = now.strftime("%Y%m%d%H%M%S")
        
        # csv output
        item_1, item_2, item_3 = ('car', 'bus', 'person')
        df_tmp = pd.DataFrame(columns=('frame','object','coord'))
        csv_res = pd.DataFrame(columns=
                               ('count_'+item_1,
                                'coordinates_'+item_1,
                                'count_'+item_2,
                                'coordinates_'+item_2,
                                'count_'+item_3,
                                'coordinates_'+item_3))
        csv_res.index.name = 'frame'
        csv_name = 'output_csv/'+time_as_name+'.csv'
        csv_res.to_csv(csv_name)
        
        # video output
        output_shape = (int(self.input_shape[0]*vidar),self.input_shape[1])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_video/.tmp.avi', fourcc, int(round(vid.get(5))), output_shape)
        
        ## hatta code above ##
        
            
        while True:
            ## for ctrl c handling this code was modified ##
            try:
                    retval, orig_image = vid.read()
                    if not retval:
                        print("Done!")
                        return
                    
                    cur_f = vid.get(1)
                    
                    im_size = (self.input_shape[0], self.input_shape[1])    
                    resized = cv2.resize(orig_image, im_size)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    to_draw = cv2.resize(resized, output_shape)
                    inputs = [image.img_to_array(rgb)]
                    tmp_inp = np.array(inputs)
                    x = preprocess_input(tmp_inp)
                    y = self.model.predict(x)
                    results = self.bbox_util.detection_out(y)
            
                    if len(results) > 0 and len(results[0]) > 0:
                        det_label = results[0][:, 0]
                        det_conf = results[0][:, 1]
                        det_xmin = results[0][:, 2]
                        det_ymin = results[0][:, 3]
                        det_xmax = results[0][:, 4]
                        det_ymax = results[0][:, 5]

                        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                        top_conf = det_conf[top_indices]
                        top_label_indices = det_label[top_indices].tolist()
                        top_xmin = det_xmin[top_indices]
                        top_ymin = det_ymin[top_indices]
                        top_xmax = det_xmax[top_indices]
                        top_ymax = det_ymax[top_indices]

                        for i in range(top_conf.shape[0]):
                            xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                            ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                            xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                            ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                            class_num = int(top_label_indices[i])
                            cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), 
                                          self.class_colors[class_num], 2)
                            text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])

                            text_top = (xmin, ymin-10)
                            text_bot = (xmin + 80, ymin + 5)
                            text_pos = (xmin + 5, ymin)
                            cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                            cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                            
                            
                            # append incoming data into csv
                            df = pd.DataFrame({'frame':[cur_f],'object':[self.class_names[class_num]],'coord':[[xmin,ymin,xmax,ymax]]})
                            df_tmp = pd.concat([df_tmp,df.loc[df['frame']==cur_f]],axis=0)
                            
                            #print(text)
                    
                    ## FPS ##
                    curr_time = timer()
                    exec_time = curr_time - prev_time
                    prev_time = curr_time
                    accum_time = accum_time + exec_time
                    curr_fps = curr_fps + 1
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                        
                    cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
                    cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                    
                    
                    ## create output csv file ##
                    df_res = df_tmp.loc[df_tmp['frame']==cur_f]
                    def ray(x):
                        res_1 = df_res[df_res['object']==x]
                        a=res_1.groupby('object').agg({'frame':'count'})
                        b=res_1.groupby('object')['coord'].apply(list)
                        res_2=pd.concat([a,b],axis=1)
                        res_3 = res_2.rename({x:cur_f})
                        return res_3
                    
                    df_res_1 = ray(item_1)
                    df_res_2 = ray(item_2)
                    df_res_3 = ray(item_3)
                    df_res_fin = pd.concat([df_res_1,df_res_2,df_res_3],axis=1)
                    print(df_res_fin)
                
                    with open(csv_name,'a') as f:
                        df_res_fin.to_csv(f, header=False)
                        
                        
                    ## write out the video file ##
                    out.write(to_draw)
            
            
            ## handling ctrl c ##
            except KeyboardInterrupt:
                out.release()
                cmd = ['avconv -i ./output_video/.tmp.avi -vcodec libx264 ./output_video/%s.mp4' % time_as_name]
                cmd2 = ['rm ./output_video/.tmp.avi']
                subprocess.call(cmd, shell=True)
                subprocess.call(cmd2, shell=True)
                break
            
        
