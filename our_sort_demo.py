import os
import cv2
import numpy as np

from YOLO3 import YOLO3
from yolov3.util import process_result
from yolov3.detector import detect_frame
from deep_sort import DeepSort
from yolov3.darknet import Darknet
from util1 import COLORS_10, draw_bboxes

import time

class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("YOLO3/cfg/yolo_v3.cfg","/local/b/cam2/data/HumanBehavior/yolov3.weights","YOLO3/cfg/coco.names", is_xywh=True)
        self.deepsort = DeepSort("/local/b/cam2/data/HumanBehavior/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("sort_demo.avi", fourcc, 30, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        
    def detect(self):
        xmin, ymin, xmax, ymax = self.area


        model = Darknet("./yolov3/cfg/yolov3.cfg")
        model.load_weights("/local/b/cam2/data/HumanBehavior/yolov3.weights")
        model.cuda()
        model.eval()
        print("Loaded YOLO....")

        # keep track of frame number we are on
        frame_num = 1

        while self.vdo.grab(): 
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)]

            #bbox_xywh, cls_conf, cls_ids = self.yolo3(im)
            '''
            print("xy: \n", bbox_xywh)
            print("conf: \n", cls_conf)
            print("ids: \n", cls_ids)
            print("-----------------")
            '''
            bbox_xywh, cls_confs, cls_ids = detect_frame(model, im)
            '''
            print("xy: \n", bbox_xywh)
            print("conf: \n", cls_conf)
            print("ids: \n", cls_ids)
            print("-----------------")
            '''
            # convert to MOT 16 format
            xs  = bbox_xywh[0]
            ys  = bbox_xywh[1]
            ws = bbox_xywh[2]
            hs = bbox_xywh[3]
            new_xs = xs - ws / 2
            new_ys = ys - hs / 2
            
            MOT16_bbox = np.zeros(10)
            first = True
            for cls_id, cls_conf,x,y,w,h in zip(cls_ids, cls_confs, new_xs, new_ys, ws, hs):
                # MOT16_bbox = np.array([frame_num, cls_ids, new_x, new_y, w, h, cls_conf, -1, -1, -1])
                MOT16_bbox = np.append([MOT16_bbox], [[frame_num, cls_id, new_x, new_y, w, h, cls_conf, -1, -1, -1], axis=0)

             

            frame_num += 1

            end = time.time()
            #print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

            #cv2.imshow("test", ori_im)
            #cv2.waitKey(1)

            #self.output.write(ori_im)
            # TODO: remove when processing whole video
            break
        print(MOT16_bbox)
        print("Done...")


if __name__=="__main__":
    import sys
    if len(sys.argv) == 1:
        print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    else:
        #cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("test", 800,600)
        det = Detector()
        det.open(sys.argv[1])
        det.detect()
