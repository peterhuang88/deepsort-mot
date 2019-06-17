import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys

from YOLO3 import YOLO3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes
import warnings
import time

import json

# check on gpu?

# yolov3 detect
# deepsort track
# => bbox
#
# homography value: map real_img -> map2d
#

# ignore DeprecationWarning
warnings.filterwarnings("ignore")

# start a dict, to store coords
# use later for extracting info

# ex: coords = {
#  1 : [(x1,y1), pos2, pos3]
#  2 : [pos1, pos2]
# }
coord_dict = {}


# add h = homography value
def draw_points_on_map(map2d, h, bbox, identities):
    for i, box in enumerate(bbox):
        # top-left, bot-right of ori_img
        x1, y1, x2, y2 = [int(i) for i in box]
        x_mid = int((x1 + x2) / 2)
        y_mid = y2

        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]

        # convert to map2d
        a = np.array([[x_mid, y_mid]], dtype='float32')
        a = np.array([a])

        # ret = [ [[x, y]] ]
        ret = cv2.perspectiveTransform(a, h)
        pt = ret[0][0]  # ndarray: x, y
        pt = pt.tolist()  # cvt to python.list
        pt[0] = int(pt[0])
        pt[1] = int(pt[1])

        # draw on map
        cv2.circle(map2d, (pt[0], pt[1]), 5, color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(map2d, str(id), (pt[0] + 2, pt[1]), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # add to coordinate_dict
        if id in coord_dict:
            coord_dict[id].append( (pt[0], pt[1]) )
        else:
            coord_dict[id] = [ (pt[0], pt[1]) ]

    return map2d




# --- find homography matrix ---
pts_real = np.array(
    [[960, 663], [1287, 574], [1549, 909], [901, 526], [1302, 636]]
)
pts_map = np.array(
    [[504, 320], [392, 270], [578, 307], [316,315], [481,290]]
)

# calculate matrix H
h, status = cv2.findHomography(pts_real, pts_map)

# --- /find homography matrix ---


# --- test map point from map1 -> map2 ---

# a = np.array([[480, 272]], dtype='float32')
# a = np.array([a])
#
# b = cv2.perspectiveTransform(a, h)
#print(b)

# --- /test ---

# --- test on frame ---

# open map
img_map = cv2.imread("../video/purdue/msee.png")


# cv2.circle(img_map, (318, 256), 5, (0,0,255), -1)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img_map, '1', (320, 256), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


#cv2.imshow("map2d", img_map)
# --- /test ---

# --- try a video ---

# create capture, writer, yolo, deepsort
video = cv2.VideoCapture()
video.open('../video/purdue/trimmed.MTS')
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_area = 0, 0, video_width, video_height

img_map = cv2.imread("../video/purdue/msee.png")
img_height, img_width = img_map.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer1 = cv2.VideoWriter("result.avi", fourcc, 20, (video_width, video_height))
video_writer2 = cv2.VideoWriter("map.avi", fourcc, 20, (img_width, img_height))


yolo3 = YOLO3("YOLO3/cfg/yolo_v3.cfg","YOLO3/yolov3.weights","YOLO3/cfg/coco.names", is_xywh=True)
deepsort = DeepSort("deep/checkpoint/ckpt.t7")  # look into this checker
class_names = yolo3.class_names


for i in range(50):

    start = time.time()

    # open video
    # read new frame, reset map
    ret, frame1 = video.read()
    img_map = cv2.imread("../video/purdue/msee.png")

    if frame1 is None:
        break

    # frame1: in bgr, for displaying
    # frame2: in rgb, for detect, track
    frame2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # what is cls_conf, cls_ids ?
    bbox_xywh, cls_conf, cls_ids = yolo3(frame2)  # can change detector
    if bbox_xywh is not None:

        # look into this later
        mask = (cls_ids == 0)

        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:, 3] *= 1.2  # make it bigger to surround obj
        cls_conf = cls_conf[mask]

        outputs = deepsort.update(bbox_xywh, cls_conf, frame2)

        # draw bbox on ori_img
        if len(outputs) > 0:
            #print("run this")

            # 4col=bbox, 1col=id
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]

            frame1 = draw_bboxes(frame1, bbox_xyxy, identities, offset=(0, 0))

            # draw pts on map2d
            img_map = draw_points_on_map(img_map, h, bbox_xyxy, identities)

            # add to dict
            # if identities in


    end = time.time()
    print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))


    # cv2.imshow("process video", frame1)
    # cv2.imshow("map2d", img_map)

    video_writer1.write(frame1)
    video_writer2.write(img_map)

    cv2.waitKey(1)

# --- /try a video ---

# write dict to file
print("dict size: {}".format(len(coord_dict)))

coord_string = json.dumps(coord_dict)
f = open("coords.json", "w")
f.write(coord_string)
f.close()





