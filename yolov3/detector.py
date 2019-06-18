import torch
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', required=True, help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

    args = parser.parse_args()

    return args

def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches

def draw_bbox(imgs, bbox, colors, classes):
    img = imgs[int(bbox[0])]
    label = classes[int(bbox[-1])]
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())

    color = random.choice(colors)
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)

def detect_frame(model, frame):
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    #colors = pkl.load(open("pallete", "rb"))
    # classes = load_classes("data/coco.names")
    #colors = [colors[1]]

    frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
    frame_tensor = Variable(frame_tensor)

    frame_tensor = frame_tensor.cuda()

    detections = model(frame_tensor, True).cpu()
           
    #print(detections.shape)

    #processresult changes the variable 'detections'
    detections = process_result(detections, 0.5, 0.4)
    cls_conf = detections[:, 6].cpu().data.numpy()
    cls_ids = detections[:, 7].cpu().data.numpy()
    print(detections)
   # print(cls_conf.cpu().data.numpy(), "\n",cls_ids.cpu().data.numpy(),"\n" ,detections)
    
   # print('Getting here')
    
    if len(detections) != 0:
        detections = transform_result(detections, [frame], input_size)
                
    xywh = detections[:, 1:5]
    xywh[:, 0] = (detections[:, 1] + detections[:, 3]) / 2
    xywh[:, 1] = (detections[:, 2] + detections[:, 4]) / 2
                
    # TODO: width and hieght are double what they're supposed to be and dunno why
    xywh[:, 2] = abs(detections[:, 3] - detections[:, 1]) *2
    xywh[:, 3] = abs(detections[:, 2] - detections[:, 4]) *2
    xywh = xywh.cpu().data.numpy() #-> THe final bounding box that can be replaced in the deepSort
    ######################################################                                
    #print(xywh)
    return xywh, cls_conf, cls_ids


def detect_video(model, args):

   # draw_bbox([frame], detection, colors, classes)
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")
    colors = [colors[1]]
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')
    while cap.isOpened():
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            
            print(detections.shape)
            

            #processresult changes the variable 'detections'
            # print(detections)
            #print(args.obj_thresh)
            #print(args.nms_thresh)
            detections, a, b = process_result(detections, args.obj_thresh, args.nms_thresh)

            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)
                for detection in detections:
                    draw_bbox([frame], detection, colors, classes)
                
                xywh = detections[:,1:5]
                xywh[:, 0] = (detections[:, 1] + detections[:, 3]) / 2
                xywh[:, 1] = (detections[:, 2] + detections[:, 4]) / 2
                
                # TODO: width and hieght are double what they're supposed to be and dunno why
                xywh[:, 2] = abs(detections[:, 3] - detections[:, 1]) #*2
                xywh[:, 3] = abs(detections[:, 2] - detections[:, 4]) #*2
                xywh = xywh.cpu().data.numpy() #-> THe final bounding box that can be replaced in the deepSort
            ######################################################                                
            #print ("xy: \n", xywh)
            out.write(frame)
            if read_frames % 30 == 0:
                print('Number of frames processed:', read_frames)
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    out.release()

    print('Detected video saved to ' + output_path)

    return


def detect_image(model, args):

    print('Loading input image(s)...')
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    batch_size = int(model.net_info['batch'])

    imlist, imgs = load_images(args.input)
    print('Input image(s) loaded')

    img_batches = create_batches(imgs, batch_size)

    # load colors and classes
    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")

    if not osp.exists(args.outdir):
        os.makedirs(args.outdir)

    start_time = datetime.now()
    print('Detecting...')
    for batchi, img_batch in enumerate(img_batches):
        img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
        img_tensors = torch.stack(img_tensors)
        img_tensors = Variable(img_tensors)
        if args.cuda:
            img_tensors = img_tensors.cuda()
        detections = model(img_tensors, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)
        if len(detections) == 0:
            continue

        detections = transform_result(detections, img_batch, input_size)
        for detection in detections:
            draw_bbox(img_batch, detection, colors, classes)

        for i, img in enumerate(img_batch):
            save_path = osp.join(args.outdir, 'det_' + osp.basename(imlist[batchi*batch_size + i]))
            cv2.imwrite(save_path, img)

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))

    return

def main():

    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('/local/b/cam2/data/HumanBehavior/yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    if args.video:
        detect_video(model, args)

    else:
        detect_image(model, args)



if __name__ == '__main__':
    main()
