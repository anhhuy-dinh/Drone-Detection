import cv2
import math
import numpy as np
import os
import glob
import profile
import time
import copy
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

CLASSES = ('__background__', 'drone')
           
NMS_THRESH = 0.1
IOU_THRESH = 0.2
MAX_THRESH=1e-2

# import darknet functions to perform object detections
from yolov4.darknet.darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("yolov4/darknet/cfg/yolov4-obj_v2.cfg", "yolov4/darknet/data/obj.data", "yolov4/backup/yolov4-obj_v2_best.weights")
width = network_width(network)
height = network_height(network)

def convert4cropping(image, bbox, width_ratio, height_ratio, expand=0):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox

    image_h, image_w, _ = image.shape
    
    left = int((x - (w / 2.))*width_ratio) - expand
    right = int((x + (w / 2.))*width_ratio) + expand
    top = int(math.ceil((y - (h / 2.))*height_ratio)) - expand
    bottom = int(math.ceil((y + (h / 2.))*height_ratio)) + expand
    
    return [left, top, right, bottom]

# darknet helper function to run detection on image
def darknet_helper(img, width, height, thresh=0.5):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width/width
    height_ratio = img_height/height
    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, thresh=thresh)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def iou_score(predicted_bbox, gt_bboxes, return_gt=False):
  if len(gt_bboxes) == 0:
    if return_gt:
      return 0, 0
    return 0
  else:
    IoUs = np.array([bb_intersection_over_union(predicted_bbox, bbox) for bbox in gt_bboxes], dtype='float')
    # print(np.max(IoUs), gt_bboxes[np.argmax(IoUs)])
    if return_gt:
      return np.max(IoUs), gt_bboxes[np.argmax(IoUs)]
    return np.max(IoUs)

def ground_truth_bbox(txtfile, width, height, return_size=False):
  with open(txtfile, "r") as f:
    frame = f.readlines()
    f.close()

  bbox_list = {}
  unique_ground_truth = 0
  size = {'small': 0, 'medium': 0, 'large': 0}
  for i in range(len(frame)):
    no_frame, no_objs = [int(num) for num in frame[i].split()[:2]]
    bbox_list[no_frame] = []
    unique_ground_truth += no_objs
    if no_objs != 0:
      for j in range(no_objs):
        bbox = [int(num) for num in frame[i].split()[(2+5*j):(6+5*j)]]
        bbox_list[no_frame].append([max(0,bbox[0]-1), max(0,bbox[1]-1), min(width-1,bbox[0]+bbox[2]+1), min(height-1, bbox[1]+bbox[3]+1)])
        s = np.sqrt(bbox[2] * bbox[3])
        if s < 32: 
          size['small'] += 1
        elif s <= 64:
          size['medium'] += 1
        else:
          size['large'] += 1
  if return_size:
    return bbox_list, unique_ground_truth, size        
  return bbox_list, unique_ground_truth

def createLinks(dets_all):
    links_all = []

    frame_num = len(dets_all[0])
    cls_num = len(CLASSES) - 1
    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num - 1):
            dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind + 1]
            box1_num = len(dets1)
            box2_num = len(dets2)
            
            if frame_ind == 0:
                areas1 = np.empty(box1_num)
                for box1_ind, box1 in enumerate(dets1):
                    areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            else:
                areas1 = areas2

            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            links_frame = []
            for box1_ind, box1 in enumerate(dets1):
                area1 = areas1[box1_ind]
                x1 = np.maximum(box1[0], dets2[:, 0])
                y1 = np.maximum(box1[1], dets2[:, 1])
                x2 = np.minimum(box1[2], dets2[:, 2])
                y2 = np.minimum(box1[3], dets2[:, 3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if
                             ovr >= IOU_THRESH]
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all


def maxPath(dets_all, links_all):

    for cls_ind, links_cls in enumerate(links_all):

        max_begin = time.time()
        delete_sets=[[]for i in range(0,len(dets_all[0]))]
        delete_single_box=[]
        dets_cls = dets_all[cls_ind]

        num_path=0
        # compute the number of links
        sum_links=0
        for frame_ind, frame in enumerate(links_cls):
            for box_ind,box in enumerate(frame):
                sum_links+=len(box)

        while True:

            num_path+=1

            rootindex, maxpath, maxsum = findMaxPath(links_cls, dets_cls,delete_single_box)

            if (maxsum<MAX_THRESH or sum_links==0 or len(maxpath) <1):
                break
            if (len(maxpath)==1):
                delete=[rootindex,maxpath[0]]
                delete_single_box.append(delete)
            rescore(dets_cls, rootindex, maxpath, maxsum)
            t4=time.time()
            delete_set,num_delete=deleteLink(dets_cls, links_cls, rootindex, maxpath, NMS_THRESH)
            sum_links-=num_delete
            for i, box_ind in enumerate(maxpath):
                delete_set[i].remove(box_ind)
                delete_single_box.append([[rootindex+i],box_ind])
                for j in delete_set[i]:
                    dets_cls[i+rootindex][j]=np.zeros(5)
                delete_sets[i+rootindex]=delete_sets[i+rootindex]+delete_set[i]

        for frame_idx,frame in enumerate(dets_all[cls_ind]):

            a=range(0,len(frame))
            keep=list(set(a).difference(set(delete_sets[frame_idx])))
            dets_all[cls_ind][frame_idx]=frame[keep,:]


    return dets_all


def findMaxPath(links,dets,delete_single_box):

    len_dets=[len(dets[i]) for i in range(len(dets))]
    max_boxes=np.max(len_dets)
    num_frame=len(links)+1
    a=np.zeros([num_frame,max_boxes])
    new_dets=np.zeros([num_frame,max_boxes])
    for delete_box in delete_single_box:
        new_dets[delete_box[0],delete_box[1]]=1
    if(max_boxes==0):
        max_path=[]
        return 0,max_path,0

    b=np.full((num_frame,max_boxes),-1)
    for l in range(len(dets)):
        for j in range(len(dets[l])):
            if(new_dets[l,j]==0):
                a[l,j]=dets[l][j][-1]



    for i in range(1,num_frame):
        l1=i-1;
        for box_id,box in enumerate(links[l1]):
            for next_box_id in box:

                weight_new=a[i-1,box_id]+dets[i][next_box_id][-1]
                if(weight_new>a[i,next_box_id]):
                    a[i,next_box_id]=weight_new
                    b[i,next_box_id]=box_id

    i,j=np.unravel_index(a.argmax(),a.shape)

    maxpath=[j]
    maxscore=a[i,j]
    while(b[i,j]!=-1):

            maxpath.append(b[i,j])
            j=b[i,j]
            i=i-1


    rootindex=i
    maxpath.reverse()
    return rootindex, maxpath, maxscore


def rescore(dets, rootindex, maxpath, maxsum):
    newscore = maxsum / len(maxpath)

    for i, box_ind in enumerate(maxpath):
        dets[rootindex + i][box_ind][4] = newscore


def deleteLink(dets, links, rootindex, maxpath, thesh):

    delete_set=[]
    num_delete_links=0

    for i, box_ind in enumerate(maxpath):
        areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in dets[rootindex + i]]
        area1 = areas[box_ind]
        box1 = dets[rootindex + i][box_ind]
        x1 = np.maximum(box1[0], dets[rootindex + i][:, 0])
        y1 = np.maximum(box1[1], dets[rootindex + i][:, 1])
        x2 = np.minimum(box1[2], dets[rootindex + i][:, 2])
        y2 = np.minimum(box1[3], dets[rootindex + i][:, 3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h

        ovrs = inter / (area1 + areas - inter)
        #saving the box need to delete
        deletes = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= thesh]
        delete_set.append(deletes)

        #delete the links except for the last frame
        if rootindex + i < len(links):
            for delete_ind in deletes:
                num_delete_links+=len(links[rootindex+i][delete_ind])
                links[rootindex + i][delete_ind] = []

        if i > 0 or rootindex > 0:

            #delete the links which point to box_ind
            for priorbox in links[rootindex + i - 1]:
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)
                        num_delete_links+=1

    return delete_set,num_delete_links

def seq_nms(dets):
    links = createLinks(dets)
    dets=maxPath(dets, links)
    return dets

def detection(video_path, ann_path, name_video, model=None, thresh_yolo=0.5, expand_bbox=0, difference=0, dsize=(64,64), return_size=False):  
  cap = cv2.VideoCapture(video_path)
  WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  if return_size == True:
    gt_bboxes, no_gts, size_gt = ground_truth_bbox(ann_path, WIDTH, HEIGHT, return_size=True)
  else:
    gt_bboxes, no_gts = ground_truth_bbox(ann_path, WIDTH, HEIGHT)

  no_frame = TP = FP = count_FN = 0
  dets_all = [[] for i in range(len(CLASSES)-1)]
  frame_no_detect = []
  pred_size = {'small': 0, 'medium': 0, 'large': 0}
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame_dets = []
    detections, width_ratio, height_ratio = darknet_helper(frame, width, height, thresh_yolo)
    for label, confidence, bbox in detections:
      predicted_bbox = convert4cropping(frame, bbox, width_ratio, height_ratio, expand_bbox)
      left, top, right, bottom = predicted_bbox
      if (no_frame - difference) in gt_bboxes.keys(): 
        if model is not None:
          predicted_name = recognization(model, frame, left, top, right, bottom, dsize)
        
          if predicted_name == 'none':
            IoU, gt = 0, 0
          else:
            IoU, gt = iou_score(predicted_bbox, gt_bboxes[no_frame-difference], return_gt=True)
        else:
          IoU, gt = iou_score(predicted_bbox, gt_bboxes[no_frame-difference], return_gt=True)

        if IoU >= 0.5:
          TP += 1
          if gt != 0:
            w = gt[2] - gt[0]
            h = gt[3] - gt[1]
            s = np.sqrt(w*h)
            if s < 32: 
              pred_size['small'] += 1
            elif s <= 64:
              pred_size['medium'] += 1
            else:
              pred_size['large'] += 1
        else: 
          FP += 1
        
        cls_box = np.array([left-10, top-10, right+10, bottom+10], dtype=np.float64)
        cls_score = np.array([eval(confidence)], dtype=np.float64)
        cls_det = np.hstack((cls_box, cls_score)).astype(np.float64)
        frame_dets.append(cls_det)
      else:
        continue
    if frame_dets != []:
      dets_all[0].append(np.array(frame_dets))
    else:
      frame_no_detect.append(no_frame)
    no_frame += 1

  FN = no_gts - TP
  TP_rate = TP/no_gts
  print("============== YOLOv4 ==============")
  print("\nNumber of ground truth objects = {}".format(no_gts))
  print("TP = {}, FP = {}, FN = {}".format(TP,FP,FN))
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = (2*precision*recall) / (precision+recall)
  print("* Precision: ", precision)
  print("* Recall   : ", recall)
  print("* F1 score : ", F1)

  cap.release()
  if return_size:
    if model is not None:
      size_df = {'ground_truth': size_gt, 'YOLOv4+MobileNet': pred_size}
    else:
      size_df = {'ground_truth': size_gt, 'YOLOv4': pred_size}
    return dets_all, frame_no_detect, size_df

  return dets_all, frame_no_detect

def evaluate(video_path, ann_path, name_video, model=None, thresh_yolo=0.5, thresh_conf=10.0, expand_bbox=0, difference=0, dsize=(64,64), output_video_path=None, size_df=None):
  if output_video_path is not None:
    try:
      os.mkdir(output_video_path)
    except OSError:
      pass
  if size_df is not None:
    return_size = True
  else: 
    return_size = False

  if return_size:
    dets_all, frame_no_detect, pre_size = detection(video_path, ann_path, name_video, model, thresh_yolo, expand_bbox, difference, dsize, return_size)
  else:
    dets_all, frame_no_detect = detection(video_path, ann_path, name_video, model, thresh_yolo, expand_bbox, difference, dsize)
  result = seq_nms(dets_all)
  cap = cv2.VideoCapture(video_path)
  WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  gt_bbox, unique_ground_truth = ground_truth_bbox(ann_path, WIDTH, HEIGHT)
  out = cv2.VideoWriter(output_video_path + name_video + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (int(WIDTH), int(HEIGHT)))

  no_frame = TP = FP = FN = 0
  count_temp = count = 0
  pred_size = {'small': 0, 'medium': 0, 'large': 0}
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    if (count_temp) in frame_no_detect: 
      count_temp += 1
      out.write(frame)
      continue

    for bbox in result[0][count]:
      if bbox[-1] >= thresh_conf:
        pred_bbox = [bbox[0]+10, bbox[1]+10, bbox[2]-10, bbox[3]-10] 
        IOU, gt = iou_score(pred_bbox, gt_bbox[count_temp - difference], return_gt=True)
        if IOU >= 0.5: 
          TP += 1
          w = gt[2] - gt[0]
          h = gt[3] - gt[1]
          s = np.sqrt(w*h)
          if s < 32: 
            pred_size['small'] += 1
          elif s <= 64:
            pred_size['medium'] += 1
          else:
            pred_size['large'] += 1
          left, top, right, bottom = int(bbox[0]+10), int(bbox[1]+10), int(bbox[2]-10), int(bbox[3]-10)
          cv2.rectangle(frame, (left, top), (right, bottom), color=(0,255,0), thickness = 1)
          confidence = bbox[-1]
          font_scale = 0.5
          font = cv2.FONT_HERSHEY_SIMPLEX 
          text = "drone: {:.1f}%".format(confidence)
          thickness = 2 
          (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
          pos = ((left+right)//2 - text_width//2, top-5)
          cv2.rectangle(frame, (pos[0]-2, pos[1]-text_height-2), (pos[0]+text_width+2, pos[1]+2), (0,255,0), -1)
          cv2.putText(frame, text, pos, font, font_scale, (0,0,0), thickness)
        else: FP += 1
          
    count += 1; count_temp += 1
    out.write(frame)
  FN = unique_ground_truth - TP
  print("============== YOLOv4 + Seq-NMS ==============")
  print("\nunique_ground_truth = {}".format(unique_ground_truth))
  print("TP = {}, FP = {}, FN = {}".format(TP, FP, FN))
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = (2*precision*recall) / (precision+recall)
  print("* Precision: ", precision)
  print("* Recall   : ", recall)
  print("* F1 score : ", F1)
  cap.release()
  out.release()

  if return_size:
    pre_size['YOLOv4+Seq-NMS'] = pred_size
    for method in size_df.keys():
      for size in size_df[method].keys():
        size_df[method][size] += pre_size[method][size]
    return size_df

if __name__ == "__main__":
    name_video = 'gopro_008.mp4'
    video_path = 'yolov4/test_videos/' + name_video
    ann_path = 'yolov4/test_videos/annotations/' + name_video[slice(0,-4)] + '.txt'
    output_video_path = 'yolov4/test_folder/'
    evaluate(video_path, ann_path, name_video[slice(0,-4)], model=None, thresh_yolo=0.1, expand_bbox=1, difference=0, dsize=(64,64), output_video_path=output_video_path, thresh_conf=25.0)