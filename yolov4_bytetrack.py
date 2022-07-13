from cmath import exp
import cv2
import math
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
import os.path as osp
from loguru import logger
import torch
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess, vis
from ByteTrack.yolox.utils.visualize import plot_tracking
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
import argparse
import time
from yolov4.darknet.darknet import *

IMAGE_EXT = [".jpg"]
device = "gpu"

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
def darknet_helper(network, img, class_names, width, height, thresh=0.5):
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

def get_gt_bboxes(txtfile, width, height, return_size=False):
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

def get_image_list(path):
  image_names = []
  for maindir, subdir, file_name_list in os.walk(path):
    for filename in file_name_list:
      apath = osp.join(maindir, filename)
      ext = osp.splitext(apath)[1]
      if ext in IMAGE_EXT:
        image_names.append(apath)
  return image_names

def write_results(filename, results):
  save_format = "{frame}, {id}, {x1}, {y1}, {w}, {h}, {s}, -1, -1, -1\n"
  with open(filename, 'w') as f:
    for frame_id, tlwhs, track_ids, scores in results:
      for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
        if track_id < 0:
          continue
        x1, y1, w, h = tlwh
        line = save_format.format(
            frame=frame_id, 
            id=track_id, 
            x1=round(x1, 1), 
            y1=round(y1,1),
            w=round(w, 1),
            h=round(h,1),
            s=round(score,2)
        )
        f.write(line)
  logger.info('save results to {}'.format(filename))

def get_name_video(path):
    start = path.rfind('/') + 1
    end = path.rfind('.')
    return path[start:end]

class Predictor(object):
  def __init__(
      self,
      model,
      exp,
      trt_file=None,
      decoder=None,
      device='cuda' if torch.cuda.is_available() else 'cpu',
      fp16=False
  ):
    self.model = model
    self.decoder= decoder
    self.num_classes = exp.num_classes
    self.class_names = exp.class_names
    self.confthre = exp.test_conf
    self.nmsthre = exp.nmsthre
    self.test_size = exp.test_size
    self.device = device
    self.fp16 = fp16
    self.expand = exp.expand

    self.rgb_means = (0,485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)

  def inference(self, img, timer):
    img_info = {"id": 0}
    if isinstance(img, str):
      img_info["file_name"] = osp.basename(img)
      img = cv2.imread(img)
    else:
      img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    
    with torch.no_grad():
      timer.tic()
      outputs = []
      detections, width_ratio, height_ratio = darknet_helper(self.model, img, self.class_names, width, height, self.nmsthre)
      for label, score, bbox in detections:
        pd_bbox = convert4cropping(img, bbox, width_ratio, height_ratio, self.expand)
        l, t, r, b = pd_bbox
        cls_bbox = np.array([l, t, r, b], dtype=np.float64)
        cls_score = np.array([eval(score+'/100')], dtype=np.float64)
        cls_det = np.hstack((cls_bbox, cls_score)).astype(np.float64)
        outputs.append(cls_det)
      #logger.info("Infer time: {:.4f}s".format(time.time() - t0))

    return np.array(outputs), img_info

def image_demo(predictor, vis_folder, current_time, args, exp):
  if osp.isdir(args.path):
    files = get_image_list(args.path)
  else:
    files = [args.path]
  files.sort()
  tracker = BYTETracker(args, frame_rate=args.fps)
  timer = Timer()
  results = []

  for frame_id, img_path in enumerate(files, 1):
    outputs, img_info = predictor.inference(img_path, timer)
    if outputs is not None:
      online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], exp.test_size)
      online_tlwhs = []
      online_ids = []
      online_scores = []
      for t in online_targets:
        tlwh = t.tlwh
        print(tlwh)
        tid = t.track_id
        online_tlwhs.append(tlwh)
        online_ids.append(tid)
        online_scores.append(t.score)
        # save results
        results.append(
            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
        )
      timer.toc()
      online_im = plot_tracking(
          img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
      )
    else:
      timer.toc()
      online_im = img_info['raw_img']

    if args.save_result:
      timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
      save_folder = osp.join(vis_folder, timestamp)
      os.makedirs(save_folder, exist_ok=True)
      cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

    if frame_id % 20 == 0:
      logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
    
    ch = cv2.waitKey(0)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
      break

  if args.save_result:
    res_file = osp.join(vis_folder, f"{timestamp}.txt")
    with open(res_file, 'w') as f:
      f.writelines(results)
    logger.info(f"save results to {res_file}")

def imageflow_demo(predictor, vis_folder, current_time, args, exp, evl=None):
  cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
  name_vid = get_name_video(args.path)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
  save_folder = osp.join(vis_folder, name_vid)
  os.makedirs(save_folder, exist_ok=True)
  
  if args.demo == "video":
    save_path = osp.join(save_folder, args.path.split("/")[-1])
  else:
    save_path = osp.join(save_folder, "camera.mp4")
  if exp.test_size == None:
    exp.test_size = (height, width)
  logger.info(f"video save_path is {save_path}")
  vid_writer = cv2.VideoWriter(
      save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
  )
  if args.ann_path is not None and args.ann_path != 'None':
    gt_bboxes, number_gts = get_gt_bboxes(args.ann_path, width, height)
    number_frs = TPs = FPs = FNs = 0
    number_fr = TP = FP = FN = 0
    pred_size = {'small': 0, 'medium': 0, 'large': 0}

  tracker = BYTETracker(args, frame_rate=args.frame_rate)
  timer = Timer()
  frame_id = 0
  results = []
  while True:
    if frame_id % 200 == 0:
      logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
    ret_val, frame = cap.read()
    if ret_val:
      outputs, img_info = predictor.inference(frame, timer)
      if outputs.shape[0] != 0:
        online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], exp.test_size)
        online_tlwhs = []
        online_ids =[]
        online_scores = []
        for i, t in enumerate(online_targets):
          tlwh = t.tlwh
          tlbr = [tlwh[0], tlwh[1], tlwh[2]+tlwh[0], tlwh[3]+tlwh[1]]
          tid = t.track_id
          if args.ann_path is not None and args.ann_path != 'None':
            IOU, gt = iou_score(tlbr, gt_bboxes[frame_id], return_gt=True)
            if IOU >= args.iou_thre:
              TPs += 1
              w = gt[2] - gt[0]
              h = gt[3] - gt[1]
              s = np.sqrt(w*h)
              if s < 32:
                pred_size['small'] += 1
              elif s <= 64:
                pred_size['medium'] += 1
              else:
                pred_size['large'] += 1
              online_tlwhs.append(tlwh)
              online_ids.append(tid)
              online_scores.append(t.score)
              results.append(
                  f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f}, {t.score:.2f}, -1, -1, -1\n"
              )
            else:
              FPs += 1
          else:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            results.append(
                f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f}, {t.score:.2f}, -1, -1, -1\n"
            )
          
        timer.toc()
        online_im = plot_tracking(
            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id+1, fps=1./timer.average_time
        )
      else:
        timer.toc()
        online_im = img_info['raw_img']
      if args.save_result:
        vid_writer.write(online_im)
      ch = cv2.waitKey(0)
      if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break
    else:
      break
    frame_id += 1
  
  if args.save_result:
    res_file = osp.join(vis_folder, f"{timestamp}.txt")
    with open(res_file, 'w') as f:
      f.writelines(results)
    logger.info(f"save results to {res_file}")

  if args.ann_path is not None and args.ann_path != 'None':
    FNs = number_gts - TPs
    print("="*40)
    print("unique_ground_truth = {}".format(number_gts))
    print("TP = {}, FP = {}, FN = {}".format(TPs, FPs, FNs))
    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    F1 = (2*precision*recall) / (precision+recall)
    print("* Precision: ", precision)
    print("* Recall   : ", recall)
    print("* F1 score : ", F1)

  if evl is not None:
    for tp in evl['type_metric'].keys():
      if tp == 'TP':
        evl['type_metric'][tp] += TPs
      if tp == 'FP':
        evl['type_metric'][tp] += FPs
      if tp == 'FN':
        evl['type_metric'][tp] += FNs
    for size in evl['size'].keys():
      evl['size'][size] += pred_size[size]
    return evl

def evaluate(exp, args, evl=None):
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.network

    trt_file = None
    decoder = None
    
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        if evl is not None:
            evl = imageflow_demo(predictor, vis_folder, current_time, args, evl)
            return evl
        else:
            imageflow_demo(predictor, vis_folder, current_time, args, evl)
    logger.info("Done!")

def make_parser():
  parser = argparse.ArgumentParser("ByteTrack Demo!")
  parser.add_argument(
      "demo", default="image", help="demo type, eg. image, video"
  )
  parser.add_argument("-expn", "--experiment-name", type=str, default=None)
  parser.add_argument("-n", "--name", type=str, default=None, help="model name")
  parser.add_argument(
    "--cfg-path", type=str, default="yolov4/darknet/cfg/yolov4-obj_v2.cfg", help="path to configuration file of detector"
  )
  parser.add_argument(
    "--data-path", type=str, default="yolov4/darknet/data/obj.data", help="path to data file of detector"
  )
  parser.add_argument(
    "--weight-path", type=str, default="yolov4/backup/yolov4-obj_v2_best.weights", help="path to weight file of detector"
  )
  parser.add_argument(
      "--path", default="/mydrive/Dataset/test_videos/00_09_30_to_00_10_09.mp4", help="path to images or video"
  )
  parser.add_argument(
      "--ann-path", default=None, type=str, help="path to annotation file according to input video/images for evaluation"
  )
  parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
  parser.add_argument(
      "--save_result",
      action="store_true",
      help="whether to save the inference result of image/video"
  )

  # exp file
  parser.add_argument(
      "-f",
      "--exp_file",
      default=None,
      type=str,
      help="pls input your experiment description file",
  )
  parser.add_argument("-c", "-ckpt", default=None, type=str, help="ckpt for eval")
  parser.add_argument(
      "--device",
      default="gpu",
      type=str,
      help="device to run our model, can either be cpu or gpu",
  )
  parser.add_argument("--conf", default=None, type=float, help="test conf")
  parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
  parser.add_argument("--tsize", default=None, type=int, help="test img size")
  parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
  parser.add_argument(
      "--fp16",
      dest="fp16",
      default=False,
      action="store_true",
      help="Adopting mix precision evaluating.",
  )
  parser.add_argument(
      "--fuse",
      dest="fuse",
      default=False,
      action="store_true",
      help="Fuse conv and bn for testing.",
  )
  parser.add_argument(
      "--trt",
      dest="trt",
      default=False,
      action="store_true",
      help="Using TensorRt model for testing.",
  )
  parser.add_argument(
      "--output-dir",
      default=None,
      type=str,
      help="input your output direction path"
  )
  
  # tracking args
  parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
  parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
  parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
  parser. add_argument("--match_thre_second_as", type=float, default=0.5, help="matching threshold for second association")
  parser.add_argument("--match_thre_lost", type=float, default=0.7, help="matching threshold for lost track")
  parser.add_argument(
      "--aspect_ratio_threshold", type=float, default=1.6,
      help="threshold for filtering out boxes of which aspect ratio are above the given value."
  )
  parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
  parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20")
  parser.add_argument("--augument", action="store_true", help="augumented inference")
  parser.add_argument("--iou-thre", type=float, default=0.5, help="IoU threshold for evaluation")
  parser.add_argument("--frame-rate", type=int, default=30, help="frame rate for BYTETracker")
  parser.add_argument("--low-thre", type=float, default=0.1, help="low threshold to get second detections in BYTETracker")
  return parser

class Exp:
    def __init__(
        self,
        network,
        class_names,
        num_classes = 1,
        test_size = (1056, 1888),
        test_conf = 0.1,
        nmsthre = 0.01,
        expand = 0
    ):
        self.network = network
        self.class_names = class_names
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = num_classes
        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = test_size
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = test_conf
        # nms threshold
        self.nmsthre = nmsthre
        # expand factor is used for expanding the predicted bounding box
        self.expand = expand

def main(args):
    network_best, class_names, class_colors = load_network(args.cfg_path, args.data_path, args.weight_path)
    exp = Exp(network=network_best, test_size=None, nmsthre=0.01, expand=1)
    evaluate(exp, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
