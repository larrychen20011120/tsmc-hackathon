import time
from pathlib import Path
import os
import cv2
import torch
import numpy as np

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box, color_list
from utils.torch_utils import select_device

def detect(img, save_dir, model, device, imgsz, conf_thres, iou_thres):
    img0 = img # origin image
    without_cnt = 0

    device = select_device(device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
        
    #dataset = LoadImages(source_path, img_size=imgsz, stride=stride)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = color_list()
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    s = ''
    t0 = time.time()
    img = letterbox(img, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        total = len(det)
        save_path = f'{save_dir}/result.jpg' # img.jpg
        txt_path = f'{save_dir}/result'
        s += '%gx%g ' % img.shape[2:]  # print string
        with open(txt_path + '.txt', 'w') as f:
            pass
        if len(det):
            # Rescale boxes from img_size to img size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                xyxy2 = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                line = (names[c], conf, *xyxy2)
                without_cnt = without_cnt+1 if not c else without_cnt
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%s ' + '%g ' * (len(line) - 1)).rstrip() % line + '\n')

                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)


        # Save results (image with detections)
        cv2.imwrite(save_path, img0)
        print(f" The image with the result is saved in: {save_path}")
        return without_cnt, total


    print(f'Done. ({time.time() - t0:.3f}s)')
