import os
import re
import cv2
import csv
import glob
import json
import numpy as np

# セグメンテーションマスクのパス一覧を取得
segmentation_mask_paths = glob.glob('/home/data/mvtec/*/ground_truth/*/*.png')
segmentation_mask_paths = sorted([file_path for file_path in segmentation_mask_paths if "/good/" not in file_path])
image_paths = glob.glob('/home/data/mvtec/*/test/*/*.png')
image_paths = sorted([file_path for file_path in image_paths if "/good/" not in file_path])
with open('bbox.csv') as f:
    pass
bbox = {}

for mask_path, image_path in zip(segmentation_mask_paths, image_paths):
    if not len(segmentation_mask_paths) == len(image_paths):
        print(f'seg{len(segmentation_mask_paths)}')
        print(f'img{len(image_paths)}')
        break
    # セグメンテーションマスクを読み込み
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # バウンディングボックスを描画したマスクを保存
    match = re.match(r'/home/data/mvtec/(.*?)/ground_truth/(.*?)/(\d+_mask.png)', mask_path)
    category, defect_type, filename = match.groups()
    new_dir_path = os.path.join('mvtecv2', category, 'bounding_boxes', defect_type)
    os.makedirs(new_dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(new_dir_path, filename), image)
    bb = []
    for x, y, w, h in bounding_boxes:
        bb.append([x, y, x+w, y+h])
    if category not in bbox:
        bbox[category] = {}
    if defect_type not in bbox[category]:
        bbox[category][defect_type] = {}
    bbox[category][defect_type][filename] = bb
    
with open('bbox.json', 'w') as f:
    json.dump(bbox, f, indent=4)
