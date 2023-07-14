import re
import os
import glob
import json
from PIL import Image
import numpy as np
import cv2
import csv
from tqdm import tqdm
from torchvision.ops import box_convert
import pycocotools.mask as mask_util
from groundingdino.util.inference import load_model, load_image, predict, annotate, annotatev2
from utils.iou import calculate_iou
from utils.show_result import show_result

TEXT_PROMPTS = ['anomaly', 'damege', 'defect', 'own']
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join("weights", WEIGHTS_NAME)
detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.3
with open('bbox_old.csv', 'r') as f:
    ground_truth = f.read().split('\n')
with open('dino_result.csv', 'w') as f:
    pass
result_dict = {}

def main():
    idx = 0
    test_data_path_list = glob.glob("/data/dataset/mvtec/*/test/*/*.png")
    test_data_path_list = sorted([file_path for file_path in test_data_path_list if "/good/" not in file_path])
    
    for data_path in tqdm(test_data_path_list):
        TEXT_PROMPTS[-1] = data_path.split('/')[6]
        image_source, image = load_image(data_path)
        for text in TEXT_PROMPTS:

            boxes, logits, phrases = predict(
            model=detection_model, 
            image=image, 
            caption=text, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD)

            annotatev2(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases, data_path=data_path, text=text)

            gt = ground_truth[idx]
            iou = []
            gt = list(map(float, gt.split(',')))
            for box in boxes:
                box = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").tolist()
                box = [b*1024 for b in box]
                iou.append(calculate_iou(box, list(map(float, gt))))
            
            match = re.match(r'/data/dataset/mvtec/(.*?)/test/(.*?)/(\d+.png)', data_path)
            category, defect, file = match.groups()
            if category not in result_dict:
                result_dict[category] = {}
            if defect not in result_dict[category]:
                result_dict[category][defect] = {}
            if text not in result_dict[category][defect]:
                result_dict[category][defect][text] = {}
            result_dict[category][defect][text][file] = iou
        
        idx += 1
    with open('result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == '__main__':
    main()
