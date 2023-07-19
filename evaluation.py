import re
import os
import glob
import json
from tqdm import tqdm
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotatev2
from utils.iou import calculate_max_iou_per_prediction

# initial setting
TEXT_PROMPTS = ['anomaly', 'damage', 'defect', 'own']
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join("weights", WEIGHTS_NAME)
IMAGESIZE = 256
CROPSIZE = 256
ORIGINALSIZE = 1024
detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.3
with open('bbox.json', 'r') as f:
    ground_truth = json.load(f)
result_dict = {}

def main():
    segmentation_mask_paths = glob.glob('/home/data/mvtec/*/ground_truth/*/*.png')
    segmentation_mask_paths = sorted([file_path for file_path in segmentation_mask_paths if "/good/" not in file_path])
    image_paths = glob.glob('/home/data/mvtec/*/test/*/*.png')
    image_paths = sorted([file_path for file_path in image_paths if "/good/" not in file_path])

    for mask_path, data_path in tqdm(zip(segmentation_mask_paths, image_paths)):
        seg_match = re.match(r'/home/data/mvtec/(.*?)/ground_truth/(.*?)/(\d+_mask.png)', mask_path)
        _, _, mask_name = seg_match.groups()
        match = re.match(r'/home/data/mvtec/(.*?)/test/(.*?)/(\d+.png)', data_path)
        category, defect, file = match.groups()
        gt = ground_truth[category][defect][mask_name]
        TEXT_PROMPTS[-1] = data_path.split('/')[6]
        image_source, image = load_image(data_path)

        for text in TEXT_PROMPTS:

            boxes, logits, phrases = predict(
            model=detection_model, 
            image=image, 
            caption=text, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD)
            # save result figure
            annotatev2(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases, data_path=data_path, text=text)

            # save iou
            boxes *= 1024
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
            iou = calculate_max_iou_per_prediction(boxes, gt)
            if category not in result_dict:
                result_dict[category] = {}
            if defect not in result_dict[category]:
                result_dict[category][defect] = {}
            if text not in result_dict[category][defect]:
                result_dict[category][defect][text] = {}
            result_dict[category][defect][text][file] = iou
    with open('result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == '__main__':
    main()
