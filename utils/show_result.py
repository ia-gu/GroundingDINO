import os
import re
import cv2
def show_result(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        box = [int(i*1024) for i in box.tolist()]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), [0, 0, 255], 3)
    match = re.match(r'/data/dataset/mvtec/(.*?)/test/(.*?)/(\d+.png)', image_path)
    category, defect_type, filename = match.groups()
    new_dir_path = os.path.join('result', category, defect_type)
    os.makedirs(new_dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(new_dir_path, filename), image)
    