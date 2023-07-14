import os
import re
import cv2
import csv
import glob
import numpy as np

# セグメンテーションマスクのパス一覧を取得
segmentation_mask_paths = glob.glob('/data/dataset/mvtec/*/ground_truth/*/*.png')
segmentation_mask_paths = sorted([file_path for file_path in segmentation_mask_paths if "/good/" not in file_path])
image_paths = glob.glob('/data/dataset/mvtec/*/test/*/*.png')
image_paths = sorted([file_path for file_path in image_paths if "/good/" not in file_path])
with open('bbox.csv') as f:
    pass

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

    # 非ゼロのピクセルの座標を取得
    # nonzero_pixels = np.nonzero(mask)

    # 非ゼロのピクセルが存在するか確認
    # if nonzero_pixels[0].size > 0:
        # 非ゼロのピクセルが存在する場合
        # y, x = nonzero_pixels

        # 最小および最大のx座標とy座標を取得
        # xmin = np.min(x)
        # ymin = np.min(y)
        # xmax = np.max(x)
        # ymax = np.max(y)

        # マスクのコピーを作成し、その上にバウンディングボックスを描画
        # mask_with_box = image
        # cv2.rectangle(mask_with_box, (xmin, ymin), (xmax, ymax), [0, 0, 255], 3)
    # else:
        # 非ゼロのピクセルが存在しない場合（すなわち、異常箇所がない場合）
        # mask_with_box = image

    # バウンディングボックスを描画したマスクを保存
    match = re.match(r'/data/dataset/mvtec/(.*?)/ground_truth/(.*?)/(\d+_mask.png)', mask_path)
    category, defect_type, filename = match.groups()
    new_dir_path = os.path.join('mvtecv2', category, 'bounding_boxes', defect_type)
    os.makedirs(new_dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(new_dir_path, filename), image)
    
    with open('bbox.csv', 'a') as f:
        bb = []
        writer = csv.writer(f, lineterminator="\n")
        for x, y, w, h in bounding_boxes:
            bb.append([x, y, x+w, y+h])
        writer.writerows(bb)
