import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from numpy import ndarray
from PIL import Image
from skimage import measure
from sklearn.metrics import auc
from statistics import mean

def get_pro(mask_path: str, amaps=None, boxes=None, scale_factor: float = 0.25, num_th: int = 200, image_size: int = 256) -> None:
    if boxes is not None: # BBの座標から塗りつぶしたマスクを生成
        gt_masks = Image.open(mask_path)
        gt_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()])
        gt_masks = gt_transforms(gt_masks)
        gt_masks = gt_masks.bool()
        gt_masks = gt_masks.squeeze(0).cpu().numpy().astype(int)
        gt_masks = gt_masks[np.newaxis, :, :]

        num_test_data, height, width = gt_masks.shape
        amaps = np.zeros((num_test_data, height, width), dtype=np.uint8)
        for bbox in boxes:
            start_point = (int(bbox[0] * scale_factor), int(bbox[1] * scale_factor)) # Top left corner
            end_point = (int(bbox[2] * scale_factor), int(bbox[3] * scale_factor)) # Bottom right corner
            amaps[0, start_point[1]:end_point[1], start_point[0]:end_point[0]] = 1
        
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_masks, ndarray), "type(gt_masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_masks.ndim == 3, "gt_masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == gt_masks.shape, "amaps.shape and gt_masks.shape must be same"
    assert set(gt_masks.flatten()) == {0, 1}, "set(gt_masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        # 2値化処理(segmentation)
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        # pro計算
        pros = []
        for binary_amap, mask in zip(binary_amaps, gt_masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        # fpr計算
        inverse_masks = 1 - gt_masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame([mean(pros), fpr, th])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    
    # print(df)
    import pdb
    pdb.set_trace()
    pro_auc = auc(df["fpr"], df["pro"])

    return None, pros
