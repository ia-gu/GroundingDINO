import cv2
import numpy as np

def make_mask(image, bounding_boxes):

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create a black image with the same dimensions
    mask = np.zeros((height, width), dtype=np.uint8)

    for bbox in bounding_boxes:
        start_point = (int(bbox[0]), int(bbox[1]))  # Top left corner
        end_point = (int(bbox[2]), int(bbox[3]))  # Bottom right corner
        color = 255  # White
        thickness = -1  # Fill
        mask = cv2.rectangle(mask, start_point, end_point, color, thickness)
        
    return mask