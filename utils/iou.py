def calculate_max_iou_per_prediction(boxes1, boxes2):
    """Calculate the maximum IoU for each prediction box with respect to all ground truth boxes.

    Arguments:
    boxes1, boxes2 -- lists of bounding boxes; each box is a list of [xmin, ymin, xmax, ymax]

    Returns:
    max_ious -- list of tuples; each tuple contains the maximum IoU for a prediction and the index of the GT box that achieved it
    """
    max_ious = []
    for i, box1 in enumerate(boxes1):
        ious = []
        for box2 in boxes2:
            iou = calculate_iou(box1, box2)
            ious.append(iou)
        max_iou = max(ious)
        # max_iou_index = ious.index(max_iou)
        # max_ious.append((max_iou, max_iou_index))
        max_ious.append(max_iou)
    try:
        return max(max_ious)
    except:
        return [0]

def calculate_iou(box1, box2):
    """Calculate intersection over union (IoU) of two bounding boxes.

    Arguments:
    box1, box2 -- each box is a list

    Returns:
    iou -- IoU value
    """

    # Calculate overlap area's boundaries
    xmin_overlap = max(box1[0], box2[0])
    ymin_overlap = max(box1[1], box2[1])
    xmax_overlap = min(box1[2], box2[2])
    ymax_overlap = min(box1[3], box2[3])

    # Calculate overlap area
    overlap_area = max(0, xmax_overlap - xmin_overlap + 1) * max(0, ymax_overlap - ymin_overlap + 1)

    # Calculate area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - overlap_area

    # Calculate IoU
    iou = overlap_area / float(union_area)

    return iou
