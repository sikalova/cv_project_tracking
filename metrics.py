def iou_score(bbox1, bbox2):
    assert len(bbox1) == 4
    assert len(bbox2) == 4
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    area_intersection = max(0, y2_inter - y1_inter) * max(0, x2_inter - x1_inter)
    area_a = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_b = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    area_union = area_a + area_b - area_intersection
    if area_union == 0:
        return 0.0
    iou = area_intersection / area_union
    return iou
