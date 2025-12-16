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
def motp(obj, hyp, threshold=0.5):
    dist_sum = 0
    match_count = 0
    matches = {}
    for frame_obj, frame_hyp in zip(obj, hyp):
        frame_obj_dict = {detection[0]: detection for detection in frame_obj}
        frame_hyp_dict = {detection[0]: detection for detection in frame_hyp}
        matched_new = {}
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                iou = iou_score(frame_hyp_dict[hyp_id][1:], frame_obj_dict[obj_id][1:])
                if iou > threshold:
                    match_count += 1
                    dist_sum += iou
                    matched_new[obj_id] = hyp_id
                    del frame_obj_dict[obj_id]
                    del frame_hyp_dict[hyp_id]
        iou_list = []
        for obj_id, obj_detection in frame_obj_dict.items():
            for hyp_id, hyp_detection in frame_hyp_dict.items():
                iou = iou_score(obj_detection[1:], hyp_detection[1:])
                if iou > threshold:
                    iou_list.append((iou, obj_id, hyp_id))
        iou_list.sort(reverse=True, key=lambda x: x[0])
        for iou, obj_id, hyp_id in iou_list:
            if obj_id in matched_new or hyp_id in matched_new.values():
                continue
            dist_sum += iou
            match_count += 1
            matched_new[obj_id] = hyp_id
        matches = matched_new
    MOTP = 0 if match_count == 0 else dist_sum / match_count
    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    dist_sum = 0
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    count_obj = 0
    matches = {}
    for frame_idx, (frame_obj, frame_hyp) in enumerate(zip(obj, hyp)):
        frame_obj_dict = {detection[0]: detection for detection in frame_obj}
        frame_hyp_dict = {detection[0]: detection for detection in frame_hyp}
        len_frame_obj = len(frame_obj_dict)
        len_frame_hyp = len(frame_hyp_dict)
        old_frame_hyp_dict = frame_hyp_dict.copy()
        matched_new = {}
        count_obj += len(frame_obj_dict)
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                iou = iou_score(frame_hyp_dict[hyp_id][1:], frame_obj_dict[obj_id][1:])
                if iou > threshold:
                    match_count += 1
                    dist_sum += iou
                    matched_new[obj_id] = hyp_id
                    del frame_obj_dict[obj_id]
                    del frame_hyp_dict[hyp_id]
        iou_list = []
        for obj_id, obj_detection in frame_obj_dict.items():
            for hyp_id, hyp_detection in frame_hyp_dict.items():
                iou = iou_score(obj_detection[1:], hyp_detection[1:])
                if iou > threshold:
                    iou_list.append((iou, obj_id, hyp_id))
        iou_list.sort(reverse=True, key=lambda x: x[0])
        for iou, obj_id, hyp_id in iou_list:
            if obj_id in matched_new or hyp_id in matched_new.values():
                continue
            dist_sum += iou
            match_count += 1
            matched_new[obj_id] = hyp_id
        for obj_new, hyp_new in matched_new.items():
            if obj_new in matches and matches[obj_new] != hyp_new:
                mismatch_error += 1
matches = matched_new if len(matched_new) > 0 else matches
        missed_count += len_frame_obj - len(matched_new)
        false_positive += len_frame_hyp - len(matched_new)
        pass

    MOTP = 0 if match_count == 0 else dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / count_obj

    return MOTP, MOTA
