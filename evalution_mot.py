import os
import cv2
import numpy as np
from metrics import motp_mota
from cross_correlation import CorrelationTracker

def parse_mot_gt(gt_path):
    gt_dict = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            obj_id = int(parts[1])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            xmin = left
            ymin = top
            xmax = left + width
            ymax = top + height
            detection = [obj_id, xmin, ymin, xmax, ymax]
            if frame not in gt_dict:
                gt_dict[frame] = []
            gt_dict[frame].append(detection)
    num_frames = max(gt_dict.keys())
    ground_truth = []
    for i in range(1, num_frames + 1):
        if i in gt_dict:
            ground_truth.append(gt_dict[i])
        else:
            ground_truth.append([])
    return ground_truth

def run_tracker_on_mot_sequence(seq_path, detection_rate=5):
    img_dir = os.path.join(seq_path, 'img1')
    images = sorted([img for img in os.listdir(img_dir) if img.endswith('.jpg')])
    tracker = CorrelationTracker(detection_rate=detection_rate, return_images=False)
    hypotheses = []
    print(f"Обработка последовательности: {os.path.basename(seq_path)}")
    print(f"Всего кадров: {len(images)}")
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = tracker.update_frame(frame)
        if hasattr(detections, 'tolist'):
            detections = detections.tolist()
        hypotheses.append(detections)
        if len(hypotheses) % 50 == 0:
            print(f"Обработано {len(hypotheses)} кадров...")
    return hypotheses

def main():
    dataset_path = "data/MOT15/train/ADL-Rundle-6" 
    gt_path = os.path.join(dataset_path, "gt", "gt.txt")
    ground_truth = parse_mot_gt(gt_path)
    hypotheses = run_tracker_on_mot_sequence(dataset_path, detection_rate=5)
    min_len = min(len(ground_truth), len(hypotheses))
    ground_truth = ground_truth[:min_len]
    hypotheses = hypotheses[:min_len]
    motp, mota = motp_mota(ground_truth, hypotheses, threshold=0.5)
    print("="*40)
    print(f"РЕЗУЛЬТАТЫ MOT15 (Detection Rate = 5)")
    print("="*40)
    print(f"MOTA (Accuracy):  {mota:.4f}")
    print(f"MOTP (Precision): {motp:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
