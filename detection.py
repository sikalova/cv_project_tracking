import os
import numpy as np
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
model.conf = 0.5
model.iou = 0.45

COLOR_MAP = {}

def get_color(label):
    return COLOR_MAP.setdefault(label, np.random.randint(0, 256, size=3))

def detection_cast(detections):
    if len(detections) == 0:
        return np.empty((0, 5), dtype=np.int32)
    return np.array(detections, dtype=np.int32).reshape((-1, 5))

def rectangle(shape, ll, rr, line_width=5):
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []
    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))
    return tuple(zip(*result))

def extract_detections(frame, min_confidence=0.5, labels=None):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    final_detections = []

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        label_name = model.names[cls_id]

        if conf < min_confidence:
            continue

        if labels is not None:
             if label_name.lower() not in [l.lower() for l in labels]:
                continue

        final_detections.append([cls_id, x1, y1, x2, y2])

    return detection_cast(final_detections)

def draw_detections(frame, detections):
    frame = frame.copy()
    for detection in detections:
        label = detection[0]
        xmin = detection[1]
        ymin = detection[2]
        xmax = detection[3]
        ymax = detection[4]
        rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
        color = get_color(label)
        frame[rr, cc] = color
    return frame

def main():
    dirname = os.path.dirname(__file__)
    img_path = os.path.join(dirname, "data", "test2.png")
    
    if os.path.exists(img_path):
        frame = Image.open(img_path).convert("RGB")
        frame = np.array(frame)
        detections = extract_detections(frame, min_confidence=0.5, labels=['person', 'car', 'bicycle', 'bus', 'dog'])
        result_frame = draw_detections(frame, detections)
        output_path = os.path.join(dirname, "data", "result_detection.jpg")
        Image.fromarray(result_frame).save(output_path)

if __name__ == "__main__":
    main()