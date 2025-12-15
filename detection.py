import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage import io
from config import model, COCO_INSTANCE_CATEGORY_NAMES

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

@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)
    results = model(input_tensor)[0]
    pred_boxes = results['boxes'].cpu().numpy()
    pred_scores = results['scores'].cpu().numpy()
    pred_labels = results['labels'].cpu().numpy()
    final_detections = []
    for i in range(len(pred_scores)):
        score = pred_scores[i]
        label_id = int(pred_labels[i])
        if label_id < len(COCO_INSTANCE_CATEGORY_NAMES):
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
        else:
            continue
        if score < min_confidence:
            continue
        if labels is not None:
            if label_name.lower() not in [l.lower() for l in labels]:
                continue
        xmin, ymin, xmax, ymax = pred_boxes[i]
        final_detections.append([label_id, xmin, ymin, xmax, ymax])
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
    frame = Image.open(img_path).convert("RGB")
    frame = np.array(frame)
    detections = extract_detections(frame, labels=['person', 'car', 'cat'])
    frame = draw_detections(frame, detections)
    io.imshow(frame)
    io.show()

if __name__ == "__main__":
    main()
