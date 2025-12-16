import os
import numpy as np
from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score
from moviepy.editor import VideoFileClip

class Tracker:
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images
        self.frame_index = 0
        self.labels = labels
        self.detection_history = []
        self.last_detected = {}
        self.tracklet_count = 0
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        detections = extract_detections(frame)
        for detection in detections:
            detection[0] = self.new_label()
        return detection_cast(detections) 

    @property
    def prev_detections(self):
        start_frame = max(0, self.frame_index - self.lookup_tail_size)
        last_seen = {}
        for frame_num in range(start_frame, self.frame_index):
            frame_detections = self.detection_history[frame_num]
            for detection in frame_detections:
                tracklet_id = detection[0]
                last_seen[tracklet_id] = detection
        
        return detection_cast(list(last_seen.values()))

    def bind_tracklet(self, detections):
        detections = detections.copy()
        prev_detections = self.prev_detections
        iou_list = []
        for curr_idx, curr_det in enumerate(detections):
            for prev_idx, prev_det in enumerate(prev_detections):
                iou = iou_score(curr_det[1:], prev_det[1:])
                iou_list.append((iou, curr_idx, prev_idx, prev_det[0]))
        iou_list.sort(reverse=True, key=lambda x: x[0])
        matched_curr = set()
        matched_prev = set()
        iou_threshold = 0.3
        for iou_value, curr_idx, prev_idx, prev_tracklet_id in iou_list:
            if curr_idx in matched_curr or prev_tracklet_id in matched_prev:
                continue
            if iou_value < iou_threshold:
                break
            detections[curr_idx, 0] = prev_tracklet_id
            matched_curr.add(curr_idx)
            matched_prev.add(prev_tracklet_id)
        for curr_idx in range(len(detections)):
            if curr_idx not in matched_curr:
                detections[curr_idx, 0] = self.new_label()
        return detection_cast(detections)
    
    def save_detections(self, detections):
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
        else:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
        self.save_detections(detections)
        self.detection_history.append(detections)
        self.frame_index += 1
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_path = os.path.join(dirname, "data", "test.mp4")
    output_path = os.path.join(dirname, "data", "test_result.mp4")
    input_clip = VideoFileClip(input_path)
    tracker = Tracker()
    output_clip = input_clip.fl_image(tracker.update_frame)
    output_clip.write_videofile(
        output_path, 
        audio=False, 
        codec='libx264', 
        fps=25,
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )

if __name__ == "__main__":
    main()