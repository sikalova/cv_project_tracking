import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template
from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker
from moviepy.editor import VideoFileClip


def gaussian(shape, x, y, dx, dy):
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-((X - x)**2) / dx**2 - ((Y - y)**2) / dy**2)


class CorrelationTracker(Tracker):
    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate
        self.prev_frame = None

    def build_tracklet(self, frame):
        detections = []
        gray_frame = rgb2gray(frame)
        gray_prev = rgb2gray(self.prev_frame)
        
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            template = gray_prev[ymin:ymax, xmin:xmax]
            
            if template.size == 0 or template.shape[0] == 0 or template.shape[1] == 0:
                detections.append([label, xmin, ymin, xmax, ymax])
                continue
            
            new_bbox = gray_frame
            matching = match_template(new_bbox, template, pad_input=True)
            
            bbox_height = ymax - ymin
            bbox_width = xmax - xmin
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            
            gauss = gaussian(matching.shape, center_x, center_y, bbox_width // 2, bbox_height // 2)
            output = matching * gauss
            
            best_y, best_x = np.unravel_index(np.argmax(output), output.shape)
            
            new_xmin = best_x - bbox_width // 2
            new_ymin = best_y - bbox_height // 2
            new_xmax = best_x + bbox_width // 2
            new_ymax = best_y + bbox_height // 2
            
            detections.append([label, new_xmin, new_ymin, new_xmax, new_ymax])
        
        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)
        
        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1
        
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(dirname, "data", "test.mp4")
    input_clip = VideoFileClip(video_path)
    tracker = CorrelationTracker(detection_rate=5, return_images=True)
    processed_clip = input_clip.fl_image(tracker.update_frame)
    processed_clip.write_videofile("result_video.mp4", audio=False, fps=25)


if __name__ == "__main__":
    main()
