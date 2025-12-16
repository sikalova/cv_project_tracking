
import os
import argparse
from moviepy.editor import VideoFileClip
from cross_correlation import CorrelationTracker
from metrics import motp_mota

def run_demo(input_video, output_video, detection_rate=5):
    print(f"\n" + "="*40)
    print(f"   ЗАПУСК ШАГА 1: ДЕМОНСТРАЦИЯ (Видео)")
    print(f"="*40)
    print(f"Входной файл: {input_video}")
    print(f"Сохранение в: {output_video}")
    print(f"Детектор срабатывает каждые {detection_rate} кадров")
    tracker = CorrelationTracker(detection_rate=detection_rate)
    input_clip = VideoFileClip(input_video)
    output_clip = input_clip.fl_image(tracker.update_frame)
    output_clip.write_videofile(
        output_video, 
        audio=False, 
        codec='libx264',    
        fps=25,             
        preset='medium',
        ffmpeg_params=['-pix_fmt', 'yuv420p'],
        verbose=False,
        logger='bar'
    )
    print(f"Видео готово!")


def run_evaluation(video_path, rate_fast=5):
    print(f"\n" + "="*40)
    print(f"   ЗАПУСК ШАГА 2: ОЦЕНКА КАЧЕСТВА (Метрики)")
    print(f"="*40)
    if not os.path.exists(video_path):
        print(f"ОШИБКА: Видео {video_path} не найдено!")
        return
    print("--> Генерация 'Эталона' (Запуск нейросети на каждом кадре)...")
    tracker_gt = CorrelationTracker(detection_rate=1)
    tracker_gt.return_images = False 
    clip = VideoFileClip(video_path)
    ground_truth = []
    for frame in clip.iter_frames():
        detections = tracker_gt.update_frame(frame)
        ground_truth.append(detections)
    print(f"--> Генерация 'Гипотезы' (Запуск нейросети раз в {rate_fast} кадр(ов))...")
    tracker_fast = CorrelationTracker(detection_rate=rate_fast)
    tracker_fast.return_images = False
    hypotheses = []
    for frame in clip.iter_frames():
        detections = tracker_fast.update_frame(frame)
        hypotheses.append(detections)
    print("--> Подсчет метрик...")
    motp, mota = motp_mota(ground_truth, hypotheses, threshold=0.5)
    print("\n" + "="*30)
    print(f"ИТОГОВЫЙ ОТЧЕТ (Detection Rate = {rate_fast})")
    print("="*30)
    print(f"MOTA (Качество трекинга): {mota:.4f}")
    print(f"MOTP (Точность рамок):    {motp:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Трекер объектов (SSD + Correlation)")
    parser.add_argument('--mode', type=str, default='all', choices=['demo', 'eval', 'all'],
                        help="Режим: 'demo' (видео), 'eval' (метрики) или 'all' (оба сразу)")
    
    parser.add_argument('--video', type=str, default='data/test.mp4',
                        help="Путь к видеофайлу")
    
    parser.add_argument('--rate', type=int, default=5,
                        help="Как часто запускать SSD (раз в N кадров)")

    args = parser.parse_args()
    dir_name = os.path.dirname(args.video)
    if not dir_name: dir_name = "."
    base_name = os.path.basename(args.video)
    name_no_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(dir_name, f"{name_no_ext}_result.mp4")
    if args.mode == 'all':
        run_demo(args.video, output_path, args.rate)
        run_evaluation(args.video, args.rate)
    elif args.mode == 'demo':
        run_demo(args.video, output_path, args.rate)
    elif args.mode == 'eval':
        run_evaluation(args.video, args.rate)