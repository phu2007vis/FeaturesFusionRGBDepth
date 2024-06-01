from ultralytics import YOLO
import torch
import cv2
device = "cpu" if not torch.cuda.is_available() else "cuda"


def extract_one_video(model
                      list_frame):
    pass


if __name__ == "__main__":
    model_name = "yolov8x.pt"
    model = YOLO(model_name)