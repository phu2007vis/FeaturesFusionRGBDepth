import os
from typing import List

import cv2
import numpy as np
import torch

def save_frames_as_video(frames: List[np.ndarray],
                         output_path: str,
                         fps: int = 30) -> None:
    """
    Save a list of frames as a video.

    Args:
        frames (List[np.ndarray]): List of frames to be saved.
        output_path (str): Output path for the video.
        fps (int, optional): Frames per second. Defaults to 30.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def visualize_rgb(dataloader: torch.utils.data.DataLoader,
                  output_folder: str,
                  percent_visualize = 1,
                  std_dev: float = 0,
                  mean_val: float = 255,
                  fps: int = 30) -> None:
    """
    Visualize RGB data from a dataloader and save as videos.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader containing the RGB data.
        output_folder (str): Output folder to save the visualizations.
        std_dev (float, optional): Standard deviation. Defaults to 0.
        mean_val (float, optional): Mean value. Defaults to 255.
        fps (int, optional): Frames per second. Defaults to 30.
        percent_visualize: percent visualize of dataloader. Defaults to 1 (iter all dataloader)
    """
    print(f"Save a visualize at {output_folder}")
    assert(percent_visualize>0 and percent_visualize <=1)
    count = 0 
    num_visalize = int(percent_visualize*len(dataloader))
    print(f"Visualize total {num_visalize} samples")
    for x, labels in dataloader:
        b, c, t, w, h = x.shape
        x = x * mean_val + std_dev
        for i in range(b):
            label = torch.softmax(labels[i],dim = -1).max(-1)[1]
            video = x[i].permute(1, 2, 3, 0)
            video_np = np.clip(np.array(video, dtype=np.uint8), 0, 255)
            count += 1
            subfolder = os.path.join(output_folder, "action_"+ str(label.item()))
            os.makedirs(subfolder, exist_ok=True)
            output_path = os.path.join(subfolder, f"{count}.mp4")
            save_frames_as_video(video_np, output_path, fps=fps)
            if num_visalize <= count:
                exit()
                
