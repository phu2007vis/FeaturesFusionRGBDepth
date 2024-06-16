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
        labels = labels.max(1)[1]
        b, c, t, w, h = x.shape
        x = x * mean_val + std_dev
        for i in range(b):
            label = labels[i]
            video = x[i].permute(1, 2, 3, 0)
            video_np = np.clip(np.array(video, dtype=np.uint8), 0, 255)
            count += 1
            subfolder = os.path.join(output_folder, "action_"+ str(label.item()))
            os.makedirs(subfolder, exist_ok=True)
            output_path = os.path.join(subfolder, f"{count}.mp4")
            save_frames_as_video(video_np, output_path, fps=fps)
            if num_visalize <= count:
                exit()
def visualize_poses_from_result(numpy_result,
                                frame_size = 224,
                                shift_top = 0.2,
                                shift_left = 0.2):

    # Load numpy result
    
    
    list_frame = []
    # Iterate over frames
    for i in range(numpy_result.shape[0]):
        
        # Get detected landmarks for current frame
        landmarks = numpy_result[i]
        frame = np.zeros((frame_size,frame_size,3),dtype=np.uint8)
        # Visualize landmarks
        for landmark in landmarks:
            try:
                x, y, z = landmark
            except:
                x,y = landmark
            x -= shift_left
            y -= shift_top
            
            # Do something to visualize the landmarks, like drawing circles
            cv2.circle(frame, (max(int(x * frame.shape[1]),0), max(int(y * frame.shape[0]),0)), 5, (0, 255, 0), -1)
        
        list_frame.append(frame)
    return list_frame

def visualize_pose(dataloader: torch.utils.data.DataLoader,
                  output_folder: str,
                  percent_visualize = 1,
                  std_dev: float = 0,
                  mean_val: float = 1,
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
    for x,_, labels in dataloader:
        labels = labels.max(1)[1]
        b, t, n = x.shape
        x = x * mean_val + std_dev
        for i in range(b):
            label = labels[i]
            data = x[i].reshape(t,-1,2)
            video_list = visualize_poses_from_result(data)
            count += 1
            subfolder = os.path.join(output_folder, "action_"+ str(label.item()))
            os.makedirs(subfolder, exist_ok=True)
            output_path = os.path.join(subfolder, f"{count}.mp4")
            save_frames_as_video(video_list, output_path, fps=fps)
            if num_visalize <= count:
                exit()
