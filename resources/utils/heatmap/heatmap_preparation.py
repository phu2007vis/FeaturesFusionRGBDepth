import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from resources.utils.data_prepareation import *
from resources.utils.visualize import save_frames_as_video
import multiprocessing as mlp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.4, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

elbow_connections = [[12,14],[14,16],[11,13],[13,15]]
elbow_color = (255,165,0)
body_color = (255, 192, 203)
# Function to draw heatmap based on pose landmarks connections
hand_color = [(255,0,0)]*4
hand_color.extend([(0,255,0)]*4)
hand_color.extend([(0,0,255)]*4)
hand_color.extend([(255,255,255)]*4)
hand_color.extend([(112,112,0)]*4)
hand_color_mapping = {}

for i in range(1,21):
    hand_color_mapping[i] = hand_color[i-1]

def draw_hand_heatmap(image, landmarks,connections):
    
    if landmarks is None:
        return image
    for connection in connections:
        landmark1 = landmarks.landmark[connection[0]]
        landmark2 = landmarks.landmark[connection[1]]
       
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = image.shape
        pt1 = (int(landmark1.x * width), int(landmark1.y * height))
        pt2 = (int(landmark2.x * width), int(landmark2.y * height))
        hand_color_index = max(connection[0],connection[1])
        hand_color = hand_color_mapping[hand_color_index]
        cv2.line(image, pt1, pt2, hand_color, 2)

    return image
def draw_body_heatmap(image,landmarks,connections,containing_head = False):
    if landmarks is None:
        return image
    for connection in connections:
        is_head = connection[0] < 11
        if is_head and not containing_head:
            continue
        body_color_index = max(connection[0],connection[1])
        if body_color_index >26:
            continue
        landmark1 = landmarks.landmark[connection[0]]
        landmark2 = landmarks.landmark[connection[1]]
       
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = image.shape
        pt1 = (int(landmark1.x * width), int(landmark1.y * height))
        pt2 = (int(landmark2.x * width), int(landmark2.y * height))
      
        cv2.line(image, pt1, pt2, body_color, 2)
    for connection in elbow_connections:
      
        landmark1 = landmarks.landmark[connection[0]]
        landmark2 = landmarks.landmark[connection[1]]
       
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = image.shape
        pt1 = (int(landmark1.x * width), int(landmark1.y * height))
        pt2 = (int(landmark2.x * width), int(landmark2.y * height))
      
        cv2.line(image, pt1, pt2, elbow_color, 2)

    return image
def draw_all_heatmap(annotated_image):
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    results = holistic.process(annotated_image)
    annotated_image = np.zeros_like(annotated_image)
    annotated_image = draw_hand_heatmap(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    annotated_image = draw_hand_heatmap(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    annotated_image = draw_body_heatmap(annotated_image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,containing_head=True)
    return annotated_image
     
def extract_heatmap_from_array(all_frame):
    list_heatmap_frame = []
    for frame in all_frame:
        heatmap = draw_all_heatmap(frame)
        list_heatmap_frame.append(heatmap)
    return np.array(list_heatmap_frame)

def pre_process(param):
    video_path,heatmap_raw_outpath,heatmap_path, opts = param 
    video_object = get_video_generator(video_path, opts)
    all_frame = np.array(video_object.frames)[:,:]
    list_heatmap_frame = []
    for frame in all_frame:
        heatmap = draw_all_heatmap(frame)
        list_heatmap_frame.append(heatmap)
    np.save(heatmap_raw_outpath, np.array(list_heatmap_frame))
    save_frames_as_video(list_heatmap_frame,heatmap_path)
    video_object.reset()
    


def mass_process(opts):
    param_list = []
    for sub_folder in os.listdir(opts.data_root):
        
        sub_folder_path = os.path.join(opts.data_root,sub_folder)
        rgb_folder  = os.path.join(sub_folder_path,"rgb")
        
        heatmap_raw_outfolder = os.path.join(sub_folder_path,"heatmap_raw")
        heatmap_folder =  os.path.join(sub_folder_path,"heatmap")
        
        os.makedirs(heatmap_raw_outfolder,exist_ok=True)
        os.makedirs(heatmap_folder,exist_ok=True)
        
        for rgb_file_name in os.listdir(rgb_folder):
            
            input_path = os.path.join(rgb_folder,rgb_file_name)
            
            heatmap_raw_output_name = rgb_file_name.replace(".avi",".npy")
            heatmap_raw_output_path = os.path.join(heatmap_raw_outfolder,heatmap_raw_output_name)
            
            heatmap_output_name = rgb_file_name
            heatmap_output_path = os.path.join(heatmap_folder,heatmap_output_name)
            
            param = [input_path,heatmap_raw_output_path,heatmap_output_path,opts]
            param_list.append(param)
    pre_process(param_list[0])
    with mlp.get_context("spawn").Pool(mlp.cpu_count()-3) as pool:
        for result in tqdm(pool.imap_unordered(pre_process,param_list),total=len(param_list)):
            pass
            
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-process the video into formats which i3d uses.')
    parser.add_argument(
        '--data_root',
        type=str,
        default=r"/work/21013187/SignLanguageRGBD/data/ver2_all_rgb_only",
        help='Where you want to save the output input_folder')
    # Sample arguments
    parser.add_argument(
        '--sample_num',
        type=int,
        default='32',
        help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')
    parser.add_argument(
        '--resize',
        type=int,
        default='224',
        help="Resize the short edge video to '--resize'. Mention that this is only the pre-process, random crop"
             "will be applied later when training or testing, so here 'resize' can be a little bigger.")
    
    args = parser.parse_args()
    mass_process(args)



# OpenCV video capture
# cap = cv2.VideoCapture('/work/21013187/SignLanguageRGBD/data/76-100/76_81/A76P1/rgb/126_A76P9_.avi')  # You can change the parameter to a video file path if needed
# ret, frame = cap.read()
# IMAGE_WIDTH, IMAGE_HEIGHT = frame.shape[:2][::-1]
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Adjust resolution and FPS as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated_image = draw_all_heatmap(frame)
   
	
  
#     image_with_heatmap = annotated_image

#     # Convert the image back to BGR for OpenCV display
#     image_with_heatmap = cv2.cvtColor(image_with_heatmap, cv2.COLOR_RGB2BGR)

#     # Write the frame into the output video file
#     out.write(image_with_heatmap)


# cap.release()
# out.release()
# cv2.destroyAllWindows()
