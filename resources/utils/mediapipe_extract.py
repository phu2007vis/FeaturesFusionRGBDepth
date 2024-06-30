import mediapipe as mp
import cv2
from data_prepareation import video_loader
import numpy as np
from  visualize import save_frames_as_video
import multiprocessing  as mps
import os
import tqdm
# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


def detectPose(image):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append([landmark.x , landmark.y ,landmark.z]) 
    else: 
        return None
    return landmarks
def extract_one_file(video_path,save_path):
    video_list = video_loader(video_path)
    landmark_list = []
    for frame in video_list:
        pose = detectPose(frame)
        if pose is not None:
            landmark_list.append(pose)
    if len(landmark_list) > 20:
        numpy_result = np.array(landmark_list)
        np.save(save_path,numpy_result)
    else:
        try:
            os.remove(save_path)
        except:
            pass
    
def extract_one_file_paralel(param):
    video_path,save_path = param
    video_list = video_loader(video_path)
    landmark_list = []
    for frame in video_list:
        pose = detectPose(frame)
        if pose is not None:
            landmark_list.append(pose)
    if len(landmark_list) > 20:
        numpy_result = np.array(landmark_list)
        np.save(save_path,numpy_result)
    else:
        try:
            os.remove(save_path)
        except:
            pass
    



def visualize_poses(visualize_path, np_result_path,frame_size = 224):
    """
    Load numpy result and visualize the detected poses on the video frames.

    Args:
        video_path (str): Path to the video file.
        np_result_path (str): Path to the numpy result file.
    """

    # Load numpy result
    numpy_result = np.load(np_result_path)
    
    list_frame = []
    shift_top = 0.2
    shift_left = 0.2
    # Iterate over frames
    for i in range(numpy_result.shape[0]):
        
        # Get detected landmarks for current frame
        landmarks = numpy_result[i]
        frame = np.zeros((frame_size,frame_size,3),dtype=np.uint8)
        # Visualize landmarks
        for landmark in landmarks:
        
            x, y, z = landmark
            x -= shift_left
            y -= shift_top
            
            # Do something to visualize the landmarks, like drawing circles
            cv2.circle(frame, (max(int(x * frame.shape[1]),0), max(int(y * frame.shape[0]),0)), 5, (0, 255, 0), -1)
        
        list_frame.append(frame)

    save_frames_as_video(list_frame,visualize_path)
def extract_all_folder(root_folder,skip_exist = True,safe_mode = True):
    '''
    structure of root_folder:
    --root_folder
        --sub_folder
            --rgb
                video1
                video2
                video3
                ...
        --subfolder2
        ....
    '''
    paralel_list = []
    for sub_folder_name in os.listdir(root_folder):
        sub_folder_path   = os.path.join(root_folder,sub_folder_name,'rgb')
        output_folder_path = os.path.join(root_folder,sub_folder_name,'mediapipe_landmarks')
        if os.path.exists(sub_folder_path) and len(os.listdir(sub_folder_path)) > 0:
            os.makedirs(output_folder_path,exist_ok=True)
        else:
            print(f"Folder {sub_folder_path} not exists or empty !")
        for video_name in os.listdir(sub_folder_path):
            video_id = video_name.split(".")
            video_id = video_id[0] if len(video_id) <=2 else '.'.join(video_id[:-1])
            video_id = video_id
            skeleton_name = video_id+'.npy'
            video_path = os.path.join(sub_folder_path,video_name)
            skeleton_path = os.path.join(output_folder_path,skeleton_name)
            if os.path.exists(skeleton_path) and skip_exist:
                continue
            paralel_list.append((video_path,skeleton_path))
            
    for video_path,skeleton_path in paralel_list:
        print(f"{video_path} extract save at {skeleton_path}")
   
    if safe_mode:
        check = input("Do you want to continue (y/n): ")
        if  'y' not in check:
            exit()
    cpu = mps.cpu_count() -2
    with mps.get_context("spawn").Pool(processes=cpu ) as p:
        with tqdm.tqdm(total=len(paralel_list)) as pbar:
            for _ in p.imap_unordered(extract_one_file_paralel,paralel_list):
                pbar.update(1)
    # pbar = tqdm.tqdm(total=len(paralel_list))
    # for video_path,skeleton_path in paralel_list:
    #    extract_one_file(video_path,skeleton_path)
    #    pbar.update(1)
if __name__ == "__main__":
    # input_video , landmark_path,visualize_path = r"/work/21013187/SignLanguageRGBD/data/ver1/A1P1/rgb/259_P1_.avi","visualize/landmkar.npy","visualize/lanm.mp4"
    # extract_one_file(input_video,landmark_path)
    # visualize_poses(visualize_path,landmark_path)
    
    
    # cpu = mps.cpu_count()-10
    # with Pool(processes=cpu ) as p:
    #     max_ = 30
    #     with tqdm(total=max_) as pbar:
    #         for _ in p.imap_unordered(_foo, range(0, max_)):
    #             pbar.update(1)
    folder_path = "/work/21013187/SignLanguageRGBD/data/76-100/data"
    extract_all_folder(folder_path,False,safe_mode=False)