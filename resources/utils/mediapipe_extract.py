import mediapipe as mp
import cv2
from data_prepareation import video_loader
import numpy as np
from  visualize import save_frames_as_video
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
    
    return landmarks
def extract_one_file(video_path,save_path):
    video_list = video_loader(video_path)
    landmark_list = []
    for frame in video_list:
        landmark_list.append(detectPose(frame))
    numpy_result = np.array(landmark_list)
    np.save(save_path,numpy_result)
    


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
        frame = np.zeros((frame_size,frame_size))
        # Visualize landmarks
        for landmark in landmarks:
        
            x, y, z = landmark
            x -= shift_left
            y -= shift_top
            
            # Do something to visualize the landmarks, like drawing circles
            cv2.circle(frame, max(int(x * frame.shape[1]),0), max(int(y * frame.shape[0]),0), 5, (0, 255, 0), -1)
        
        list_frame.append(frame)
    print("Saving video")
    save_frames_as_video(list_frame,visualize_path)

if __name__ == "__main__":
    input_video , landmark_path,visualize_path = r"C:\Users\Admin\phuoc\Baseline_ViSL1\test_data\data_folder\A1P18\rgb\58_A1P18_.avi","test_data/landmkar.npy","test_data/lanm.mp4"
    # extract_one_file(input_video,landmark_path)
    visualize_poses(visualize_path,landmark_path)