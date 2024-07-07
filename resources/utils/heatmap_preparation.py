import cv2
import mediapipe as mp
import numpy as np
import time


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
    annotated_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(annotated_image)
    # annotated_image = np.zeros_like(annotated_image)
    annotated_image = draw_hand_heatmap(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    annotated_image = draw_hand_heatmap(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    annotated_image = draw_body_heatmap(annotated_image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,containing_head=True)
    return annotated_image
     



# OpenCV video capture
cap = cv2.VideoCapture('/work/21013187/SignLanguageRGBD/data/76-100/76_81/A76P1/rgb/126_A76P9_.avi')  # You can change the parameter to a video file path if needed
ret, frame = cap.read()
IMAGE_WIDTH, IMAGE_HEIGHT = frame.shape[:2][::-1]
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Adjust resolution and FPS as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_image = draw_all_heatmap(frame)
   
	
  
    image_with_heatmap = annotated_image

    # Convert the image back to BGR for OpenCV display
    image_with_heatmap = cv2.cvtColor(image_with_heatmap, cv2.COLOR_RGB2BGR)

    # Write the frame into the output video file
    out.write(image_with_heatmap)


cap.release()
out.release()
cv2.destroyAllWindows()
