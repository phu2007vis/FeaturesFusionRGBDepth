#tranverse all avi file in the folder and get the duration of each video. summary and print min, max, average duration of all videos.

import os
import cv2
import datetime
def get_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    return duration

def main(path):
    print('path:', path)
    duration_list = []
    for root, dirs, files in os.walk(path):
        
        for file in files:
            if file.endswith('.avi'):
                video_path = os.path.join(root, file)
                duration = get_duration(video_path)
                duration_list.append(duration)
                print(file, duration)
        #print summary as hh:mm:ss
    print('min:', str(datetime.timedelta(seconds=min(duration_list))))
    print('max:', str(datetime.timedelta(seconds=max(duration_list))))
    print('average:', str(datetime.timedelta(seconds=sum(duration_list)/len(duration_list))))

if __name__ == '__main__':
    path = '/work/21010294/DepthData/OutputSplitAbsoluteVer2/'
    main(path)
