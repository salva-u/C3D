
""" This code has been obtained from https://github.com/ptirupat/AnomalyDetection_CVPR18 
    This is helper code for C3D feature Extractor
    NOT MY WORK
"""

from utils.array_util import *
import parameters as params
import cv2


def get_video_clips(video_path): #cuts into video segments based on frames/sec
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, params.frame_count, params.frame_count) #arr, size=16, stride=16
    return clips, len(frames)


def get_video_frames(video_path): 
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    return frames
