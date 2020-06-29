"""
This code is based on the work of https://github.com/ptirupat/AnomalyDetection_CVPR18 it has been modified to extract features for many videos
in their work they hav referenced the followigng repository https://github.com/WaqasSultani/AnomalyDetectionblob/master/Testing_Anomaly_Detector_public.py
"""
#MAIN FILE: This code is used to extract features to reproduce the baseline work
#CHANGE the input and output file paths and run the file 
import os
from c3d import *
import collections
import configuration as cfg
import time
import csv
import os
from os import listdir
import numpy as np
from datetime import datetime
import shutil
from utils.video_util import *
from utils.array_util import *
#get Files
videosPath = cfg.input_folder #directory where video files are stored 
allFiles= listdir(videosPath)
allFiles.sort()
print(allFiles) 

outputPath = cfg.output_folder #directory where you want to save the Extracted Features
def extractVideoFeatures():
    startTime = datetime.now()
        # build models
    feature_extractor = c3d_feature_extractor()
    print("Models initialized")
    for i in range(len(allFiles)):
         videoP = videosPath + allFiles[i]
         runExtractor(videoP, feature_extractor)
    print("Total Time taken: " , str(datetime.now() - startTime))

def runExtractor(filePath, featEx):
    videoName = os.path.basename(filePath).split('.')[0] #get File name
    # read video
    video_clips, num_frames = get_video_clips(filePath)
    print("Number of clips in the video : ", len(video_clips), " ", num_frames)


    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < params.frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = featEx.predict(clip)[0] #4096 features are collected for each video clip 
        rgb_features.append(rgb_feature)
        print("Processed clip : ", i)
    rgb_features = np.array(rgb_features)

    # bag features
    rgb_feature_bag = interpolate(rgb_features, params.features_per_bag) # takes the features and puts them into 32 bags
    print("Length of this bag ", len(rgb_feature_bag))
    #print(rgb_feature_feature_bag.squeeze())
    np.savetxt(outputPath+videoName+'.txt', rgb_feature_bag.squeeze())



extractVideoFeatures()



