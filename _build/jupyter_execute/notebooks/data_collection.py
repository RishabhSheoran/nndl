#!/usr/bin/env python
# coding: utf-8

# # Data Collection
# 
# For data collection, we considered a lot of sources initially.
# * Facebook
# * Twitter
# * YouTube
# * Google images
# * Etc.
# 
# 
# However, eventually we decided to choose one source only to ensure constant aspect ratio, structure and resolution in the images. We chose **YouTube** as it had the largest pool of continuous videos, and the longest samples from official sources (channels) to avoid any copyright issues.
# 
# 
# We used:
# 
# 1.	Open-source library ```PyTube``` to download the video
# 2.	Custom class ```FrameExtractor``` to sample the video
# 3.	Resize function to reinforce same dimensions of images
# 
# The data is collected by downloading YouTube videos using the package ```pytube``` and sampled using a custom class ```FrameExtractor```.
# 

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install pytube')
get_ipython().system('{sys.executable} -m pip install opencv-python')


# In[27]:


from pytube import YouTube
import os
import shutil
import math
import datetime
import matplotlib.pyplot as plt
import cv2


# ## class ```FrameExtractor``` 
# 
# Class used for extracting frames from a video file.
# 
# Functions:
# 
# 1. get_video_duration - returns the length of the video
# 2. get_n_images - returns the number of images given a particular sampling rate
# 3. extract_frames - extracts and stores the frames from a given downloaded video

# In[28]:


class FrameExtractor():
 
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')
        return duration
        
    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} frames resulted in {n_images} images.')
        return n_images
        
    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the directory: {dest_path}')
        
        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():
            
            success,image = self.vid_cap.read() 
            
            if not success:
                break
            
            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)  
                img_cnt += 1
                
            frame_cnt += 1
        
        self.vid_cap.release()
        cv2.destroyAllWindows()


# ## Downloading and sampling the videos
# 
# The links to the Youtube videos are obtained from a file called ```links.txt``` which is manually fed in.
# We have chosen all videos from Season 5 of F.R.I.E.N.D.S.
# 
# We also extract:
# 1. ```last_video_index``` : index of the last downloaded video
# 2. ``` last_sampled_index``` :  index of the last sampled video 
# 
# This ensures no work is redone when we add more videos to increase the size of the dataset.

# In[29]:


with open('../data/other/links.txt') as file:
    urls = file.readlines()

path_videos = "../data/videos/"    
path_images = "../data/raw_images/"

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

image_dir = sorted_alphanumeric(os.listdir(path_images))
video_dir = sorted_alphanumeric(os.listdir(path_videos))

last_video_index = 0
last_sampled_index = 0

try: last_video_index = int(video_dir[-1].split('_')[0])
except: pass

try: last_sampled_index = int(image_dir[-1].split('_')[0])
except: pass
    
print("Last downloaded video index:", last_video_index)
print("Last sampled video index:", last_sampled_index)


# ```image_count``` keeps track of the number of images sampled and ```total_duration``` keeps track of the length of the videos sampled.
# 
# 
# 1. The video is downloaded using pytube's ```Youtube``` class and the URL from ```links.txt``` 
# 2. The video is sampled using the ```FrameExtractor``` class from above.
# 3. Relevant information is extracted and displayed.

# In[31]:


image_count = len(image_dir)
total_duration = 0

for index, video_url in enumerate(urls):
    video_url = video_url.strip()
    
    # Extracting code from the URL
    code = video_url[video_url.index('=')+1:]
    image_name = '{}_{}'.format(index, code)
    print(image_name)
    
    duration, n_images = 0, 0
    
    # Checking if it has already been downloaded
    if index > last_video_index or last_video_index == 0: 
        # Downloading the video
        yt = YouTube(video_url)
        yt = yt.streams.filter(file_extension='mp4', res='360p').first()
        fps = yt.fps
        video = yt.download(path_videos, filename=image_name+".mp4")
        print("Downloaded video {}_{}".format(index, code))
    else: print("Already downloaded.")
    
    
    # Checking if it has already been sampled
    if index > last_sampled_index or last_sampled_index == 0:
        # Extracting frames from the video per second
        fe = FrameExtractor(video)
        duration = fe.get_video_duration()
        n_images = fe.get_n_images(every_x_frame=fps)
        fe.extract_frames(every_x_frame=fps, 
                          img_name=image_name, 
                          dest_path=path_images)
        print("Sampled video {}_{}".format(index, code))
    else: print("Already sampled.")
        
    
    total_duration += duration
    image_count += n_images
    
    print()
    
print("Total duration of the videos =", total_duration)
print("Total number of images =", image_count)


# ## Resizing all of the images to a fixed size

# In[17]:


from PIL import Image

f = r'../data/raw_images/'
width, height = 256, 144

for file in os.listdir(f):
    if file.startswith('.'): continue
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((width, height))
    img.save(f_img)

print("Reshaped all images to ({}, {})".format(width, height))


# In[ ]:




