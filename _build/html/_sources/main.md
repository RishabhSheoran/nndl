# Introduction

## Group Members

- Krishna Kumar Hariprasannan (A0229171H, krishnakh@u.nus.edu)
- Rishabh Sheoran (A0236146H, rishabh.sheoran@u.nus.edu)
- Saisha Ajay Chhabria (A0244449X, saisha.chhabria@u.nus.edu)

## Motivation

Online video streaming platforms often face the problem of computing cast metadata for a given movie or television show. This is because the cast information provided by production houses are either incomplete or not available (for older movies). Cast metadata has numerous usages that include search (for example, “movies by Chadwick Boseman”), recommendation based on past ratings, etc. In addition, some online video streaming systems also provide a feature that identifies and tracks characters in a scene and makes this information available to users upon interaction with an interface element (Button click, Mouse hover, etc.). An example of such a system can be seen in Amazon Prime Video and the X-Ray feature in its video player.

## Description

The aforementioned problem can be translated into a multi-label classification problem wherein each label describes the presence/absence of a known character in a video. For a given movie video, the model is expected to predict if a known character (one of the classes) is present in the video and in a particular scene. As a video is merely a sequence of images, one can draw parallels between making the prediction in videos to that of images. Hence, we can consider an image dataset for solving the problem and the solution is expected to be applicable for videos by extension.

## Proposed Solution

Our project aims to explore different classes of neural networks (MLP, CNN, RNN, ANN) as a solution to the aforementioned problem of facial recognition. We use an image dataset that is scraped and sampled from images and videos of the TV show - ‘Friends’ to train and test our models (identify the show’s main characters in those images). We analyze and compare the performances of those neural networks classes as well as interpret the reasons behind the performance differences. We also explore and recommend improvements to those baseline models to enhance their classification performance. Finally, a discussion is made on the performance differences between considered networks.
