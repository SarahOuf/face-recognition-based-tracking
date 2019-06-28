# face-recognition-based-tracking

The application tracks different humans in a video by detecting their faces, some of the faces are saved therefore they are recognized during tracking. Using Computer Vision methods, the application extracts the features of the face and uses a classifier to perform the recognition, For each detected face it is tracked throughout the video.

## Installation Guide

To run the project you need to install the following:
* cv2
* numpy
* pickle
* PIL
* sklearn
* glob

## Models

The face recognition models was trained with two different metrics: a SVM and a KNN classifiers and for feature extraction:
HAAR, HOG and both where used for training with the SVM and the KNN classifiers, those are already available in the models folder.

For the gender classfication HOG feature extractor was used, it was also trained on a SVM and KNN classifier.
The models can be found [here](https://drive.google.com/open?id=1ky4H2r1SVB3lc2_rFnAtr2yUBLF6eLnT).

## Runing the project

To run the project on the existing data and video just run the Main file.

To train a face recognition model run the HAAR file.

To train a gender classifier model run the Gender file.
