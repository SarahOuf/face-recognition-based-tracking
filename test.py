from __future__ import print_function
import numpy as np
import cv2
import pickle
import os
import sklearn.preprocessing
import sys
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker
# Set video to load
face_cascade=cv2.CascadeClassifier('libs/haarcascade_frontalface_default.xml')
videoPath = "Test/Vid.MOV"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(1,1),maxSize=(60,60))


## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
for (x, y, w, h) in faces:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner

  bboxes.append((x,y,w,h))
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")


#print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
trackerType = "CSRT"

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)
# Process video and track objects
i=0
while cap.isOpened():


  success, frame = cap.read()
  if not success:
    break

  # get updated location of objects in subsequent frames
  i=i+1
  if i%5==0:
   print(i)
   success, boxes = multiTracker.update(frame)

  # draw tracked objects
   for i, newbox in enumerate(boxes):
     print(i)
     p1 = (int(newbox[0]), int(newbox[1]))
     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

  # show frame
   cv2.imshow('MultiTracker', frame)
   cv2.waitKey(0)


  # quit on ESC button


