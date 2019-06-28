import cv2
import random
colors=[]
Paths=[]
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'CSRT']
def createTracker(trackerType):
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  return tracker
#Multitracker = cv2.MultiTracker_create()
Multitracker=[]
def CreateMultiTracking(frame, boxes):
     for (x,y,w,h) in boxes:
          BoxPoints=(x, y, w, h)
          colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
          path = [BoxPoints]
          Paths.append(path)
          Tracker = createTracker('CSRT')
          Tracker.init(frame, BoxPoints)
          Multitracker.append(Tracker)

# def Kalaman():
#    kalman=cv2.KalmanFilter(4, 2, 0)
#    kalman.measurementMatrix = cv2.RealScalar(1)
#    kalman.process_noise_cov = cv2.RealScalar(1e-5)
#    kalman.measurement_noise_cov = cv2.RealScalar(1e-1)
#    kalman.error_cov_post = cv2.RealScalar(0.1)
#








