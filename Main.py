import cv2
import Controller as CO
import Tracking as T

#-------------Paths of haar algorithm -----------------------
face_cascade = cv2.CascadeClassifier('libs/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('libs/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('libs/Nariz.xml')
#-------------ENd Paths of haar algorithm -----------------------

#------------------------Test Videos-----------------------#
# cap = cv2.VideoCapture('Test/cm1.mp4')
cap = cv2.VideoCapture('Test/Vid.mp4')
#------------------------End Test Videos-----------------------#

Genders=["Male", "Female"]
Names=["Galal", "SarahOuf",  "Menna", "Omar",  "Nada", "SaraAhmed", "Raghad"]
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #get videos frames count
print(frames)
boxes=[]
framess = []
GenderAcc=[0,0]
RecAcc=[0,0,0,0,0,0,0]
first=1
for i in range(frames):
    flag, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert image to gray
    #if first frame and have faces Create Multi tracking

    if (first == 1):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces)>0):
            T.CreateMultiTracking(frame, faces)
            first = 0
    elif(i%30==0): #each 30 frame run Face detection and recognition and clear previous trackers and create new trackers to correct any mistake  in the previous framees
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        T.Multitracker.clear()
        T.colors.clear()
        T.Paths.clear()
        T.CreateMultiTracking(frame,faces)

    for i in range(len(T.Multitracker)):
        success, ObjectBox=T.Multitracker[i].update(frame)
        T.Paths[i].append(ObjectBox)

        if(success):
            for Box in T.Paths[i]:
                Center = (int((Box[0] + Box[2]/2)),int((Box[1] + Box[3]/2)))
                cv2.circle(frame, Center, 5, T.colors[i], -1)
        else:
            T.Multitracker.remove(T.Multitracker[i])
            T.colors.remove(T.colors[i])
            T.Paths.remove(T.Paths[i])

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)    #applying box around face
        roi_gray = gray[y:(y+h), x:(x+w)]    #the face gray scale
        Face=frame[y:(y+h), x:(x+w)]
        #### Gender and
        #  Recognition CALL
        G,R=CO.Run(Face,0)
        GenderAcc[G]=GenderAcc[G]+1
        RecAcc[R]=RecAcc[R]+1
        ###############
        cv2.putText(frame, str(Genders[G]),(x, y), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
        cv2.putText(frame, str(Names[R]),(x+140, y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255), 2)
        roi_color = frame[y:(y+h), x:(x+w)]#the face colored to draw
    faces=[]
    cv2.imshow('video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
print(GenderAcc)
print(RecAcc)
cap.release()
cv2.destroyAllWindows()
