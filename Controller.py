import cv2
import os
import glob
import numpy as np
import PIL
import pickle as pc

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

Features = []
Labels = []
mode =1
#   mode = 0  HOG
#   mode = 1  HAAR
#   mode = 2  HOGHAAR

def Append(L1, L2):
    temp = []
    for element in L1:
        temp.append(element)
    for element in L2:
        temp.append(element)
    return temp
def HOGHAAR(image):

    hOG = Hog(image)
    HAAR = Haar_Cascade(image)
    HAAR = np.ravel(HAAR)
    hogHaar = Append(HAAR, hOG)

    return hogHaar
def Hog(image):
    HOG = cv2.HOGDescriptor()
    image = np.array(image)
    img = cv2.resize(image, (64, 128))
    hog = HOG.compute(img)
    hog = np.ravel(hog)
    return hog
def Horizontal_Haar(slice):
    White = 0
    Black = 0
    for i in range(0, 4):
        for j in range(0, 8):
            White = White+(slice[i, j]*1)
            Black = Black+(slice[i+4, j]*-1)
    diff = Black+White
    return diff
def Vertical_Haar(slice):
    White = 0
    Black = 0
    for i in range(0, 8):
        for j in range(0, 8):
            if i < 2 and i > 5:
                 White = White + slice[i, j]*1
            else :
                 Black = Black+slice[i, j]*-1
    diff= Black+White
    return diff
def Haar_Cascade(image):
    image = np.array(image)
    Image_Slice = []
    NormValue = cv2.norm(image, cv2.NORM_L2)
    image=np.true_divide(image,NormValue)
    #image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (64, 64))
    x = 0
    y = 0
    HaarVector = []
    for height in range(0, 8):
        for width in range(0, 8):
            Slice = image[x:x+8, y:y+8]
            y = y+8
            Image_Slice.append(Slice)
            HaarVector.append(Horizontal_Haar(Slice))
            HaarVector.append(Vertical_Haar(Slice))

        x = x+8
        y = 0
    HaarVector = np.ravel(HaarVector)
    return HaarVector
def read_Folder(paths, labels, count):
    for i in range(0, count):
        Get_Features(paths[i], labels[i])
    return Features, Labels
def Get_Features(path, label):

    imagePath = os.path.join(path, '*g')
    images = glob.glob(imagePath)
    for image in images:

        image = cv2.imread(image)

        if (mode == 1):

            Features.append(Haar_Cascade(image))
            Labels.append(label)

        elif( mode == 0):
            Features.append(Hog(image))
            Labels.append(label)
        else:
            Features.append(HOGHAAR(image))
            Labels.append(label)

def Run(image, runMode):        #runMode = 0 then KNN    runMode = 1 then SVM
    image = np.asarray(image)
    image = PIL.Image.fromarray(image)
    feat = Haar_Cascade(image)
    Gf=Hog(image)
    Gfeat=[]
    feat2 = []
    feat2.append(feat)
    Gfeat.append(Gf)
    var = []
    Gendervalue = 0
    RecognitionValue=0
    if len(feat2) > 0:
       feat2 = np.float32(feat2)
       Gfeat=np.float32(Gfeat)
       with open("Models/Gender_HOG_KNN.pkl", 'rb')as file:
         KNN=pc.load(file)
         Gendervalue = KNN.predict(Gfeat)
         Gendervalue = Gendervalue[0]
       RecSVM = cv2.ml_SVM.load("Models/HAAR_SVM.dat")
       var = RecSVM.predict(feat2)
       RecognitionValue = var[1][0]
       RecognitionValue = int(RecognitionValue)


    #    if runMode == 1:
    #         GenderSVM = cv2.ml_SVM.load("Models/Gender_HOG_SVM.dat")
    #         RecSVM=cv2.ml_SVM.load("Models/HAAR_SVM.dat")
    #         print("SVM")
    #         var = GenderSVM.predict(Gfeat)
    #         Gendervalue = var[1][0]
    #         Gendervalue = int(Gendervalue)
    #         var = RecSVM.predict(feat2)
    #         RecognitionValue = var[1][0]
    #         RecognitionValue = int(RecognitionValue)
    #
    #
    #    else:
    #     with open("Models/Gender_HOG_KNN.pkl", 'rb')as file:
    #         print("KNN")
    #         KNN=pc.load(file)
    #         Gendervalue = KNN.predict(Gfeat)
    #         Gendervalue = Gendervalue[0]
    #     with open("Models/HOG_KNN.pkl", 'rb')as file:
    #         print("KNN")
    #         KNN=pc.load(file)
    #         RecognitionValue = KNN.predict(feat2)
    #         RecognitionValue = RecognitionValue[0]
    return Gendervalue,RecognitionValue



