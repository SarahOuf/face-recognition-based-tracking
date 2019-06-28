import cv2
import os
import glob
import numpy as np
import PIL
import pickle as pc
#import Gender as G
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import  Controller as CO

Path = []
Label = []
Path.append("C:/Users/dell/Downloads/dataset/ahmed")
Label.append(0)
Path.append("C:/Users/dell/Downloads/dataset/sarahouf")
Label.append(1)
Path.append("C:/Users/dell/Downloads/dataset/Menna")
Label.append(2)
Path.append("C:/Users/dell/Downloads/dataset/omar")
Label.append(3)
Path.append("C:/Users/dell/Downloads/dataset/Nada")
Label.append(4)
Path.append("C:/Users/dell/Downloads/dataset/SaraAhmed")
Label.append(5)
Path.append("C:/Users/dell/Downloads/dataset/Raghad")
Label.append(6)

###################################################################

#SVM

def SVM_Train(P, L):
    Features, Labels = CO.read_Folder(P, L, 7)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setKernel(cv2.ml.SVM_RBF)
    # svm.setGamma(0.3)
    FT = np.float32(Features)
    LB = np.array(Labels)
    svm.train(FT, cv2.ml.ROW_SAMPLE, LB)
    svm.save("Models/HOGHAAR_SVM.dat")
    print("Done")

def KNN_Train(P, L):
    Features, Labels = CO.read_Folder(P, L, 7)
    FT = np.float32(Features)
    LB = np.array(Labels)
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(FT, LB)
    # predict the response
    file = "Models/HOGHAAR_KNN.pkl"
    with open(file, 'wb') as file:
        pc.dump(knn, file)
    print("DoneKNN")

SVM_Train(Path, Label)
KNN_Train(Path, Label)



