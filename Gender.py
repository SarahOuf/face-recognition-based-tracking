import cv2
import numpy as np
import  Controller as CO
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import  Controller as CO
import pickle as pc



Path = []
Label = []
Features = []
Labels = []
Path.append("D:/male")
Label.append(0)
Path.append("D:/female")
Label.append(1)

def Train_SVM():
    Features, Labels = CO.read_Folder(Path,Label,2)
    #HR.read_Folder(Path, Label,2)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)

    FT = np.float32(Features)
    LB = np.array(Labels)
    svm.train(FT, cv2.ml.ROW_SAMPLE, LB)
    svm.save("Models/Gender_HOG_SVM.dat")
    print("Done Gender")

#Train_SVM()


def KNN_Train(P, L):
    Features, Labels = CO.read_Folder(P, L, 2)
    FT = np.float32(Features)
    LB = np.array(Labels)
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(FT, LB)
    # predict the response
    file = "Models/Gender_HOG_KNN.pkl"
    with open(file, 'wb') as file:
        pc.dump(knn, file)
    print("DoneKNN")
KNN_Train(Path, Label)

