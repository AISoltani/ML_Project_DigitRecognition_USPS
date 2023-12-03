# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
import itertools
import os, re
import math
import cv2

# tools for preprocessing on input data...resize and scaling

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat_tf(dataset, labels):
    image_size = 28
    num_labels = 10
    num_channels = 1
    dataset = dataset.reshape((-1, image_size, image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

# Preprocessing with change input data size to fixed optimized size

def process_usps_data(patch):
    path_to_data = patch
    img_list = os.listdir(path_to_data)
    sz = (28,28)
    validation_usps = []
    validation_usps_label = []
    for i in range(10):
        label_data = path_to_data + str(i) + '/'
        img_list = os.listdir(label_data)
        for name in img_list:
            if '.png' in name:
                img = cv2.imread(label_data+name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = resize_and_scale(img, sz, 255)
                validation_usps.append(resized_img.flatten())
                validation_usps_label.append(i)
    validation_usps = np.array(validation_usps)
    validation_usps_label= np.array(validation_usps_label)
    return validation_usps, validation_usps_label

# Numeric confusion matrix function to describes the performance of a classifier

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Normalizing input data...

def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)

    ## Automatic:
    # X_cov = np.cov(X_std.T)

    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0] - 1)

    return X_std, X_mean, X_cov

#################################################################################################################

#KNN

def Knn_Classifier(x_train,y_train,x_test,y_test):

#Find besk K for KNN classifier

    accuracies = []
    kVals = range(1, 30, 2)
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`

        knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=k, weights='distance')
        knn.fit(x_train, y_train)
        y_te_pred = knn.predict(x_test)
        acc = accuracy_score(y_test, y_te_pred)

        # evaluate the model and update the accuracies list

        accuracies.append(acc)
        print("k=%d, accuracy=%.2f%%" % (k, acc * 100))

    bestk = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on test data" % (kVals[bestk], accuracies[bestk] * 100))

    # Create KNN model object to classification

    knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=kVals[bestk], weights='distance')
    knn.fit(usps_dataset_train, usps_labels_train)

    # Evaluate prediction, Stimate scores...F1 and...

    y_te_pred = knn.predict(usps_dataset_test)
    acc = accuracy_score(usps_labels_test, y_te_pred)
    prec = precision_score(usps_labels_test, y_te_pred, average="macro")
    rec = recall_score(usps_labels_test, y_te_pred, average="macro")
    f1 = f1_score(usps_labels_test, y_te_pred, average="macro")
    print("Acc: %3.5f, P: %3.5f, R: %3.5f, F1: %3.5f" % (acc * 100, prec * 100, rec * 100, f1 * 100))
    a = usps_labels_test
    b = y_te_pred

    # Create confusion matrix

    plt.figure()
    cfs = confusion_matrix(usps_labels_test.argmax(axis=1), y_te_pred.argmax(axis=1))
    le = preprocessing.LabelEncoder()
    enc = le.fit(usps_labels_test.argmax(axis=1))
    class_names = enc.classes_
    plot_confusion_matrix(cfs, classes=class_names, title='KNN Confusion matrix, without normalization')
    print("Total calssification report:\n")
    print(classification_report(usps_labels_test, y_te_pred))
    plt.figure()
    cfs2 = confusion_matrix(usps_labels_test.argmax(axis=1), y_te_pred.argmax(axis=1))
    sns.heatmap(cfs2, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nKNN Classifier')
    ylabel('True')
    xlabel('Predicted label')
    plt.show()


##########################################################################################################

# PCA Classifier

def PCA_Classifier(x_train,y_train,x_test,y_test,y_valid,y_test2,y_test3):

    data = x_train

    # Compute Covariance

    data_cov = np.cov(data, rowvar=False)

    # Stimate EigenValues/Vectores

    e_vals, e_vecs = LA.eigh(data_cov)

    # Sorting EigenValue/Vectores

    idx = e_vals.argsort()[::-1]
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    # Constructor

    def Dev_PCA(data, eigv):
        y = np.matmul(data, eigv)
        return y

    # Find best d with (POV=95%)

    def find_suitable_d(p, e_vals):
        ps = np.sum(e_vals)
        i = 0
        pov = 0
        a = 0
        while (pov < p):
            a += e_vals[i]
            pov = a / ps
            i += 1
        return i

    # Reconstruct PCA

    def Rec_PCA(data_new, eigv):
        y = np.matmul(data_new, np.transpose(eigv))
        return y

    # Splitor

    def split_feature(e_vecs, d):
        new_e_vecs = e_vecs[:, 0:d]
        return new_e_vecs

    # Start to finding suitable d

    x = np.arange(1, data.shape[1] + 1)
    d = find_suitable_d(0.95, e_vals)
    print("Best Suitable d For Component PCA Is With (POV=95%) : ", d)

#    PDA_data = Dev_PCA(data, split_feature(e_vecs, d))
#    rec_data = Rec_PCA(PDA_data, split_feature(e_vecs, d))

    # Normalizing mean, cov, data

    #Normalizing...

    X_std, X_mean, X_cov = get_normed_mean_cov(data)
    X_std_validation, _, _ = get_normed_mean_cov(x_test)

    #PCA component cons-reconstractor...

    pca = PCA(n_components=d, whiten=True)
    pca.fit(X_std)
    X_red = pca.transform(X_std)

    usps_labels_train = y_valid
    usps_labels_test = y_test2
    linclass2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)

    linclass2.fit(X_red, y_valid)

    X_red_validation = pca.transform(X_std_validation)
    yhat_validation = linclass2.predict(X_red_validation)

    # Create confusion matrix-color map


    plt.figure()
    pca_cm = confusion_matrix(y_test2, yhat_validation)
    sns.heatmap(pca_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nPCA + SVC')
    ylabel('True')
    xlabel('Predicted label')
    total = pca_cm.sum(axis=None)
    correct = pca_cm.diagonal().sum()
    print(d," Component PCA Accuracy: %0.2f %%" % (100.0 * correct / total))

    #Numeric confusion matrix and stimating score,f1,accuracy...

    plt.figure()
    cfs3 = confusion_matrix(y_test2, yhat_validation)
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test2)
    class_names = enc.classes_
    plot_confusion_matrix(cfs3, classes=class_names, title='PCA Confusion matrix, without normalization')
    print("Total calssification report:\n")
    print(classification_report(y_test2, yhat_validation))
    plt.show()

##########################################################################################################

#LDA

def LDA_Classifier(x_train,y_train,x_test,y_test,y_valid,y_test2):

    #Normalizing input data...

    X_std, X_mean, X_cov = get_normed_mean_cov(x_train)
    X_std_validation, _, _ = get_normed_mean_cov(x_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_std)
    X_test = sc.transform(X_std_validation)

    lda = LDA(n_components=500)
    X_train = lda.fit_transform(X_train, y_valid)
    X_test = lda.transform(X_test)

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_valid)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test2, y_pred)
    print(cm)
    print("LDA Accuracy : " + str(accuracy_score(y_test2, y_pred)))

    plt.figure()
    sns.heatmap(cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nLDA + RFC')
    ylabel('True')
    xlabel('Predicted label')
    total = cm.sum(axis=None)
    correct = cm.diagonal().sum()
    print("LDA Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    cfs4 = confusion_matrix(y_test2, y_pred)
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test2)
    class_names = enc.classes_
    plot_confusion_matrix(cfs4, classes=class_names, title='LDA Confusion matrix, without normalization')
    print("Total calssification report:\n")
    print(classification_report(y_test2, y_pred))

    plt.show()


###########################################################################################################

# Baysian with gussian distrbution

def Bayesian_Classifier(x_train,y_train,x_test,y_test,y_valid,y_test2):
    fixed_x_test = x_test
    fixed_y_test = y_test2
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_valid)
    x_test = fixed_x_test
    y_test = fixed_y_test
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    predicted = clf.predict(fixed_x_test)
    expected = y_test
    matches = (predicted == expected)
  #  accuracy = (matches.sum() / float(len(matches))) * 100
    print(metrics.classification_report(expected, predicted))


    plt.figure()
    baysian = confusion_matrix(expected, predicted)
    sns.heatmap(baysian, square=True, cmap='inferno')
    title('Confusion Matrix:\nBayesian (Gussian) ')
    ylabel('True')
    xlabel('Predicted label')
    total = baysian.sum(axis=None)
    correct = baysian.diagonal().sum()
    print("Bayesian With Gussian Distrbution Accuracy Is: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    cfs5 = confusion_matrix(expected, predicted)
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test2)
    class_names = enc.classes_
    plot_confusion_matrix(cfs5, classes=class_names, title='Bayesian( Gussian ) Confusion matrix, without normalization')

    plt.show()


##########################################################################################################################

# QDA Classifier

def QDA_Classifier(x_train,x_test,y_test,y_valid):
    qda = QDA()
    qda.fit(x_train, y_valid)
    y_te_pred = qda.predict(x_test)

    acc = accuracy_score(y_test, y_te_pred)
    # print(classification_report(usps_labels_test.argmax(axis=1), y_te_pred))
    print(metrics.classification_report(y_test, y_te_pred))
    plt.figure()
    cfs2 = confusion_matrix(y_test, y_te_pred)
    sns.heatmap(cfs2, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nQDA Classifier')
    ylabel('True')
    xlabel('Predicted label')

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(cfs2, classes=class_names, title='QDA Confusion matrix, without normalization')
    total = cfs2.sum(axis=None)
    correct = cfs2.diagonal().sum()
    print("QDA Accuracy Is: %0.2f %%" % (100.0 * correct / total))
    plt.show()

##########################################################################################################

# Logistic Regression Classifier

def Logistic_Regression_Classifier(x_train,x_test,y_test,y_valid):
    lrc = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    lrc.fit(x_train, y_valid)

    predicted = lrc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    lrc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(lrc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nLRC')
    ylabel('True')
    xlabel('Predicted label')
    total = lrc_cm.sum(axis=None)
    correct = lrc_cm.diagonal().sum()
    print("LRC Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(lrc_cm, classes=class_names, title='LRC Confusion matrix, without normalization')
    plt.show()

###########################################################################################################

# Random Forest Classifier

def Random_Forest_Classifier(x_train,x_test,y_test,y_valid):
    rfc = RandomForestClassifier(max_depth=15, n_estimators=20, max_features=5)
    rfc.fit(x_train, y_valid)

    predicted = rfc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    rfc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(rfc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nRFC')
    ylabel('True')
    xlabel('Predicted label')
    total = rfc_cm.sum(axis=None)
    correct = rfc_cm.diagonal().sum()
    print("RFC Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(rfc_cm, classes=class_names, title='RFC Confusion matrix, without normalization')
    plt.show()
    
##################################################################################################################

# Multilayer Prceptron Classifier

def Multilayer_Prceptron_Classifier(x_train,x_test,y_test,y_valid):
    mlp = MLPClassifier(alpha=1)
    mlp.fit(x_train, y_valid)

    predicted = mlp.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    mlp_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(mlp_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nMLP')
    ylabel('True')
    xlabel('Predicted label')
    total = mlp_cm.sum(axis=None)
    correct = mlp_cm.diagonal().sum()
    print("MLP Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(mlp_cm, classes=class_names, title='MLP Confusion Matrix, without normalization')
    plt.show()

########################################################################################################################

# Ada Boost Classifier

def Ada_Boost_Classifier(x_train,x_test,y_test,y_valid):
    abc = AdaBoostClassifier()
    abc.fit(x_train, y_valid)

    predicted = abc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    abc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(abc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nAda Boost Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = abc_cm.sum(axis=None)
    correct = abc_cm.diagonal().sum()
    print("Ada Boost Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(abc_cm, classes=class_names, title='Ada Boost Classifier Confusion Matrix, without normalization')
    plt.show()

############################################################################################################################

# Decision Tree Classifier Classifier

def Decision_Tree_Classifier(x_train,x_test,y_test,y_valid):
    dtc = DTC(max_depth=50)
    dtc.fit(x_train, y_valid)

    predicted = dtc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    dtc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(dtc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nDecision Tree Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = dtc_cm.sum(axis=None)
    correct = dtc_cm.diagonal().sum()
    print("Decision Tree Classifier Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))



    print("injaaaaaaaaaaaaaaaaa1:",confusion_matrix)
    print("injaaaaaaaaaaaaaaaaa2:",y_test)
    print("injaaaaaaaaaaaaaaaaa3:",predicted)
    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(dtc_cm, classes=class_names, title='Decision Tree_Classifier Confusion Matrix, without normalization')
    plt.show()

################################################################################################################################

# Gaussian Process Classifier

def Gaussian_Process_Classifier(x_train,x_test,y_test,y_valid):
    gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    gpc.fit(x_train, y_valid)

    predicted = gpc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    gpc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(gpc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nGaussian Process Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = gpc_cm.sum(axis=None)
    correct = gpc_cm.diagonal().sum()
    print("Gaussian Process Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(gpc_cm, classes=class_names, title='Gaussian Process Classifier Confusion Matrix, without normalization')
    plt.show()

####################################################################################################################################

# Support Vector Machine classifier

def Support_Vector_Machine(x_train,x_test,y_test,y_valid):
    svm = SVC(gamma=0.001)
    svm.fit(x_train, y_valid)

    predicted = svm.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    svm_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(svm_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nSupport Vector Machine Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = svm_cm.sum(axis=None)
    correct = svm_cm.diagonal().sum()
    print("Support Vector Machine Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(svm_cm, classes=class_names, title='Support Vector MachineClassifier Confusion Matrix, without normalization')
    plt.show()


#########################################################################################################################################

# HOG + Linear SVM Classifier

def HOG_L_SVM(x_train,x_test,y_test2,y_valid):
    features = np.array(x_train, 'int16')
    labels = np.array(y_valid, 'int')
    gg = np.array(x_test, 'int16')

    list_hog_fd1 = []
    for feature in gg:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd1.append(fd)
    hog_features1 = np.array(list_hog_fd1, 'float64')

    # Extract the hog features
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')

    lsvm = LinearSVC()
    labels1 = np.array(y_test2, 'int')
    lsvm.fit(hog_features, validation_usps_label)
    y_pred = lsvm.predict(hog_features1)
    print(metrics.classification_report(labels1, y_pred))
    plt.figure()
    lsvm_cm = confusion_matrix(labels1, y_pred)
    sns.heatmap(lsvm_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nHOG + Linear SVM Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = lsvm_cm.sum(axis=None)
    correct = lsvm_cm.diagonal().sum()
    print("HOG + Linear SVM Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(test_usps_label)
    class_names = enc.classes_
    plot_confusion_matrix(lsvm_cm, classes=class_names, title='HOG + Linear SVM Classifier Confusion Matrix, without normalization')
    plt.show()


#####################################################################################################################################

# Stochastic Gradient Descent

def Stochastic_Gradient_Descent(x_train,x_test,y_test,y_valid):

    gpc = SGDClassifier()
    gpc.fit(x_train, y_valid)

    predicted = gpc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    gpc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(gpc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nStochastic Gradient Descent Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = gpc_cm.sum(axis=None)
    correct = gpc_cm.diagonal().sum()
    print("Stochastic Gradient Descent Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(gpc_cm, classes=class_names, title='Stochastic Gradient Descent Classifier\n Confusion Matrix, without normalization')
    plt.show()

################################################################################################################################################

# Main fnction

def PRProject_Main(n):


    if n == 1:
        Knn_Classifier(usps_dataset_train, usps_labels_train, usps_dataset_test, usps_labels_test)

    elif n == 2:
        PCA_Classifier(usps_dataset_train, usps_labels_train, usps_dataset_test, usps_labels_test,
                       validation_usps_label, test_usps_label)

    elif n == 3:
        LDA_Classifier(usps_dataset_train, usps_labels_train, usps_dataset_test, usps_labels_test,
                       validation_usps_label, test_usps_label)

    elif n==4:
        Bayesian_Classifier(usps_dataset_train, usps_labels_train, usps_dataset_test, usps_labels_test,
                            validation_usps_label, test_usps_label)
    elif n==5:
        QDA_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==6:
        Logistic_Regression_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==7:
        Random_Forest_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==8:
        Multilayer_Prceptron_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==9:
        Ada_Boost_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==10:
        Decision_Tree_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==11:
        Gaussian_Process_Classifier(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==12:
        Support_Vector_Machine(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==13:
        HOG_L_SVM(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

    elif n==14:
        Stochastic_Gradient_Descent(usps_dataset_train, usps_dataset_test, test_usps_label, validation_usps_label)

if __name__ == "__main__":
    # Reading USPS data

    # Reading train data for preprocessing...

    validation_usps, validation_usps_label = process_usps_data("USPS-data/Numerals/")
    usps_data = np.reshape(validation_usps, (20000, 28, 28))
    usps_dataset_train, usps_labels_train = reformat_tf(usps_data, validation_usps_label)
    usps_dataset_train = np.reshape(usps_dataset_train, (20000, 784))

    # Reading test data for preprocessing...

    test_usps, test_usps_label = process_usps_data("USPS-data/Test/")
    usps_data = np.reshape(test_usps, (1500, 28, 28))
    usps_dataset_test, usps_labels_test = reformat_tf(usps_data, test_usps_label)
    usps_dataset_test = np.reshape(usps_dataset_test, (1500, 784))
    print("Handwritten Digit Recognition Project\n\nTo Choose your Method  Run, First You Need To Install "
          "Required Libraries\n\n")
    print("1.Knn Classifier\n"
          "2.PCA Classifier\n"
          "3.LDA Classifier\n"
          "4.Bayesian Classifier\n"
          "5.QDA Classifier\n"
          "6.Logistic Regression Classifier\n"
          "7.Random Forest Classifier\n"
          "8.Multilayer Prceptron Classifier\n"
          "9.Ada Boost Classifier\n"
          "10.Decision Tree Classifier\n"
          "11.Gaussian Process Classifier\n"
          "12.Support Vector Machine\n"
          "13.HOG With LSVM\n"
          "14.Stochastic Gradient Descent\n"
          "Choosing The Number Of Method : ")
    n = int(input())
    PRProject_Main(n)
