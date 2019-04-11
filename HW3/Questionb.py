import numpy as np
from PIL import Image
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

plt.style.use('ggplot')

def read_faces(filepath='yalefaces'):
    X_origin = []
    X_flattlen = []
    y = []
    for (dirpath, dirnames, filenames) in os.walk(filepath):
        for filename in filenames:
            if filename[0]=='s':
                y.append(filename.split('.')[0])
                img = np.array(Image.open(dirpath+'/'+filename))
                X_origin.append(img)
                X_flattlen.append(img.flatten())
    # normalize
    return np.array(X_origin)/255.0, np.array(X_flattlen)/255.0 ,np.array(y)

def encode(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    return le

def PCA_scratch(A):
    M = np.mean(A.T,axis=1)
    Center = A-M
    V = np.cov(Center.T)
    values,vectors = np.linalg.eig(V)
    P = vectors.T.dot(Center.T)
    return P


def PCA_all(X_flatten, num_eig=64):
    pca_face = PCA(num_eig)
    X_proj = pca_face.fit_transform(X_flatten)
    # print(np.cumsum(pca_face.explained_variance_ratio_))
    # plt.plot(np.cumsum(pca_face.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    # plt.show()
    return X_proj, pca_face


def classify(X_train_proj, X_test_proj,y_train, y_test):
    logisticRegr = LogisticRegression(
        C=1e5, solver='lbfgs', multi_class='multinomial')
    clf_svm = SVC(kernel='linear', C=5)
    clf_rf =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                         max_features='auto', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, n_estimators=100,
                                         oob_score=False, random_state=0, verbose=0, warm_start=False)
    logistc_score = cross_val_score(logisticRegr, X_train_proj, y_train, cv=5)
    svm_score = cross_val_score(clf_svm, X_train_proj, y_train, cv=5)
    rf_score = cross_val_score(clf_rf, X_train_proj, y_train, cv=5)
    print("Logistic Accuracy: %0.4f (+/- %0.2f)" %
          (logistc_score.mean(), logistc_score.std() * 2))
    print("SVM Accuracy: %0.4f (+/- %0.2f)" %
          (svm_score.mean(), svm_score.std() * 2))
    print("Random Forest Accuracy: %0.4f (+/- %0.2f)" %
          (rf_score.mean(), rf_score.std() * 2))

    




def main():

    X_origin, X_flatten, label = read_faces() 
    le = encode(label)
    y = le.transform(label)
    y = np.expand_dims(y,axis=1)
    data = np.hstack((X_flatten,y))
    data = data[data[:,-1].argsort()]
    X_flatten = data[:,:-1]
    y = data[:,-1]
    #Perform PCA
    X_train, X_test, y_train, y_test = train_test_split(
                                        X_flatten, y, test_size=0.1, random_state=42)
    """
    for i in [1,2,3,10,20,30,40,50]:
        X_train_proj, pca_face = PCA_all(X_train, i)
        X_test_proj = pca_face.transform(X_test)
        print("Number of Principal Components %d" %i)
        classify(X_train_proj, X_test_proj,y_train,y_test)

    """
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, hspace=0.4, wspace=0.05)
    X_train_proj, pca_face = PCA_all(X_flatten,64)
    eigenfaces = pca_face.components_.reshape((64, 243, 320))
    for i in range(10):
        ax = fig.add_subplot(4,3,i+1, xticks=[], yticks=[])
        title = 'Eigenface %d' % (i+1)
        ax.set_title(title)
        ax.imshow(eigenfaces[i],cmap='gray',interpolation='bicubic')
        
    plt.show()
    #print(logisticRegr.score(X_test_proj, y_test))

    """
    fig = plt.figure(figsize=(6,6)) 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i_index,i in enumerate([1,10,20,30,40,50,60]):
        X_proj, pca_face = PCA_all(X_flatten,i)
        X_inv_proj = pca_face.inverse_transform(X_proj)
        X_proj_img = np.reshape(X_inv_proj, (166, 243, 320))
        for j_index,j in enumerate([0,15,165]):
            # sthash.3WmkRwVH.dpuf
            ax = fig.add_subplot(3, 7, i_index+j_index*7+1, xticks=[], yticks=[]) 
            ax.imshow(X_proj_img[j], cmap='gray', interpolation='bicubic')
    plt.show()
    """

    #plt.plot(np.arange(64),X_proj)
    #plt.show()
    # plot the faces, each image is 64 by 64 pixels 
    """
    fig = plt.figure(figsize=(8, 8))
    # sthash.3WmkRwVH.dpuf
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(10): 
        ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[]) 
        ax.imshow(np.reshape(pca_oliv.components_[i,:], (243,320)), cmap=plt.cm.bone, interpolation='nearest')
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    # sthash.3WmkRwVH.dpuf
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(166):
        # sthash.3WmkRwVH.dpuf
        ax = fig.add_subplot(21, 8, i+1, xticks=[], yticks=[]) 
        ax.imshow(np.reshape(X[i],(243,320)), cmap=plt.cm.bone, interpolation='nearest')
    plt.show()
    """

    """
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    """
if __name__ == "__main__":
    main()
