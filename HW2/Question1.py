def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def load_data():
    data = pd.read_csv('hw2_question1.csv', names=[
                       'Clump', 'CellSize', 'CellShape', 'Adhesion', 
                       'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli',
                       'Mitoses', 'Class'])
    return data

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                     random_state=0, stratify=y)

    return X_train,y_train,X_test,y_test

# Function to perform training with giniIndex.


def train_using_gini(X_train, y_train, maxdepth=10, min_impurity_split=0):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      max_depth=maxdepth,
                                      min_impurity_split=min_impurity_split)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.


def tarin_using_entropy(X_train, y_train, maxdepth=10, min_impurity_split=0):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=maxdepth,
        min_impurity_split=min_impurity_split)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def main():
    df = load_data()
    X, y = df.values[:,:-1] ,df['Class'].values
    X_train, y_train, X_test, y_test = split_data(X, y)
    #print(y_train)
    #print(Counter(y_test))
    """
    entropy
    """
    res = []
    x_value = []
    for i in range(0,21):
        i = i*0.05
        x_value.append(i)
        clf_gini = train_using_gini(X_train, y_train,10,i)
        y_pred_gini = clf_gini.predict(X_test)
        acc_entro = accuracy_score(y_test, y_pred_gini)
        res.append(acc_entro)
        print("Entropy Accuracy: ", acc_entro)

    plt.title('Gini Pre-Pruning')
    plt.xlabel('Min Gini')
    plt.ylabel('Accuray')
    plt.plot(x_value,res)
    plt.show()
    print(res)

    # res = []
    # for i in range(1,11):
    #     clf_entropy = tarin_using_entropy(X_train,y_train,i)
    #     y_pred_entropy = clf_entropy.predict(X_test)
    #     acc_entro = accuracy_score(y_test, y_pred_entropy)
    #     print("Entropy Accuracy: ", acc_entro)
    #     res.append(acc_entro)
    # print(res)
    # plt.title('Entropy Accuracy')
    # plt.xlabel('Tree Nodes')
    # plt.ylabel('Accuray')
    # plt.xticks(np.arange(1, 11, 1))
    # plt.plot(np.arange(1,11,1),res)

    # plt.show()

    """
    Gini
    """
    # res = []
    # for i in range(1,11):
    #     clf_gini = train_using_gini(X_train,y_train,i)
    #     y_pred_gini = clf_gini.predict(X_test)
    #     acc_gini = accuracy_score(y_test, y_pred_gini)
    #     print("Gini Accuracy: ", acc_gini)
    #     res.append(acc_gini)
    
    # plt.title('Gini Accuracy')
    # plt.xlabel('Tree Nodes')
    # plt.ylabel('Accuray')
    # plt.xticks(np.arange(1, 11, 1))
    # plt.plot(np.arange(1, 11, 1), res)

    # plt.show()
   #print(df['Class'].value_counts())

if __name__ == "__main__":
    main()
