import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from libsvm.python.svmutil import *
import time
import scipy
plt.style.use('ggplot')
# The feature numbers that need to be transformed in the data
t_features = [1, 6, 7, 13, 14, 15, 25, 28]
def data_preprocess():
    df = pd.read_csv('hw2_question3.csv',header=None)
    for i in t_features:
        df[i] = df[i].apply(lambda x: [0,0,1] if x==1 else([0,1,0] if x==0 else [1,0,0]))
    #print(df.head(5))
    X = df[0].values[:,None]
    tmp = []

    for i in range(1,30):
        if i in t_features:
            tmp = []
            for t in df[i].values:
                tmp.append(t)
            tmp = np.asarray(tmp)
            X = np.hstack((X, tmp))
            #(X.shape, tmp.shape)
        else:
            #print(X.shape, df[i].values[:, None].shape)
            X = np.hstack((X,df[i].values[:,None]))
    
    print(X.shape)
    y = df.values[:,-1]
        
    return X,y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=0, stratify=y)

    return X_train, y_train, X_test, y_test



def SVM(X,y,c):
    prob = svm_problem(y, X)
    par = '-t 0 -c {} -v 3'.format(c)
    param = svm_parameter(par)
    m = svm_train(prob, param)
    #lin_accuracy.append(m)
    return m


def SVM_all(X, y, type,cost, n=0.5, degree=3, v=3):
    prob = svm_problem(y, X)
    par = '-t {} -c {} -v {} -n {} -d {}'.format(type, cost, v, n, degree)
    print(par)
    param = svm_parameter(par)
    m = svm_train(prob, param)
    #lin_accuracy.append(m)
    return m

def main():
    X,y = data_preprocess()
    #X, y = df.values[:,:-1], df.values[:,-1]
    X_train, y_train, X_test, y_test = split_data(X, y)
    # Cross validation of linear svm
    """
    c_values = []
    lin_accuracy = []
    lin_timed = []
    for c in range(-5,8,1):
        c0 = time.clock()
        m = SVM(X_train,y_train,2**c)
        diff = time.clock()-c0
        lin_timed.append(diff)
        lin_accuracy.append(m)
        c_values.append(2**c)

    print('Max Accuracy and its index : {}, {}'.format(max(lin_accuracy),lin_accuracy.index(max(lin_accuracy))))
    plt.scatter(c_values, lin_accuracy, color='r')
    plt.xscale('log', basex=2)
    plt.plot(c_values, lin_accuracy)
    plt.grid(True)
    plt.title("Linear SVM training with 3 fold validation, Cost vs Accuracy")
    plt.xlabel("Cost value")
    plt.ylabel("Accuracy%")
    plt.show()


    plt.xscale('log', basex=2)
    plt.plot(c_values, lin_timed)
    plt.scatter(c_values, lin_timed, color='g')
    plt.grid(True)
    plt.title("Linear SVM training with 3 fold validation, Cost vs Time")
    plt.xlabel("Cost value")
    plt.ylabel("Time seconds") 
    plt.show()
    """
    """
    prob = svm_problem(y_train, X_train)
    m = svm_train(prob, '-t 0 -c 64 -n 3')
    svm_save_model('linear.model', m)
    m = svm_load_model('linear.model')
    p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
    print(p_acc)
    """
    c_values = []
    lin_accuracy = []
    lin_timed = []
    for c in range(-5, 7, 1):
        c0 = time.clock()
        m = SVM_all(X_train, y_train,type=1,cost=2**c,degree=3)
        diff = time.clock()-c0
        lin_timed.append(diff)
        lin_accuracy.append(m)
        c_values.append(2**c)

    print(lin_accuracy)
    print('Max Accuracy and its index : {}, {}'.format(
    max(lin_accuracy), lin_accuracy.index(max(lin_accuracy))))
    plt.scatter(c_values, lin_accuracy, color='r')
    plt.xscale('log', basex=2)
    plt.plot(c_values, lin_accuracy)
    plt.grid(True)
    plt.title("Poly 3 degree SVM training with 3 fold validation, Cost vs Accuracy")
    plt.xlabel("Cost value")
    plt.ylabel("Accuracy%")
    plt.show()

    plt.xscale('log', basex=2)
    plt.plot(c_values, lin_timed)
    plt.scatter(c_values, lin_timed, color='g')
    plt.grid(True)
    plt.title("Poly 3 degree SVM training with 3 fold validation, Cost vs Time")
    plt.xlabel("Cost value")
    plt.ylabel("Time seconds")
    plt.show()


if __name__ == "__main__":
     main()
