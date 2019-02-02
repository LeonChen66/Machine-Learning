import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import feature_selection,model_selection
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import math
import operator
from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score
sns.set(style="ticks", color_codes=False)


def import_data(dir):
    #Import the data set using a pandas frame with condition that area>0
    data = pd.read_csv(dir)
    data['category'] = (data['area'] > 0).astype(int)
    return data

def add_Class(fileName, skipZeros=False):
    data = pd.read_csv(fileName)
    data['Log-area'] = np.log10(data['area']+1)
    data['category'] = (data['area'] > 0).astype(int)
    #data.describe().to_csv("data_description/description.csv", float_format="%.2f")
    data_without = data.where(data['area'] != 0)
    data_without = data_without[data_without['area'].notnull()]
   # data_without.describe().to_csv(
    #    "data_description/description_Nozeros.csv", float_format="%.2f")
    return data_without if skipZeros else data

def plot_pair(df):
    df=df.drop(columns=['X','Y','month','day','Log-area','area'])
    #print(df)
    g = sns.pairplot(df, hue="category", size=1,
                     markers=["o", "+"], plot_kws={"s": 7}
                    ,vars=['rain', 'FFMC','DMC','DC','ISI','temp','RH','wind'])
    #g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=5)
    #plt.show()

def cross_validation(feature_set, truth_set, K, num_splits):

    kf = KFold(n_splits=num_splits)
    set_accuracy = []
    total = 0
    # Split the set indices between training and testing sets
    for train_index, test_index in kf.split(feature_set):
        correct = 0

        X_train, x_test = feature_set[train_index], feature_set[test_index]
        y_train, y_test = truth_set[train_index], truth_set[test_index]

        total = len(y_test)

        for i in range(len(x_test)):
            classification = run_knn(K, y_train, X_train, x_test[i])

            if classification == y_test[i]:
                correct += 1

        set_accuracy.append(correct)


    #print("K VALUE:", K, "---- ERRORS:", sum(set_accuracy)/len(set_accuracy))
    print(float(sum(set_accuracy))/len(set_accuracy))/total
    return float((sum(set_accuracy)))/(len(set_accuracy))/total

def cross_validation(feature_set, truth_set, K, num_splits):

    kf = KFold(n_splits=num_splits)
    set_accuracy = []
    total = 0
    # Split the set indices between training and testing sets
    for train_index, test_index in kf.split(feature_set):
        correct = 0

        X_train, x_test = feature_set[train_index], feature_set[test_index]
        y_train, y_test = truth_set[train_index], truth_set[test_index]

        total = len(y_test)

        for i in range(len(x_test)):
            classification = run_knn(K, y_train, X_train, x_test[i])

            if classification == y_test[i]:
                correct += 1

        set_accuracy.append(correct)


    return float((sum(set_accuracy)))/(len(set_accuracy))/total

def plot_forest(dataFrame):
    data = dataFrame
    basedir = 'images/'
    #print(data_without)
    """
    fig, axes = plt.subplots(1, len(data.columns.values)-1, sharey=True)

    for i, col in enumerate(data_without.columns.values[:-2]):
        data_without.plot(x=[col], y=["Log-area"], kind="scatter", ax=axes[i])
    """
    #plt.savefig('all.png')
    #df.boxplot(column='Log-area', by='month')
    for i in data.describe().columns[:-2]:
        data.plot.scatter(i, 'Log-area', grid=True)
        name = basedir+"col_{}.png".format(i)
        plt.savefig(name)

    data.boxplot(column='Log-area', by='month')
    
    plt.savefig('box_month_log.png')
    #plt.show()


#def Feature_Selection(dataFrame):


def KNN_train(X,y,parameter):
    #X_normal = preprocessing.normalize(X,norm='l1')
    #print(len(X),len(y)
    neighbor = KNeighborsClassifier(n_neighbors=parameter, algorithm='brute', p=1)
   # minmax = preprocessing.MinMaxScaler()
   # X_normal = minmax.fit_transform(X)
    #X_normal = preprocessing.normalize(X, norm='l1')

    neighbor.fit(X,y)
    #model = feature_selection.SelectFromModel(neighbor,prefit=True)
    #x_new = model.transform(X_normal)

    return neighbor



def determinN(X,y):
    # creating odd list of K for KNN
    myList = list(range(1, 50))
    # subsetting just the odd ones
    neighbors = filter(lambda x: x % 2 != 0, myList)
    # empty list that will hold cv scores
    cv_scores = []


    kf = model_selection.StratifiedKFold(n_splits=10)
# perform 10-fold cross validation

    ac = []
    res = []
    for k in neighbors:
        ac = []
        for train_index, test_index in kf.split(X, y):
            clf = KNeighborsClassifier(n_neighbors=k)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            ac.append(accuracy_score(y_test, y_pred))
        ac = np.mean(ac)
        res.append(ac)
    print(res)

    knn = KNeighborsClassifier(n_neighbors=13)


    # Use cross_val_score function
    # We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the dat
    # cv=10 for 10 folds
    # scoring='accuracy' for evaluation metric - althought they are many
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    print(np.mean(scores))
        #Y_pred = model_selection.cross_val_predict(knn, X,y, cv=10, method='predict')
        #print(sum(1 for i, j in zip(y, Y_pred) if i != j))


    #print(cv_scores)

def KNN_Scores(X,y,model):
    #minmax = preprocessing.MinMaxScaler()
    #X_normal = minmax.fit_transform(X)
    print(model.score(X,y))


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def knn(trainingSet, testInstance, k):  #self implement 
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])

	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(),
	                     key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def main():
    pd.read_csv('data_Set/train.csv')
    data = add_Class('data_Set/train.csv')
    #plot_forest(data)
    #plot_pair(data)
    
    X_train = data[data.columns[:-3]].values
    y_train = data['category'].values
    determinN(X_train,y_train)
    """
    KNNmodel = KNN_train(X_train,y_train,7)
    data_test = add_Class('data_Set/test.csv')
    X_test = data_test[data_test.columns[:-3]].values
    y_test = data_test['category'].values
    KNN_Scores(X_test,y_test,KNNmodel)
    """
    """
    training = import_data('data_Set/train.csv')
    testing = import_data('data_Set/test.csv')

    y_train = np.array(training['category'])
    y_test = np.array(testing['category'])
    #get the features matrix
    del training['area'],training['category'], testing['area'],testing['category']

    norm = np.linalg.norm(training, axis=0)
    training /= norm
    testing /= norm
    x_train = np.array(training)
    x_test = np.array(testing)
    determinN(x_train, y_train)
    model = KNN_train(x_train, y_train,7)
    KNN_Scores(x_test,y_test,model)
    """
if __name__ == "__main__":
    main()
