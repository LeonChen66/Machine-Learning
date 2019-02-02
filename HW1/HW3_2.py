
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def import_data(dir):
    #Import the data set using a pandas frame with condition that area>0
    data = pd.read_csv(dir)
    data = data[data['area'] > 0]
    return data

def add_Class(fileName, skipZeros=False):
    data = pd.read_csv(fileName)
    data['Log-area'] = np.log10(data['area']+1)
    data['category'] = (data['area'] > 0).astype(int)
    #data.describe().to_csv("data_description/description.csv", float_format="%.2f")
    data_without = data.where(data['area'] != 0)
    data_without = data_without[data_without['area'].notnull()]
    #data_without.describe().to_csv(
    #    "data_description/description_Nozeros.csv", float_format="%.2f")
    return data_without if skipZeros else data

def train_regression(X,y):
    reg = LinearRegression(fit_intercept=True,normalize=False).fit(X, y)
    #model = sm.OLS(y, X)
    #result = model.fit()
    #print(reg.score(X,y))
    
    return reg

def remove_outlier(forest_fire):
    forest_fire['area_cat'] = pd.cut(forest_fire['area'], bins=[0, 5, 10, 50, 100, 1100], include_lowest=True,
    labels=['0-5', '5-10', '10-50', '50-100', '>100'])
    forest_fire.area_cat.value_counts()
    forest_fire.drop(forest_fire[forest_fire.area >100].index, axis=0, inplace=True)
    return forest_fire

def draw_plot(forest_fire):
    forest_fire.hist(bins=50, figsize=(30, 20), ec='w',
                      xlabelsize=5, ylabelsize=5)
    corr_matrix = forest_fire.corr(method='spearman')
    ax = plt.figure(figsize=(12, 8))
    ax = sns.heatmap(corr_matrix, cmap='PiYG')
    plt.show()
    print(corr_matrix.area.sort_values(ascending=False))
    
    #visualizing relations of most related attributes
    attributes = ['area', 'wind', 'temp', 'rain', 'RH']
    sns.pairplot(forest_fire[attributes])
    plt.show()


def fit_model(X,y):
    num_instances = len(X)
    models = []
    models.append(('LiR', LinearRegression()))
    models.append(('Ridge', Ridge()))
    models.append(('Lasso', Lasso()))
    models.append(('ElasticNet', ElasticNet()))
    models.append(('Bag_Re', BaggingRegressor()))
    models.append(('RandomForest', RandomForestRegressor()))
    models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVM', SVR()))

    # Evaluations
    results = []
    names = []
    scoring = []

    for name, model in models:
        # Fit the model
        model.fit(X, Y)

        predictions = model.predict(X)

        # Evaluate the model
        score = explained_v


def OLS(x, y): #self OLS
    w = np.dot(np.linalg.inv(np.dot(np.matrix.transpose(x), x)),
               np.dot(np.matrix.transpose(x), y))
    return w


def RSS(x_test, w, y_test):  #self RSS
    RSS = np.dot((y_test - np.dot(x_test, w)), (y_test - np.dot(x_test, w)))


if __name__ == "__main__":
    """
    data_without = add_Class('data_Set/train.csv',True)
    forest_fire = data_without.iloc[:,4:-1]
    draw_plot(forest_fire)
    X_train = data_without[data_without.columns[:-3]].values
    norm = data_without.apply(lambda x: np.sqrt(x**2).sum()/x.shape[0])
    X_normal = X_train
    #X_normal = preprocessing.normalize(X_train, norm='l1',axis=1)
    
    y_train = data_without['area'].values
    model = train_regression(X_normal,y_train)
    data_test = add_Class('data_Set/test.csv')
    X_test = data_test[data_test.columns[:-3]].values
    y_test = data_test['area'].values
    X_normal_t = X_test/norm
    #X_normal_t = preprocessing.normalize(X_test, norm='l1', axis=1)
    """

    training = import_data('data_Set/train.csv')
    testing = import_data('data_Set/test.csv')

    #get output matrix
    y_train = np.array(training['area'])
    y_test = np.array(testing['area'])

    #get the features matrix
    del training['area'], testing['area']

    norm = training.apply(lambda x: np.sqrt(x**2).sum()/x.shape[0])
    training /= norm
    testing /= norm
    x_train = np.array(training)
    x_test = np.array(testing)

    model = train_regression(x_train, y_train)
    #print(model.score(X_normal_t,y_test))    
    print(np.sum(np.power(y_test-model.predict(x_test), 2))) 
