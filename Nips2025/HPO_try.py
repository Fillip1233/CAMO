import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats
#Random Forest
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from scikeras.wrappers import KerasClassifier

def ANN(optimizer = 'sgd', neurons=32,batch_size=32,epochs=20,activation='relu',loss='categorical_crossentropy'):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics = ['accuracy'])
    return model
if __name__ == "__main__":
    d = datasets.load_digits()
    X = d.data
    y = d.target


    clf = KerasClassifier(build_fn=ANN, optimizer='adam', neurons=32, batch_size=32, epochs=20, activation='relu', loss='categorical_crossentropy')
    # clf.fit(X,pd.get_dummies(y).values)
    scores = cross_val_score(clf, X, pd.get_dummies(y).values, cv=3, scoring='accuracy')

    #Random Forest
    clf = RandomForestClassifier(n_estimators = 100, random_state=0)
    # clf.fit(X,y)
    scores = cross_val_score(clf, X, y, cv=3,scoring='accuracy')
    print("Accuracy:"+ str(scores.mean()))

    # Define the hyperparameter configuration space
    rf_params = {
        # 'n_estimators': sp_randint(10,100),
        "max_features":sp_randint(1,64),
        'max_depth': sp_randint(5,50),
        "min_samples_split":sp_randint(2,11),
        "min_samples_leaf":sp_randint(1,11),
        "criterion":['gini','entropy']
    }
    n_iter_search=20 #number of iterations is set to 20, you can increase this number if time permits
    clf = RandomForestClassifier(n_estimators = 100, random_state=0)
    Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy',verbose=2,random_state=0,return_train_score=True)
    Random.fit(X, y)
    print(Random.best_params_)
    print("Accuracy:"+ str(Random.best_score_))
    print("\n每次迭代的准确率(平均交叉验证得分):")
    for i in range(n_iter_search):
        mean_score = Random.cv_results_['mean_test_score'][i]
        std_score = Random.cv_results_['std_test_score'][i]
        params = Random.cv_results_['params'][i]
        print(f"Iter {i+1}: acc = {mean_score:.4f} ± {std_score:.4f}, params = {params}")
    
    pass