
import numpy as np
import pandas as pd
import os
from pandas import read_csv
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

#ANN
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.model_selection import GridSearchCV
import scikeras
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping



if __name__ == "__main__":
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('./input/housing.csv', header=None, delimiter=r"\s+", names=column_names)
    # print(data.head(5))
    # Let's scale the columns before plotting them against MEDV
    min_max_scaler = preprocessing.MinMaxScaler()
    column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = data.loc[:,column_sels]
    y = data['MEDV']
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    # fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
    # index = 0
    # axs = axs.flatten()
    # for i, k in enumerate(column_sels):
    #     sns.regplot(y=y, x=x[k], ax=axs[i])
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    # plt.savefig('housing_regression.png', dpi=300)
    y =  np.log1p(y)
    for col in x.columns:
        if np.abs(x[col].skew()) > 0.3:
            x[col] = np.log1p(x[col])
    
    x_scaled = min_max_scaler.fit_transform(x)
    def ANN(optimizer = 'adam',neurons=32,batch_size=32,epochs=50,activation='relu',patience=5,loss='mse'):
        model = Sequential()
        model.add(Dense(neurons, input_shape=(x_scaled.shape[1],), activation=activation))
        model.add(Dense(neurons, activation=activation))
        model.add(Dense(1))
        model.compile(optimizer = optimizer, loss=loss)
        early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience
        history = model.fit(x_scaled, y,
                batch_size=batch_size,
                epochs=epochs,
                callbacks = [early_stopping],
                verbose=0) #verbose set to 1 will show the training process
        return model
    
    clf = KerasRegressor(build_fn=ANN, verbose=0)
    scores = cross_val_score(clf, x_scaled, y, cv=3,scoring='neg_mean_squared_error')
    print("MSE:"+ str(-scores.mean()))
    # rf_params = {
    #     'n_estimators': sp_randint(10,100),
    #     "max_features":sp_randint(1,13),
    #     'max_depth': sp_randint(5,50),
    #     "min_samples_split":sp_randint(2,11),
    #     "min_samples_leaf":sp_randint(1,11),
    #     "criterion":['squared_error','absolute_error']
    # }
    # n_iter_search=10
    # clf = RandomForestRegressor(random_state=0)
    # Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error'
    #                             ,random_state=0,verbose=2,return_train_score=True)
    # Random.fit(x_scaled, y)
    # print(Random.best_params_)
    # print("MSE:"+ str(-Random.best_score_))
    
    # scores_map = {}
    # scores = cross_val_score(clf, x_scaled, y, cv=3, scoring='neg_mean_squared_error')
    # scores_map['RF'] = scores
    # print("MSE: %0.2f (+/- %0.2f)" % (-scores.mean(), scores.std()))
    pass