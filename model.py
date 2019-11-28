import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from feature_engineering import feature_engineering
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def fit_FNN(features, labels):
        
    model = keras.Sequential([
                layers.Dense(16, activation='relu', input_shape=[len(features.keys())]),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation="linear")])
        
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        # Apply early stop to prevent overfitting
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.fit(features, labels, epochs = 30, validation_split = 0.2, verbose = 0)
    return model

model = fit_FNN(X_train, y_train)
model.summary()
loss, mse, mae = model.evaluate(X_test, y_test, verbose=2)
print("Testing set Mean Squared Error: {:5.2f} ".format(mse))

pred= model.predict(X_test).flatten()
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)

# SVM_regression
model = SVC(gamma='scale').fit(X_train, y_train)
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)

# KNN
model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
pred= model.predict(X_test)
rounded = np.round(pred)
error = rounded - y_test
error_mse = np.mean(error**2)
print(error_mse)

# decision tree
model = DecisionTreeRegressor().fit(X_train, y_train)
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)

# Ensemble methods
# RandomForestRegressor
model = RandomForestRegressor(random_state=1, n_estimators=10).fit(X_train, y_train)
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)

# AdaBoostRegressor: n_estimators matters, smaller n_estimators makes better performance (have roughly tried several numbers)
model = AdaBoostRegressor(random_state=0, n_estimators=3).fit(X_train, y_train)
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)

# GradientBoostingRegressor: n_estimators matters, larger n_estimators makes better performance
model = GradientBoostingRegressor(n_estimators=150, random_state=0).fit(X_train, y_train)
pred = model.predict(X_test)
error = mean_squared_error(y_test, pred)
print(error)
