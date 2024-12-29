import random
import math
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform
import matplotlib.pyplot as plt

def main():

    # Loading from file and initializing
    dev_file = np.load("train_good.npz", allow_pickle=True)
    X_dev = dev_file["x"]
    Y_dev = dev_file["y"]

    test_file = np.load("test_good.npz", allow_pickle=True)
    X_test = test_file["x"]
    Y_test = test_file["y"]

    # Preprocessing the data using StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_dev)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # Impute NaN values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_dev_scaled = imputer.fit_transform(X_dev_scaled)
    X_test_scaled = imputer.transform(X_test_scaled)

    # Define the parameter grid to search for MLPRegressor
    param_dist = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (50, 100)],
        "activation": ["tanh", "relu"],
        "solver": ["adam"],  # "sgd" tends to converge slower
        "alpha": uniform(0.0001, 0.01),  # Adjusted L2 penalty
        "learning_rate": ["constant", "adaptive"],
    }

    # Initiating the MLPRegressor with RandomizedSearchCV for hyperparameter tuning
    reg = MLPRegressor(max_iter=2000)
    n_iter_search = 20  # Increased number of parameter settings that are sampled
    random_search = RandomizedSearchCV(reg, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, n_jobs=-1)

    # Fit the model
    random_search.fit(X_dev_scaled, Y_dev)

    # Get the best parameters
    best_params = random_search.best_params_

    print("Best parameters found: {}".format(best_params))

    # Evaluate the best model
    best_regressor = random_search.best_estimator_
    regressed = best_regressor.predict(X_test_scaled)
    R2 = r2_score(Y_test, regressed)
    print("The R2 on the test set:{}".format(R2))
    mse = mean_squared_error(Y_test, regressed)
    print("The mean squared error (MSE) on the test set:{}".format(mse))
    mae = mean_absolute_error(Y_test, regressed)
    print("The mean absolute error (MAE) on the test set:{}".format(mae))
    
    # Plot parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_test, regressed, edgecolors=(0, 0, 0), alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    plt.grid(True)
    plt.show()

main()

