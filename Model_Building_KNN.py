# %% Libraries
# Load/import relevant libraries
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import Clean_and_Prepare as cp

# %% Model Building
# =============================================================================
# Define optimisation parameters and values into a dictionary
# Doesn't like 'citerion': ('mse', 'mae'),
# knn_parameters = {
#         'n_neighbors': (3, 4, 5, 6, 7),
#         'weights': ('uniform', 'distance'),
#         'algorithm': ('ball_tree', 'kd_tree', 'brute'),
#         'leaf_size': (10, 20, 30, 40, 50)
#          }
# Initiate model and apply GridSearchCV
# knn_model =  KNeighborsRegressor()
# knn = GridSearchCV(knn_model, knn_parameters, n_jobs=4, cv=5)
# knn.fit(cp.train_bf4_mdl, cp.train_outputs)
# print(knn.best_estimator_)
# Best Parameters are:
# KNeighborsRegressor(algorithm='brute', leaf_size=10, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
#                     weights='distance')
# =============================================================================
# Initiate Model
knn = KNeighborsRegressor(n_neighbors=4, weights='distance', algorithm='brute',
                          leaf_size=10)
# Train the model on training data
knn.fit(cp.train_bf4_mdl, cp.train_outputs)

# %% Model Performance
# Train Scores
knn_train_p = knn.predict(cp.train_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
knn_train_mse = np.mean((cp.train_outputs-knn_train_p)**2)
knn_train_rmse = np.sqrt(knn_train_mse)
knn_train_mae = np.mean(abs(cp.train_outputs-knn_train_p))
knn_train_mape = 100 - np.mean(100 * (abs(cp.train_outputs-knn_train_p)/cp.train_outputs))
knn_train_r2 = knn.score(cp.train_bf4_mdl, cp.train_outputs)

# Test Scores
knn_test_p = knn.predict(cp.test_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
knn_test_mse = np.mean((cp.test_outputs-knn_test_p)**2)
knn_test_rmse = np.sqrt(knn_test_mse)
knn_test_mae = np.mean(abs(cp.test_outputs-knn_test_p))
knn_test_mape = 100 - np.mean(100 * (abs(cp.test_outputs-knn_test_p)/cp.test_outputs))
knn_test_r2 = knn.score(cp.test_bf4_mdl, cp.test_outputs)
