# %% Libraries
# Load/import relevant libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import Clean_and_Prepare as cp

# %% Showing the data
# pd.plotting.scatter_matrix(cp.bf4r_inputs.loc[0:, bf4reduced_inputs.columns],
#                           alpha=0.5, figsize=[25, 25], diagonal='hist',
#                           s=200, marker='.', edgecolor='black')
# plt.show()

# %% Scale and Split Data
# Scale input data
bf4_mdl_scaled = preprocessing.scale(cp.bf4r_inputs)
# Create training and test data from scaled array
train_bf4_mdl_scaled, test_bf4_mdl_scaled, train_outputs, test_outputs = train_test_split(
        bf4_mdl_scaled, cp.outputs, test_size=0.25, random_state=None)

# %% Optimise and Train Model. Fit Data
# =============================================================================
# # Define optimisation parameters and values into a dictionary
# svr_parameters = {
#         'kernel': ('poly', 'rbf', 'sigmoid'),
#         'gamma': ('scale', 'auto'),
#         'C': (0.5, 1, 2),
#         'epsilon': (0.1, 0.05, 0.025, 0.01)
#         }
# Initiate model and apply GridSearchCV
# svr_model = svm.SVR()
# svr = GridSearchCV(svr_model, svr_parameters, n_jobs=4, cv=5)
# svr.fit(train_bf4_mdl_scaled, train_outputs)
# print(svr.best_estimator_)
# Optimal Model Parameters
# SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma='auto',
#     kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# =============================================================================
# Initiate Model
svr = svm.SVR(kernel='rbf', C=0.5, epsilon=0.01, gamma='auto')
# Train the model on training data
svr.fit(train_bf4_mdl_scaled, train_outputs)

# %% Model Performance
# Train Scores
svr_train_p = svr.predict(train_bf4_mdl_scaled)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
svr_train_mse = np.mean((train_outputs-svr_train_p)**2)
svr_train_rmse = np.sqrt(svr_train_mse)
svr_train_mae = np.mean(abs(train_outputs-svr_train_p))
svr_train_mape = 100 - np.mean(100 *
                               (abs(train_outputs-svr_train_p)/train_outputs))
svr_train_r2 = svr.score(train_bf4_mdl_scaled, train_outputs)

# Test Scores
svr_test_p = svr.predict(test_bf4_mdl_scaled)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
svr_test_mse = np.mean((test_outputs-svr_test_p)**2)
svr_test_rmse = np.sqrt(svr_test_mse)
svr_test_mae = np.mean(abs(test_outputs-svr_test_p))
svr_test_mape = 100 - np.mean(100 *
                              (abs(test_outputs-svr_test_p)/test_outputs))
svr_test_r2 = svr.score(test_bf4_mdl_scaled, test_outputs)
