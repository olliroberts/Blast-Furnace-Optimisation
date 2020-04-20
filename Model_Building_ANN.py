# %% Libraries
# Load/import relevant libraries
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import Clean_and_Prepare as cp

# %% Optimise and Train Model. Fit Data
# =============================================================================
# Define optimisation parameters and values into a dictionary
# Doesn't like 'solver': ('lbfgs', 'sgd', 'adam'),
# ann_parameters = {
#         'hidden_layer_sizes': (100, 200, 500, 750, 1000),
#         'activation': ('logistic', 'identity', 'tanh', 'relu'),
#         'learning_rate': ('constant', 'invscaling', 'adaptive'),
#         'max_iter': (200, 300, 400),
#         'warm_start': (False, True),
#         'random_state': (None, 10, 20, 30, 40)
#        }
# Initiate model and apply GridSearchCV
# ann_model = MLPRegressor(solver='lbfgs')
# ann = GridSearchCV(ann_model, ann_parameters, n_jobs=4, cv=5)
# ann.fit(cp.train_bf4_mdl, cp.train_outputs)
# print(ann.best_estimator_)
# Optimal Model Parameters
# MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
#              beta_2=0.999, early_stopping=False, epsilon=1e-08,
#              hidden_layer_sizes=1000, learning_rate='constant',
#              learning_rate_init=0.001, max_iter=400, momentum=0.9,
#              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#              random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
#              validation_fraction=0.1, verbose=False, warm_start=False)
# =============================================================================
# Initiate Model
ann = MLPRegressor(hidden_layer_sizes=(1000, ), activation='logistic',
                   solver='lbfgs', learning_rate='constant', max_iter=400,
                   random_state=None)
# Train the model on training data
ann.fit(cp.train_bf4_mdl, cp.train_outputs)

# %% Model Performance
# Train Scores
ann_train_p = ann.predict(cp.train_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
ann_train_mse = np.mean((cp.train_outputs-ann_train_p)**2)
ann_train_rmse = np.sqrt(ann_train_mse)
ann_train_mae = np.mean(abs(cp.train_outputs-ann_train_p))
ann_train_mape = 100 - np.mean(100 * (abs(cp.train_outputs-ann_train_p)/cp.train_outputs))
ann_train_r2 = ann.score(cp.train_bf4_mdl, cp.train_outputs)

# Test Scores
ann_test_p = ann.predict(cp.test_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
ann_test_mse = np.mean((cp.test_outputs-ann_test_p)**2)
ann_test_rmse = np.sqrt(ann_test_mse)
ann_test_mae = np.mean(abs(cp.test_outputs-ann_test_p))
ann_test_mape = 100 - np.mean(100 * (abs(cp.test_outputs-ann_test_p)/cp.test_outputs))
ann_test_r2 = ann.score(cp.test_bf4_mdl, cp.test_outputs)
