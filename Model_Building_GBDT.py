# %% Libraries
# Load/import relevant libraries
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import Clean_and_Prepare as cp

# %% Model Building
# =============================================================================
# Define optimisation parameters and values into a dictionary
# Doesn't like 'citerion': ('mse', 'mae'),
# gbdt_parameters = {
#         'loss': ('ls', 'lad', 'huber', 'quantile'),
#         'learning_rate': (0.1, 0.25, 0.5),
#         'n_estimators': (100, 250, 500, 1000),
#         'max_depth': (1, 2, 3, 4),
#         'random_state': (None, 10, 20, 30, 40),
#         'max_features': ('auto', 'sqrt', 'log2'),
#         'warm_start': (True, False)
#         }
# Initiate model and apply GridSearchCV
# gbdt_model =  GradientBoostingRegressor()
# gbdt = GridSearchCV(gbdt_model, gbdt_parameters, n_jobs=4, cv=5)
# gbdt.fit(cp.train_bf4_mdl, cp.train_outputs)
# print(gbdt.best_estimator_)
# Best Parameters are:
# GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                           learning_rate=0.1, loss='ls', max_depth=4,
#                           max_features='sqrt', max_leaf_nodes=None,
#                           min_impurity_decrease=0.0, min_impurity_split=None,
#                           min_samples_leaf=1, min_samples_split=2,
#                           min_weight_fraction_leaf=0.0, n_estimators=500,
#                           n_iter_no_change=None, presort='auto',
#                           random_state=None, subsample=1.0, tol=0.0001,
#                           validation_fraction=0.1, verbose=0,
#                           warm_start=False)
# =============================================================================
# Initiate Model
gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1,
                                 n_estimators=500, max_depth=4,
                                 random_state=None, max_features='sqrt',
                                 warm_start=False)
# Train the model on training data
gbdt.fit(cp.train_bf4_mdl, cp.train_outputs)

# %% Model Performance
# Train Scores
gbdt_train_p = gbdt.predict(cp.train_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
gbdt_train_mse = np.mean((cp.train_outputs-gbdt_train_p)**2)
gbdt_train_rmse = np.sqrt(gbdt_train_mse)
gbdt_train_mae = np.mean(abs(cp.train_outputs-gbdt_train_p))
gbdt_train_mape = 100 - np.mean(100 * (abs(cp.train_outputs-gbdt_train_p)/cp.train_outputs))
gbdt_train_r2 = gbdt.score(cp.train_bf4_mdl, cp.train_outputs)

# Test Scores
gbdt_test_p = gbdt.predict(cp.test_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
gbdt_test_mse = np.mean((cp.test_outputs-gbdt_test_p)**2)
gbdt_test_rmse = np.sqrt(gbdt_test_mse)
gbdt_test_mae = np.mean(abs(cp.test_outputs-gbdt_test_p))
gbdt_test_mape = 100 - np.mean(100 * (abs(cp.test_outputs-gbdt_test_p)/cp.test_outputs))
gbdt_test_r2 = gbdt.score(cp.test_bf4_mdl, cp.test_outputs)

# %% Feature Importance
print(gbdt.feature_importances_)
