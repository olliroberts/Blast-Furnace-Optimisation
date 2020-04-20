# %% Libraries
# Load/import relevant libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pydot
import Clean_and_Prepare as cp

# %% Optimise and Train Model. Fit Data
# =============================================================================
# # Define optimisation parameters and values into a dictionary
# # Doesn't like 'citerion': ('mse', 'mae'),
# rf_parameters = {
#         'n_estimators': (100, 200, 500, 1000),
#         'max_depth': (None, 5, 10),
#         'max_features': ('auto', 'sqrt', 'log2'),
#         'warm_start': (True, False),
#         'random_state': (10, 20, 30, 40, 50, 60)
#         }
# # Initiate model and apply GridSearchCV
# rf_model = RandomForestRegressor()
# rf = GridSearchCV(rf_model, rf_parameters, n_jobs=4, cv=5)
# rf.fit(cp.train_bf4_mdl, cp.train_outputs)
# print(rf.best_estimator_)
# Best Parameters are:
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                    max_features='sqrt', max_leaf_nodes=None,
#                    min_impurity_decrease=0.0, min_impurity_split=None,
#                    min_samples_leaf=1, min_samples_split=2,
#                    min_weight_fraction_leaf=0.0, n_estimators=500,
#                    n_jobs=None, oob_score=False, random_state=10, verbose=0,
#                    warm_start=True)
# =============================================================================
# Initiate Model
rf = RandomForestRegressor(max_features='sqrt', n_estimators=500,
                           warm_start=True, random_state=None)
# Train the model on training data
rf.fit(cp.train_bf4_mdl, cp.train_outputs)

# %% Model Performance
# Train Scores
rf_train_p = rf.predict(cp.train_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
rf_train_mse = np.mean((cp.train_outputs-rf_train_p)**2)
rf_train_rmse = np.sqrt(rf_train_mse)
rf_train_mae = np.mean(abs(cp.train_outputs-rf_train_p))
rf_train_mape = 100 - np.mean(100 * (abs(cp.train_outputs-rf_train_p)/cp.train_outputs))
rf_train_r2 = rf.score(cp.train_bf4_mdl, cp.train_outputs)

# Test Scores
rf_test_p = rf.predict(cp.test_bf4_mdl)
# Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
# Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
# Coefficient of determination (R2)
rf_test_mse = np.mean((cp.test_outputs-rf_test_p)**2)
rf_test_rmse = np.sqrt(rf_test_mse)
rf_test_mae = np.mean(abs(cp.test_outputs-rf_test_p))
rf_test_mape = 100 - np.mean(100 *
                             (abs(cp.test_outputs-rf_test_p)/cp.test_outputs))
rf_test_r2 = rf.score(cp.test_bf4_mdl, cp.test_outputs)

# %% View a tree
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file='tree.dot', feature_names=bf4reduced_list,
#                rounded=True, precision=1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')

# %% Variable Importance
# Get numerical feature importances
# rf_importances = list(rf.feature_importances_)
# List of tuples with variable and importance
# bf4_rf_importances = [(feature, round(importance, 4)) for feature,
#                       importance in zip(cp.bf4r_list, rf_importances)]
# Sort the feature importances by most important first
# bf4_rf_importances = sorted(bf4_rf_importances,
#                             key=lambda x: x[1], reverse=True)
# Print out the feature and importances
# [print('Variable: {:25} Importance: {}'
#       .format(*pair)) for pair in bf4_rf_importances]
