# %% Modules
import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt

# %% Create blank DataFrame
measure = ['RF Train MSE', 'RF Train RMSE', 'RF Train MAE', 'RF Train MAPE',
           'RF Train R2', 'RF Test MSE', 'RF Test RMSE', 'RF Test MAE',
           'RF Test MAPE', 'RF Test R2', 'ANN Train MSE',
           'ANN Train RMSE', 'ANN Train MAE', 'ANN Train MAPE', 'ANN Train R2',
           'ANN Test MSE', 'ANN Test RMSE', 'ANN Test MAE', 'ANN Test MAPE',
           'ANN Test R2', 'SVR Train MSE', 'SVR Train RMSE', 'SVR Train MAE',
           'SVR Train MAPE', 'SVR Train R2', 'SVR Test MSE', 'SVR Test RMSE',
           'SVR Test MAE', 'SVR Test MAPE', 'SVR Test R2', 'GBDT Train MSE',
           'GBDT Train RMSE', 'GBDT Train MAE',
           'GBDT Train MAPE', 'GBDT Train R2', 'GBDT Test MSE', 'GBDT Test RMSE',
           'GBDT Test MAE', 'GBDT Test MAPE', 'GBDT Test R2', 'KNN Train MSE',
           'KNN Train RMSE', 'KNN Train MAE', 'KNN Train MAPE', 'KNN Train R2',
           'KNN Test MSE', 'KNN Test RMSE', 'KNN Test MAE', 'KNN Test MAPE',
           'KNN Test R2']
model_performance = pd.DataFrame(measure, columns=['Measure'])

# %% Loop
x = 1
while x < 11:
    import Model_Building_RF as rf
    import Model_Building_ANN as ann
    import Model_Building_SVR as svr
    import Model_Building_GBDT as gbdt
    import Model_Building_KNN as knn

    measures = np.array([[rf.rf_train_mse, rf.rf_train_rmse, rf.rf_train_mae,
                          rf.rf_train_mape, rf.rf_train_r2, rf.rf_test_mse,
                          rf.rf_test_rmse, rf.rf_test_mae, rf.rf_test_mape,
                          rf.rf_test_r2, ann.ann_train_mse, ann.ann_train_rmse,
                          ann.ann_train_mae, ann.ann_train_mape,
                          ann.ann_train_r2, ann.ann_test_mse,
                          ann.ann_test_rmse, ann.ann_test_mae,
                          ann.ann_test_mape, ann.ann_test_r2,
                          svr.svr_train_mse, svr.svr_train_rmse,
                          svr.svr_train_mae, svr.svr_train_mape,
                          svr.svr_train_r2, svr.svr_test_mse,
                          svr.svr_test_rmse, svr.svr_test_mae,
                          svr.svr_test_mape, svr.svr_test_r2,
                          gbdt.gbdt_train_mse, gbdt.gbdt_train_rmse,
                          gbdt.gbdt_train_mae, gbdt.gbdt_train_mape,
                          gbdt.gbdt_train_r2, gbdt.gbdt_test_mse,
                          gbdt.gbdt_test_rmse, gbdt.gbdt_test_mae,
                          gbdt.gbdt_test_mape, gbdt.gbdt_test_r2,
                          knn.knn_train_mse, knn.knn_train_rmse,
                          knn.knn_train_mae, knn.knn_train_mape,
                          knn.knn_train_r2, knn.knn_test_mse,
                          knn.knn_test_rmse, knn.knn_test_mae,
                          knn.knn_test_mape, knn.knn_test_r2]]).T

    model_performance['Run_' + str(x)] = measures

    importlib.reload(rf)
    importlib.reload(ann)
    importlib.reload(svr)
    importlib.reload(gbdt)
    importlib.reload(knn)
    x += 1

# model_performance = model_performance.drop(columns='1')

# %% Avg and Std
average = model_performance.mean(axis=1)
std_dev = model_performance.std(axis=1)
model_performance = model_performance.assign(Average=average)
model_performance = model_performance.assign(Std_Dev=std_dev)

# %% Bar Graphs
x_axis = pd.Series([1, 2, 3, 4, 5])
labels = ["RF", "ANN", "SVR", "GBDT", "KNN"]
legend = ['Train', 'Test']

# MSE
train_mse = model_performance.loc[model_performance['Measure'].isin(['RF Train MSE', 'ANN Train MSE', 'SVR Train MSE', 'GBDT Train MSE', 'KNN Train MSE'])]
test_mse = model_performance.loc[model_performance['Measure'].isin(['RF Test MSE', 'ANN Test MSE', 'SVR Test MSE', 'GBDT Test MSE', 'KNN Test MSE'])]
plt.figure(1)
plt.bar(x_axis-0.1, train_mse['Average'], width=0.2, color='r', align='center', yerr=train_mse['Std_Dev'])
plt.bar(x_axis+0.1, test_mse['Average'], width=0.2, color='g', align='center', yerr=test_mse['Std_Dev'])
plt.xticks(x_axis, labels)
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Average MSE Score')
plt.legend(legend, loc=0)

# RMSE
train_rmse = model_performance.loc[model_performance['Measure'].isin(['RF Train RMSE', 'ANN Train RMSE', 'SVR Train RMSE', 'GBDT Train RMSE', 'KNN Train RMSE'])]
test_rmse = model_performance.loc[model_performance['Measure'].isin(['RF Test RMSE', 'ANN Test RMSE', 'SVR Test RMSE', 'GBDT Test RMSE', 'KNN Test RMSE'])]
plt.figure(2)
plt.bar(x_axis-0.1, train_rmse['Average'], width=0.2, color='r', align='center', yerr=train_rmse['Std_Dev'])
plt.bar(x_axis+0.1, test_rmse['Average'], width=0.2, color='g', align='center', yerr=test_rmse['Std_Dev'])
plt.xticks(x_axis, labels)
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Average RMSE Score')
plt.legend(legend, loc=0)

# MAE
train_mae = model_performance.loc[model_performance['Measure'].isin(['RF Train MAE', 'ANN Train MAE', 'SVR Train MAE', 'GBDT Train MAE', 'KNN Train MAE'])]
test_mae = model_performance.loc[model_performance['Measure'].isin(['RF Test MAE', 'ANN Test MAE', 'SVR Test MAE', 'GBDT Test MAE', 'KNN Test MAE'])]
plt.figure(3)
plt.bar(x_axis-0.1, train_mae['Average'], width=0.2, color='r', align='center', yerr=train_mae['Std_Dev'])
plt.bar(x_axis+0.1, test_mae['Average'], width=0.2, color='b', align='center', yerr=test_mae['Std_Dev'])
plt.xticks(x_axis, labels)
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('Average MAE Score')
plt.legend(legend, loc=0)

# MAPE
train_mape = model_performance.loc[model_performance['Measure'].isin(['RF Train MAPE', 'ANN Train MAPE', 'SVR Train MAPE', 'GBDT Train MAPE', 'KNN Train MAPE'])]
test_mape = model_performance.loc[model_performance['Measure'].isin(['RF Test MAPE', 'ANN Test MAPE', 'SVR Test MAPE', 'GBDT Test MAPE', 'KNN Test MAPE'])]
plt.figure(4)
plt.bar(x_axis-0.1, train_mape['Average'], width=0.2, color='r', align='center', yerr=train_mape['Std_Dev'])
plt.bar(x_axis+0.1, test_mape['Average'], width=0.2, color='g', align='center', yerr=test_mape['Std_Dev'])
plt.xticks(x_axis, labels)
plt.ylim(97, 100)
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Model')
plt.ylabel('MAPE')
plt.title('Average MAPE Score')
plt.legend(legend, loc=0)

# R2
train_r2 = model_performance.loc[model_performance['Measure'].isin(['RF Train R2', 'ANN Train R2', 'SVR Train R2', 'GBDT Train R2', 'KNN Train R2'])]
test_r2 = model_performance.loc[model_performance['Measure'].isin(['RF Test R2', 'ANN Test R2', 'SVR Test R2', 'GBDT Test R2', 'KNN Test R2'])]
plt.figure(5)
plt.bar(x_axis-0.1, train_r2['Average'], width=0.2, color='moccasin', align='center', yerr=train_r2['Std_Dev'])
plt.bar(x_axis+0.1, test_r2['Average'], width=0.2, color='paleturquoise', align='center', yerr=test_r2['Std_Dev'])
# https://matplotlib.org/gallery/color/named_colors.html
plt.xticks(x_axis, labels)
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Model')
plt.ylabel('R2')
# plt.title('Average R2 Score')
plt.legend(legend, loc=0)

# %% Make Table of Average Measures
model_performance = model_performance.set_index('Measure')

data = {'RF Train': [model_performance.loc['RF Train MSE', 'Average'], model_performance.loc['RF Train RMSE', 'Average'],
                     model_performance.loc['RF Train MAE', 'Average'], model_performance.loc['RF Train MAPE', 'Average'],
                     model_performance.loc['RF Train R2', 'Average']],
        'RF Test': [model_performance.loc['RF Test MSE', 'Average'], model_performance.loc['RF Test RMSE', 'Average'],
                    model_performance.loc['RF Test MAE', 'Average'], model_performance.loc['RF Test MAPE', 'Average'],
                    model_performance.loc['RF Test R2', 'Average']],
        'ANN Train': [model_performance.loc['ANN Train MSE', 'Average'], model_performance.loc['ANN Train RMSE', 'Average'],
                      model_performance.loc['ANN Train MAE', 'Average'], model_performance.loc['ANN Train MAPE', 'Average'], 
                      model_performance.loc['ANN Train R2', 'Average']],
        'ANN Test': [model_performance.loc['ANN Test MSE', 'Average'], model_performance.loc['ANN Test RMSE', 'Average'],
                     model_performance.loc['ANN Test MAE', 'Average'], model_performance.loc['ANN Test MAPE', 'Average'],
                     model_performance.loc['ANN Test R2', 'Average']],
        'SVR Train': [model_performance.loc['SVR Train MSE', 'Average'], model_performance.loc['SVR Train RMSE', 'Average'],
                      model_performance.loc['SVR Train MAE', 'Average'], model_performance.loc['SVR Train MAPE', 'Average'],
                      model_performance.loc['SVR Train R2', 'Average']],
        'SVR Test': [model_performance.loc['SVR Test MSE', 'Average'], model_performance.loc['SVR Test RMSE', 'Average'],
                     model_performance.loc['SVR Test MAE', 'Average'], model_performance.loc['SVR Test MAPE', 'Average'],
                     model_performance.loc['SVR Test R2', 'Average']],
        'GBDT Train': [model_performance.loc['GBDT Train MSE', 'Average'], model_performance.loc['GBDT Train RMSE', 'Average'],
                       model_performance.loc['GBDT Train MAE', 'Average'], model_performance.loc['GBDT Train MAPE', 'Average'],
                       model_performance.loc['GBDT Train R2', 'Average']],
        'GBDT Test': [model_performance.loc['GBDT Test MSE', 'Average'], model_performance.loc['GBDT Test RMSE', 'Average'],
                      model_performance.loc['GBDT Test MAE', 'Average'], model_performance.loc['GBDT Test MAPE', 'Average'],
                      model_performance.loc['GBDT Test R2', 'Average']],
        'KNN Train': [model_performance.loc['KNN Train MSE', 'Average'], model_performance.loc['KNN Train RMSE', 'Average'],
                      model_performance.loc['KNN Train MAE', 'Average'], model_performance.loc['KNN Train MAPE', 'Average'],
                      model_performance.loc['KNN Train R2', 'Average']],
        'KNN Test': [model_performance.loc['KNN Test MSE', 'Average'], model_performance.loc['KNN Test RMSE', 'Average'],
                     model_performance.loc['KNN Test MAE', 'Average'], model_performance.loc['KNN Test MAPE', 'Average'],
                     model_performance.loc['KNN Test R2', 'Average']],
        }

amp = pd.DataFrame(data, index=['MSE', 'RMSE', 'MAE', 'MAPE %', 'R2']).T

model_performance = model_performance.reset_index()

# %% Backgound Gradient & Export
amp.style.background_gradient(cmap='RdYlGn')

writer = pd.ExcelWriter('amp.xlsx', engine='xlsxwriter')
amp.to_excel(writer, sheet_name='Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']

worksheet.conditional_format('B2:B11', {'type': '3_color_scale'})
worksheet.conditional_format('C2:C11', {'type': '3_color_scale'})
worksheet.conditional_format('D2:D11', {'type': '3_color_scale'})
worksheet.conditional_format('E2:E11', {'type': '3_color_scale'})
worksheet.conditional_format('F2:F11', {'type': '3_color_scale'})

writer.save()
