# %% Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %% Import Data
# Import from bf4extension
bf4 = pd.read_csv('bf4extension.csv')
# Import from VariableReductionTemplateBF4
# Only BF4 Extended Data PDP Analysis Sheet
vrt = pd.read_excel('VariableReductionTemplateBF4.xlsx',
                    sheet_name='BF4 Extended Data PDP analysis', header=1)
# Replace spaces in column headers with '_'
vrt.columns = [c.replace(' ', '_') for c in vrt.columns]

# %% Prepare DataFrame for model
# Create list of important/unimportant variables
# Important
ivars = vrt.Variable_Name_[vrt.BASE_BF4 == 1]

# Unimportant
uivars = vrt['Variable_Name_'][vrt['BASE_BF4'] == 0]

# Remove unimportant variables and NaN rows from bf4
bf4r = bf4.drop(columns=uivars)
bf4r = bf4r.dropna()

# Save and remove data time column
bf4_datetime = pd.to_datetime(bf4r.datetime_day)
bf4r = bf4r.drop(columns='datetime_day')

# Define output value and remove
outputs = bf4r['eta_co_daily_calc']
bf4r_inputs = bf4r.drop(columns='eta_co_daily_calc')

# Saving bf4reduced names for later use
bf4r_list = list(bf4r_inputs.columns)

# %% Train and test split
# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
train_bf4_mdl, test_bf4_mdl, train_outputs, test_outputs = train_test_split(
        bf4r_inputs, outputs, test_size=0.25, random_state=None)

# Check all the correct size
# print('Training bf4 Shape:', train_bf4_mdl.shape)
# print('Training outputs Shape:', train_outputs.shape)
# print('Testing bf4 Shape:', test_bf4_mdl.shape)
# print('Testing outputs Shape:', test_outputs.shape)

# %% Setting baseline
# The baseline predictions are the historical averages
# baseline_preds = np.mean(bf4r.eta_co_daily_calc)
# Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - bf4r.eta_co_daily_calc)
# print('Average Baseline Error: ', round(np.mean(baseline_errors), 10))