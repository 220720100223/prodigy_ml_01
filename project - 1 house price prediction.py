#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

    


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[3]:


data = pd.read_csv(r"D:\house price prediction\train.csv")
data.head


# In[4]:


# Remove unnecessary columns
columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']
data.drop(columns_to_drop, axis=1, inplace=True)


# In[5]:


# Handle missing values (if any)
data.dropna(inplace=True)  # Drop rows with missing values

# Split data into features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']


# In[6]:


categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]


# In[7]:


numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]


# In[8]:


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# In[9]:


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[10]:


X_preprocessed = preprocessor.fit_transform(X)


# In[11]:


X_preprocessed = pd.DataFrame(X_preprocessed)


# In[12]:


# Step 2: Split the Data
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


# In[13]:


xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


# In[14]:


# Train the model
xgb_reg.fit(X_train, y_train)


# In[15]:


# Predict on validation set
y_pred = xgb_reg.predict(X_val)


# In[16]:


# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)


# In[17]:


# Step 5: Model Interpretation (Optional)
# Analyze feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_reg.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance)


# In[18]:


# Step 1: Load Test Data
test_data = pd.read_csv(r"D:\house price prediction\test.csv")

# Step 2: Preprocess Test Data
X_test_preprocessed = preprocessor.transform(test_data)

# Step 3: Make Predictions
y_test_pred = xgb_reg.predict(X_test_preprocessed)

# Step 4: Save Predictions (Optional)
# Assuming you want to save predictions to a CSV file
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': y_test_pred})
output.to_csv('predictions.csv', index=False)


# In[19]:


predictions = pd.read_csv('predictions.csv')
print(predictions.head())


# In[ ]:




