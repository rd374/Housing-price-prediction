# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:26:23 2024

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Advanced-House-Price-Prediction--6672da4c4c4afd9548954c42f61c8c3c42f07684/Advanced-House-Price-Prediction--6672da4c4c4afd9548954c42f61c8c3c42f07684/train.csv")
print((df.head()))
# split the data in two files to ensure no leakages
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(df,df['SalePrice'],test_size=0.1, random_state=0)
print(x_train.shape,x_test.shape)

# Capturing null values 
feature_with_nullvalues= [feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes=='O']
for feature in feature_with_nullvalues:
                              print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))
# replacing missing value with a new label
def replace_cat_feature(df,feature_with_nullvalues):
    data=df.copy()
    data[feature_with_nullvalues]=data[feature_with_nullvalues].fillna('Missing')
    return data
df= replace_cat_feature(df, feature_with_nullvalues)
print(df[feature_with_nullvalues].isnull().sum())
pd.set_option('display.max_columns', None)
print(df.head())
# numerical feature with null values
numerical_with_nullvalues= [feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes!='O']
for feature in numerical_with_nullvalues:
    print("{}: {}% missing value".format(feature,np.around(df[feature].isnull().mean(),4)))
 # replacing the numerical missing values with median
for feature in numerical_with_nullvalues:
    median_value= df[feature].median()
# capturing nan value
    df[feature+'nan']= np.where(df[feature].isnull(),1,0)
    df[feature].fillna(median_value,inplace=True)
df[numerical_with_nullvalues].isnull().sum()
print(df[numerical_with_nullvalues].isnull().sum())
# temporal variables as yrsold price waas reducing
pd.set_option('display.max_columns', None)
print(df.head())
for feature in ['GarageYrBlt','YearRemodAdd','YearBuilt' ]:
    df[feature]= df['YrSold']-df[feature]
pd.set_option('display.max_columns', None)
print(df.head())
#skewed numerical feature taking numerical feature doesnot have 0 value for analysis
import numpy as np
num_features= ['LotFrontage','LotArea','GrLivArea','SalePrice','1stFlrSF']
# log normal distribution
for feature in num_features:
    df[feature]= np.log(df[feature])
print(df.head())
# rare category feature: we will remove categorical variable that are present less than 1% of observation
categorical_feature=[feature for feature in df.columns if df[feature].dtypes=='O']
print(categorical_feature)
# rare variable creating percentage with total observation
for feature in categorical_feature:
    temp= df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df= temp[temp>0.01].index
    df[feature]= np.where(df[feature].isin(temp_df),df[feature],'Rare_var')
pd.set_option('display.max_columns', None)
print(df.head(10))
import pandas as pd
# convert all non-numeric data into numeric data
from sklearn.preprocessing import LabelEncoder
dft = pd.DataFrame(df)
le = LabelEncoder()

for column in dft.columns:
    if dft[column].dtype == 'object':
        dft[column] = le.fit_transform(dft[column])
print(dft)
from sklearn.preprocessing import StandardScaler
# feature scaling because diff unit of measurement
scaler = StandardScaler()
feature_scale = [feature for feature in dft.columns if feature not in ['Id','SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(dft[feature_scale])
MinMaxScaler(copy=True, feature_range=(0,1))
# transform the train and test set and drop id and saleprice
data= pd.concat([dft[['Id','SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dft[feature_scale]),columns=feature_scale)],
                   axis=1)
print(data.head())
data.to_csv('x_train.csv',index=False)

dataset= pd.read_csv('x_train.csv')
# dependent variable
y_train= dataset[['SalePrice']]
# dropping dependent feature 
x_train= dataset.drop(['Id','SalePrice'],axis=1)
# apply lesso regression model and suitable alpha=0.005
# transformed based on weights - lesso
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
feature_sel_model= SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x_train, y_train)
SelectFromModel(estimator=Lasso(alpha=0.005,copy_X=True , fit_intercept=True,max_iter=1000,
                                normalize=False, positive=False, precompute=False, random_state=0,
                                selection= 'cyclic', tol=0.0001,warm_start=False),
                                max_features=None, norm_order=1,prefit=False, threshold=None)
print(feature_sel_model.get_support())
# true indicates that feature is important and false indicate not important
#

selected_feat=x_train.columns[(feature_sel_model.get_support())]
#
print('total features:{}'.format(x_train.shape[1]))
print('selected features:{}'.format(len(selected_feat)))
print('features with coefficients shrank to zero:{}'.format(np.sum(feature_sel_model.estimator_.coef_==0)))
print(selected_feat)
x_train=x_train[selected_feat]
print(x_train.head())

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train,y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
lasso_model = Lasso(alpha=0.005, max_iter=1000, tol=0.0001, normalize=True)
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.linear_model import Lasso
param_grid = {'alpha':[0.005,0.001,0.1]}
alphas_array = np.array(param_grid['alpha'])
alphas, coefs, _ = lasso_model.path(x_train, y_train, alphas= alphas_array)
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i].T)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Paths')
plt.show()
## residual
residuals = y_train - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()




