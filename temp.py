# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


import seaborn as sns
# display all columns
import pandas as pd

# Read the Excel file
df = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Advanced-House-Price-Prediction--6672da4c4c4afd9548954c42f61c8c3c42f07684/Advanced-House-Price-Prediction--6672da4c4c4afd9548954c42f61c8c3c42f07684/train.csv')
# Display the first few rows of the dataframe
print(df.head(10))
print(df.shape)
print(df.columns)
df.columns
# figuring out missing values 
features_with_nullvalues= [features for features in df.columns if df[features].isnull().sum()>1]
for feature in features_with_nullvalues: print(feature, np.round(df[feature].isnull().mean(),4), '% missing values')
null_values = df.isnull().sum()
print("Number of null values in each column:\n", null_values)
# relationship between missing values and sales price
for feature in features_with_nullvalues: data= df.copy()
# creating a variable that indicates 1 when info is missing else 0 (to check the relation b/w dependent/independent variale)
data[feature]= np.where(data[feature].isnull(),1,0)

data.groupby(feature)['SalePrice'].median().plot.bar()
plt.title('salesprice and null values')
plt.show()
data.groupby(feature)['MSSubClass'].median().plot.bar()
plt.title('MSSubClass and null values')
plt.show()
# here with the relation b/w the missing values and dependent variable is visible
# need to replace the null values into something meaningful using Feature Engineering section
# numerical values
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Number of numerical variables',len(numerical_features))
# visualise numerical variables
df[numerical_features].head()
print(numerical_features)
# temporal variables (4)- datetime variables or no of days analytical purposes 
year_feature = [ feature for feature in numerical_features if 'Yr'in feature or 'Year' in feature]
print(year_feature)
# exploring content of year 
for feature in year_feature: print(feature, df[feature].unique())
# year of house hold and sales price (dependent variable) analysing through the median of sales price
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House price vs year sold')
#  interp: sales price is decreasing as year increases opposite to hypothesis

for feature in year_feature:
	if feature != 'YrSold':
		data = df.copy()
		print("Copy created")
data[feature]=data['YrSold']-data[feature]
# Normalize the data
plt.scatter(data[feature], data['SalePrice'])
plt.xlabel('feature')
plt.ylabel('SalePrice')
plt.show()
data = df.copy()
data['years difference']= data['YrSold']-data['YearRemodAdd']
print(data['years difference'])
plt.scatter(data['years difference'], data['SalePrice'])
plt.xlabel('Yrdiff_Yrsold_YearRemodAdd')
plt.ylabel('SalePrice')
plt.show()
# numerical variable continous or discrete
discrete_feature= [feature for feature in numerical_features if len(df[feature].unique())<25 and feature not in year_feature+['Id']]
print("discrete variables count: {}".format(len(discrete_feature)))
print(discrete_feature)
print(df[discrete_feature].head())
      # lets find relation b/w them and sales price
for feature in discrete_feature:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.show()
# exponentially rising relationship b/w sale price and overallQual (monotonic relationship)
for feature in discrete_feature:
    data = df.copy()
    print("Copy created")
    data.groupby(('OverallQual'))['SalePrice'].median().plot.bar()
plt.xlabel('overallQual')
plt.ylabel('SalePrice')
plt.show()
# zig-zag relation
for feature in discrete_feature:
    data = df.copy()
    print("Copy created")
    data.groupby(('OverallCond'))['SalePrice'].median().plot.bar()
plt.xlabel('OverallCond')
plt.ylabel('SalePrice')
plt.show()   
#some kind of relation b/w discrete and sale price
# continous variables
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))
 # plotting the continous variables     
# interpret: distribution is skewed
for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()  
# since the distribution is skewed use logarithmic transformation
for feature in continuous_feature:
    data= df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel('feature')
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()
# interpretation : monotonic relationship when variable inc sales price inc

# Outliers
for feature in continuous_feature:
    data= df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel("feature")
        plt.title(feature)
        plt.show()
# interpretation 
# categorical variables
categorical_features=[feature for feature in df.columns if data[feature].dtypes=='O']
print(categorical_features)
df[categorical_features].head()
# unique no of categories 
for feature in categorical_features:
    print('The Feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))
  # relationship b/w categorical variable and dependent variable
for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    














      
      
      