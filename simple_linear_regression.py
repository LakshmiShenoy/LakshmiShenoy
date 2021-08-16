#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression - example by Lakshmi Shenoy

# ### Business area: HR & Recruitment 
# ### Problem Statement: HR asks - How do I predict the Salary of employees - in the org and new hires?
# ### Solution: Salary prediction can depend on multiple factors like :  Yrs of experience, Educational Qualification, Skills of Candidate, Current Market situation, Economic Situation, Financial Situation of Orgnazation and so on its an endless list
# ### ML Model: Different types of Regressions coupled with Dimention reduction can be applied
# ### In this notebook we will look at Simple Linear Regression where we have 1 independent variable which is yrs of experience and salary is being predicted accordingly
# ### 

# ## Importing the libraries

# In[20]:


#Here we import 3 basic libraries needed for this notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[4]:


# we have a Salaries dataset in git which we import -save it to you folder where you save the notebook
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[5]:


# We import train_test_split to Split arrays or matrices into random train and test subset
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# ## Training the Simple Linear Regression model on the Training set

# In[6]:


# More info available at https://scikit-learn.org/stable/modules/linear_model.html
# Similar to implementation done in 1st sem by Prof GN using excel - in this notebook its python code
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[7]:


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
y_pred = regressor.predict(X_test)


# ## Visualising the Training set results

# In[8]:


# More info at https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
# Here we do a basic visualization of how our train dataset looks like - how the regression line fits
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Test set results

# In[10]:


# Now we use the test data set to see if the seults can be predicted
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[14]:


# The coefficients
print('Coefficients: \n', regressor.coef_)


# In[16]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[26]:


r_sq = regressor.score(X_train, y_train)
print('coefficient of determination:', r_sq)
print('intercept:', regressor.intercept_)
print('slope:', regressor.coef_)

