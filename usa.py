#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


df=pd.read_csv(r"F:\dataset\USA_Housing.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


vf=df.drop(['Address'],axis=1, inplace=True)


# In[7]:


df


# In[8]:


df.columns


# In[9]:


# Putting feature variable to X
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[10]:


X


# In[11]:


# Putting response variable to y
y=df['Price']


# In[12]:


y


# In[13]:


# Let's plot a pair plot of all variables in our dataframe
sns.pairplot(df)


# In[15]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(df,x_vars=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'],y_vars=['Price'])


# In[16]:


sns.heatmap(df.corr(),annot=True)


# In[17]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7 ,test_size = 0.3,random_state=2)


# In[19]:


X_train


# In[20]:


X_test


# In[21]:


y_train


# In[22]:


y_test.shape


# In[23]:


y_train.values


# In[24]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Importing RFE and LinearRegression


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lr=LinearRegression()


# In[27]:


lr.fit(X_train.values,y_train.values)


# In[28]:


lr.intercept_


# In[29]:


# Let's see the coefficient
coeffi=pd.DataFrame(lr.coef_,X_test.columns,columns=["coefficenct"])
coeffi.round()


# In[53]:


# Making predictions using the model
y_pred=lr.predict(X_test)
y_pred


# In[31]:


from sklearn.metrics import mean_squared_error,r2_score


# In[32]:


mse=mean_squared_error(y_test,y_pred)
r_sq=r2_score(y_test,y_pred)


# In[33]:


print('mean_squared_error:',mse)
print('r2:',r_sq)


# In[ ]:


# using statsmodel 


# In[34]:


import statsmodels.api as sm


# In[35]:


#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm=sm.add_constant(X_train)
lm=sm.OLS(y_train,X_train_sm).fit()


# In[36]:


lm.params


# In[37]:


print(lm.summary())


# In[38]:


X_train.drop(["Avg. Area Number of Bedrooms"],axis=1,inplace=True)


# In[39]:


X_test.drop(["Avg. Area Number of Bedrooms"],axis=1,inplace=True)


# In[40]:


X_train


# In[41]:


lr.fit(X_train,y_train)


# In[42]:


lr.intercept_


# In[43]:


# Making predictions using the model
y_pred=lr.predict(X_test)
y_pred


# In[44]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[45]:


r=r2_score(y_pred,y_test)
r


# In[49]:


# Actual and Predicted
c = [i for i in range(1,1501,1)] # generating index 
fig = plt.figure(figsize=(12,8))
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label

