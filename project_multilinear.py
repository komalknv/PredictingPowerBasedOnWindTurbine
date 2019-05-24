#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv(r'C:\Users\cse\Desktop\T1.csv')


# In[3]:


dataset


# In[4]:


dataset.describe()


# In[5]:


dataset.fillna(dataset.mean(),inplace=True)


# In[6]:


dataset


# In[7]:


dataset.isnull()


# In[8]:


x=dataset.iloc[:,1:3].values


# In[9]:


x


# In[10]:


y=dataset.iloc[:,3].values


# In[11]:


y


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


labelencoder_x=LabelEncoder()


# In[14]:


x[:,1]=labelencoder_x.fit_transform(x[:,1])


# In[15]:


x


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


x_train


# In[19]:


x_test


# In[20]:


y_train


# In[21]:


y_test


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


mlr=LinearRegression()


# In[24]:


mlr.fit(x_train,y_train)


# In[25]:


y_pred=mlr.predict(x_test)


# In[26]:


mlr.intercept_


# In[27]:


x_train.shape


# In[28]:


mlr.coef_


# In[29]:


pre=mlr.predict([[0,1]])


# In[30]:


pre


# In[31]:


from sklearn.metrics import r2_score


# In[32]:


r2_score(y_test,y_pred)


# In[ ]:




