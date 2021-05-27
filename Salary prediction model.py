#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


ds = pandas.read_csv("Salary_Data.csv")


# In[3]:


ds


# In[4]:


ds.info()


# In[5]:


x = ds["YearsExperience"]


# In[6]:


type(x)


# In[7]:


x=x.values


# In[8]:


type(x)


# In[9]:


x=x.reshape(30,1)


# In[10]:


x


# In[11]:


y = ds["Salary"]


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=42)


# In[15]:


X_test


# In[16]:


y_test


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model.fit(X_test,y_test)


# In[20]:


model.coef_


# In[21]:


model.predict([[2.5]])


# In[22]:


model.predict(X_test)


# In[23]:


# y = wx
model.predict([[10.5]])


# In[24]:


model.predict([[1.1]])


# In[25]:


# correctness
36187/39343 * 100
# 91 % 


# In[26]:


model.predict([[0]])


# In[27]:



model.intercept_


# In[28]:



model.predict([[1.1]])


# In[29]:


y_pred = model.predict(X_test)


# In[ ]:





# In[30]:


import matplotlib.pyplot as plt


# In[31]:


plt.scatter(X_test, y_test)


# In[32]:


plt.scatter(X_test, y_test,color="red")
plt.plot(X_test, y_pred)
plt.xlabel("years of experience")
plt.ylabel("salary")


# In[ ]:





# In[ ]:





# In[ ]:




