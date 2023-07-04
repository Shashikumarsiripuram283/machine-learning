#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


iris = load_iris()


# In[4]:


dir(iris) 


# In[5]:


iris.data


# In[6]:


iris.target 


# In[7]:


iris.target_names 


# In[8]:


iris.feature_names


# In[9]:


import pandas as pd
df = pd.DataFrame(iris.data, columns= iris.feature_names) 


# In[10]:


df


# In[11]:


df['target'] = iris.target


# In[12]:


df.head()


# In[13]:


import matplotlib.pyplot as plt 
import seaborn as sns
sns.pairplot (df, hue= 'target', palette="brg") 
plt.show ()


# In[14]:


x = df.drop(['target'], axis = 'columns')
x


# In[15]:


y = df.target
y


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=10)


# In[18]:


from sklearn.svm import SVC
model = SVC() 


# In[19]:


model


# In[20]:


model.fit(x_train, y_train)


# In[21]:


model.score (x_test, y_test) 


# In[22]:


model.predict ([iris.data [ 50]]) 


# In[23]:


model.predict([[7.7, 2.6, 6.9, 2.3]]) 

