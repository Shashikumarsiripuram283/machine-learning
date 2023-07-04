#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("C:\\Users\\Shashi kumar\\Desktop\\houseprice_prediction.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.info()


# In[9]:


df = df.drop(['stories','hotwaterheating','airconditioning'],axis="columns")


# In[10]:


df


# In[11]:


Q1 = df.price.quantile(0.25)
Q3 = df.price.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.price >= Q1 - 1.5*IQR) & (df.price <= Q3 + 1.5*IQR)]


# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[13]:


le


# In[14]:


df['mainroad'] = le.fit_transform (df['mainroad'])
df['guestroom'] = le.fit_transform (df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['prefarea'] = le.fit_transform (df['prefarea'])
df['furnishingstatus'] = le.fit_transform (df['furnishingstatus'])


# In[15]:


df


# In[16]:


x = df.drop(['price'], axis='columns')
x


# In[17]:


y = df['price']
y


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=10)


# In[19]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[20]:


model


# In[21]:


model.fit(x_train,y_train)


# In[26]:


model.predict([[9320,3,0,1,1,1,0,1,2]])


# In[ ]:





# In[ ]:




