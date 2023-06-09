#!/usr/bin/env python
# coding: utf-8

# # Missing Values Occurs in dataset when some of the values or information not stored in dataset
# There are 3 Machanisms
# 1. Missing Completely at Random'(MCAR)
# 2. Missing at Random(MAR)
# 3. Missing data not at Random (MNAR)
# 

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


df=sns.load_dataset('titanic')


# In[4]:


df.head()


# In[6]:


##how to check missing value in dataset
df.isnull()
#true-Misssing value in Dataset
#False-No missing Value in Dataset


# In[10]:


# it will tell you how much missing data in columnwise
df.isnull().sum()


# In[13]:


# check by graph missing value
sns.heatmap(df.isnull())


# In[15]:


#Handling missing Values
df.head()


# # by using dropna()this will delete all null values but it will delete all rows also-So this is not a good practice

# In[24]:


#rowwise deletaion
df.dropna()


# In[25]:


df.shape


# # Handling missing values by deleting Column wise it will delete all columns which have null values

# In[30]:



df1=df.dropna(axis=1)


# In[31]:


df1.columns


# In[32]:


df.columns


# # Some More Techniques- Imputation Techniques
# 1. Mean Value Imputation-This Technique work well when our Data is Normally Distributed

# In[40]:


sns.distplot(df['age'])


# In[41]:


df.age.isnull().sum()


# In[42]:


df['Age_mean']=df['age'].fillna(df['age'].mean())


# In[43]:


df[['Age_mean','age']]


# # 2. Median Value Imputation
# This Technique work well when we have Outlier In our data

# In[46]:


df['age_median']=df['age'].fillna(df['age'].median())


# In[47]:


df[['age_median','age']]


# # Mode Value Imputation
# This technique used for Categorial data

# In[50]:


df['embarked_mode']=df['embarked'].fillna(df['embarked'].mode())


# In[51]:


df[['embarked_mode','embarked']]


# In[54]:


df[df['embarked'].isnull()]


# In[55]:


df['embarked'].isnull().sum()


# In[56]:


df['embarked'].unique()


# In[62]:


df[df['age'].notna()]


# In[63]:


df[df['age'].notna()]['embarked'].mode()


# In[69]:


mode=df[df['age'].notna()]['embarked'].mode()[0]


# In[72]:


df['embarked_mode']= df['embarked'].fillna(mode)


# In[74]:


df[['embarked','embarked_mode']]


# In[75]:


df['embarked_mode'].isnull().sum()


# In[ ]:




