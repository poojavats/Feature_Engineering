#!/usr/bin/env python
# coding: utf-8

# # Two Types of Techniques to Handled Imbalance data set
# 1. Downsampling- Try to decrease the value whose value is high
# 2. UpSampling- Try to Increase the value whose Value is low

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


np.random.seed(123)  #Random Value should not be changed
#Create the DataFrame with two classes
n_samples=1000
class_0_ratio=0.9
n_class_0= int(n_samples*class_0_ratio)
n_class_1= n_samples-n_class_0


# In[3]:


n_class_0,n_class_1


# In[4]:


class_0=pd.DataFrame({'Feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
                     'Feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
                     'target':[0]*n_class_0})
class_1=pd.DataFrame({'Feature_1':np.random.normal(loc=2,scale=1,size=n_class_1),
                     'Feature_2':np.random.normal(loc=2,scale=1,size=n_class_1),
                     'target':[1]*n_class_1})


# In[5]:


class_0


# In[6]:


class_1


# In[7]:


df=pd.concat([class_0,class_1]).reset_index(drop=True)


# In[8]:


df.head()


# In[9]:


df['target'].value_counts()


# # Upsampling

# In[10]:


df_minority=df[df['target']==1]
df_majority=df[df['target']==0]


# In[11]:


df_minority


# In[12]:


df_majority


# In[13]:


##Upsamping Perform
from sklearn.utils import resample


# In[14]:


df_minority_upsample=resample(df_minority,replace=True, ##sample with replacement
                             n_samples=len(df_majority), ##to match the majority class
                                          random_state=42)


# In[15]:


df_minority_upsample


# In[16]:


df_minority_upsample.value_counts()


# In[17]:


df_minority_upsample['target'].value_counts()


# In[18]:


df_upsample=pd.concat([df_majority,df_minority_upsample])


# In[19]:


df_upsample


# In[20]:


df_upsample['target'].value_counts()


# In[21]:


df_upsample.head()


# # DownSampling
# 
# 

# In[22]:


class_0=pd.DataFrame({'Feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
                     'Feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
                     'target':[0]*n_class_0})
class_1=pd.DataFrame({'Feature_1':np.random.normal(loc=2,scale=1,size=n_class_1),
                     'Feature_2':np.random.normal(loc=2,scale=1,size=n_class_1),
                     'target':[1]*n_class_1})


# In[23]:


df=pd.concat([class_0,class_1]).reset_index(drop=True)


# In[24]:


df.head()


# In[25]:


df_minority=df[df['target']==1]
df_majority=df[df['target']==0]


# In[26]:


df_majority_downsample=resample(df_majority,replace=False, ##sample without replacement
                             n_samples=len(df_minority), ##to match the Minority class
                                          random_state=42)


# In[27]:


df_majority_downsample


# In[28]:


df_majority_downsample.shape


# In[31]:


df_downsample=pd.concat([df_minority,df_majority_downsample])


# In[32]:


df_downsample['target'].value_counts()


# In[ ]:




