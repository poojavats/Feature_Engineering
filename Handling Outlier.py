#!/usr/bin/env python
# coding: utf-8

# # 5 Number Summary
# 1. Minimum Value
# 2. q1-25 Percentile
# 3. Median
# 4. q3-75 Percentile
# 5 Maximum

# In[2]:


import numpy as np
import pandas as pd


# In[6]:


l_marks=[45,32,56,75,88,1000,88,669,45,78,25,47,65.69,75]
np.percentile(l_marks,[25])


# In[7]:


##[Lower Fence ---> Higher Fence]


# In[9]:


q1=np.percentile(l_marks,[25])
q1


# In[11]:


minimum,q1,q2,q3,max=np.quantile(l_marks,[0,0.25,0.50,0.75,1.0])


# In[13]:


max


# In[15]:


IQR=q3-q1


# In[16]:


IQR


# In[19]:


lower_fence=q1-1.5*(IQR)
higher_fence=q1+1.5*(IQR)


# In[20]:


lower_fence


# In[21]:


higher_fence


# In[ ]:




