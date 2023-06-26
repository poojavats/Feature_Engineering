#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification


# In[4]:


X,y=make_classification(n_samples=1000,n_features=2,n_redundant=0,n_clusters_per_class=1,weights=[0.90],random_state=1)


# In[5]:


X


# In[6]:


y


# In[7]:


import pandas as pd
import numpy as np


# In[15]:


df=pd.DataFrame(X,columns=['f1','f2'])
df2=pd.DataFrame(y,columns=['target'])
final_df=pd.concat([df,df2],axis=1)


# In[16]:


final_df.head()


# In[17]:


final_df['target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(final_df['f1'],final_df['f2'],c=final_df['target'])


# In[21]:


get_ipython().system('pip install imblearn')


# In[22]:


from imblearn.over_sampling import SMOTE


# In[25]:


# Transform dataset
oversample=SMOTE()
X,y=oversample.fit_resample(final_df[['f1','f2']],final_df['target'])


# In[29]:


X.shape,y.shape


# In[30]:


len(y[y==0]),len(y[y==1])


# In[31]:


df=pd.DataFrame(X,columns=['f1','f2'])
df2=pd.DataFrame(y,columns=['target'])
Oversampled_final_df=pd.concat([df,df2],axis=1)


# In[33]:


plt.scatter(Oversampled_final_df['f1'],Oversampled_final_df['f2'],c=Oversampled_final_df['target'])


# In[ ]:




