#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


train_data = pd.read_csv('train_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)
print(train_data.shape)


# In[7]:


test_data = pd.read_csv('test_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)
print(test_data.shape)


# In[8]:


pd_data = pd.concat([train_data, test_data])
print(pd_data.shape)


# In[11]:


tf_idf = TfidfVectorizer(lowercase=True, max_features=100000)


# In[12]:


tf_idf_vector = tf_idf.fit_transform(pd_data.textbody)
print(tf_idf_vector.shape)


# In[14]:


import scipy.sparse
scipy.sparse.save_npz('tf_idf_vector_data.npz', tf_idf_vector)

