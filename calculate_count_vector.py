#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[84]:


train_data = pd.read_csv('train_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)
print(train_data.shape)


# In[85]:


test_data = pd.read_csv('test_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)
print(test_data.shape)


# In[86]:


pd_data = pd.concat([train_data, test_data])
print(pd_data.shape)


# In[71]:


vectorizer = CountVectorizer(lowercase=True, max_features=100000)


# In[73]:


count_vector = vectorizer.fit_transform(pd_data.textbody)
print(count_vector.shape)


# In[79]:


import scipy.sparse
scipy.sparse.save_npz('count_vector_data.npz', count_vector)

