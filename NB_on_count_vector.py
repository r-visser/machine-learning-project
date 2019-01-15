#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import scipy
from sklearn.naive_bayes import MultinomialNB


# In[8]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def evaluation_scores(y_pred, y_test):
    precision = precision_score(y_test, y_pred, average='macro')
    print("precision: "+str(precision))
    recall = recall_score(y_test, y_pred, average='macro') 
    print("recall: "+str(recall))
    f_score_class = f1_score(y_test, y_pred, average=None)
    print("f-score per class: "+str(f_score_class))
    f_macro = f1_score(y_test, y_pred, average='macro')  
    print("f-score macro: "+str(f_macro))
    acc = accuracy_score(y_test, y_pred)
    print("accuracy score: "+str(acc))


# In[41]:


train_data = pd.read_csv('train_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)


# In[42]:


test_data = pd.read_csv('test_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)


# In[47]:


sparse_matrix = scipy.sparse.load_npz('count_vector_data.npz')


# In[48]:


x_train = sparse_matrix[:train_data.shape[0]]
y_train = train_data.hyperpartisan


# In[49]:


x_test = sparse_matrix[train_data.shape[0]:]
y_test = test_data.hyperpartisan


# In[50]:


clf = MultinomialNB()
model = clf.fit(x_train, y_train)


# In[51]:


y_pred = model.predict(x_test)
evaluation_scores(y_pred, y_test)

