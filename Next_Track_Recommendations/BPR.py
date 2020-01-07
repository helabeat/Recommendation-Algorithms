#!/usr/bin/env python
# coding: utf-8

# In[1]:


from _operator import itemgetter
from math import sqrt
import random
import time
import numpy as np
import pandas as pd
import os
import psutil
import gc

from pympler import asizeof
from math import log10
import scipy.sparse
from scipy.sparse.csc import csc_matrix
from datetime import datetime as dt
from datetime import timedelta as td


# In[ ]:


class BPR:
    def __init__(self, n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None
    
    def init(self, data):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)
    
    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)
    
    def fit(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init(data)
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
#             print(it, np.mean(c))
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view',
                     timestamp=0):
        iidx = self.itemidmap[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.itemidmap[predict_for_item_ids]
        return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_item_ids)
    
    def results(self, session_id, input_item_id, predict_for_item_ids):
        score = self.predict_next(session_id, input_item_id, predict_for_item_ids)
        score_frame = score.to_frame() 
        score_frame.reset_index(inplace=True)
        score_frame.columns = ['item_id','score']
        sort_by_score = score_frame.sort_values('score',ascending=False)
        predictions = []
        for i in sort_by_score.head(10).item_id:
            predictions.append(i)
        return predictions
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    


# In[ ]:


# bpr = BPR()
# bpr.fit(data)
# ids = data.ItemId.unique()
# bpr.results(session_id = 7,
#                 input_item_id = 224,
#                 predict_for_item_ids = ids)

