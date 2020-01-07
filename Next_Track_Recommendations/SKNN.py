#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


data = pd.read_csv('Copy of explicit_data - Session Data - Songs.csv')


# In[17]:


class SKNN:
    def __init__( self, k=100, sample_size=1000, sampling='recent',  similarity = 'cosine', remind=False, pop_boost=0, 
                 extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.session_key = session_key
        self.pop_boost = pop_boost
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        
        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        
        self.sim_time = 0
        
        
    # Trains the predictor
    # Training data : Session Ids, Item Ids and timestamp
    
    def train_data(self, train, items=None):
        index_session = 0 #train.columns.get_loc( self.session_key )
        index_item = 1 #train.columns.get_loc( self.item_key )
        index_time = 2 #train.columns.get_loc( self.time_key )
            
        session = -1
        session_items = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
                
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        
        
        
        
        
        
    # Give prediction scores for a selected set of items on how likely they be the next item in the session
    # output : Prediction scores for selected items on how likely to be the next items of the session. Indexed by the item IDs.
        
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0 ):
        if( self.session != session_id ): #new session
            
            if( self.extend ):
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
                
            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()
            
        if type == 'view':
            self.session_items.append( input_item_id )
        
        if skip:
            return
                        
        neighbors = self.find_neighbors( set(self.session_items), input_item_id, session_id )
        scores = self.score_items( neighbors )
        
        # add some reminders
        if self.remind:
             
            reminderScore = 5
            takeLastN = 3
             
            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                #reminderScore = reminderScore + (cnt/100)
                 
                oldScore = scores.get( elem )
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                #print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt/100)
                 
                scores.update({elem : newScore})
                
        #push popular ones
        if self.pop_boost > 0:
               
            pop = self.item_popularity( neighbors )
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})
                
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        predict_for_item_ids = np.array(predict_for_item_ids)
#         print(predict_for_item_ids)
        
        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=
                           predictions, index=predict_for_item_ids)
        
        if self.normalize:
            series = series / series.max()
        
        return series
    
    # Give the item popularity for the given list of sessions

    
    def item_popularity(self, sessions):
        result = dict()
        max_pop = 0
        for session in sessions:
            items = self.items_for_session( session )
            #print(items)
            for item in items:
                
                #print(item)
                
                count = result.get(item)
                #print(count)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})
                    
                if( result.get(item) > max_pop ):
                    max_pop =  result.get(item)
         
        for key in result:
            #print(max_pop)
            result.update({key: ( result[key] / max_pop )})
                   
        return result
    
    def cosine(self, first, second):
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def random(self, first, second):
        return random.random()
    
    def items_for_session(self, session):
        return self.session_item_map.get(session);
    
    def sessions_for_item(self, item_id):
        return self.item_session_map.get( item_id )
    
    def most_recent_sessions( self, sessions, number ):
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
    
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                
            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
            
    def calc_similarity(self, session_items, sessions ):
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session( session )
            
            similarity = getattr(self , self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))
                
        return neighbors
    
    def find_neighbors( self, session_items, input_item_id, session_id):
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors )
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
    def score_items(self, neighbors):
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            
            for item in items:
                old_score = scores.get( item )
                new_score = session[1]
                
                if old_score is None:
                    scores.update({item : new_score})
                else: 
                    new_score = old_score + new_score
                    scores.update({item : new_score})
                    
        return scores
    
    def results(self, session_id, input_item_id, predict_for_item_ids):
        score = self.predict_next(session_id, input_item_id, predict_for_item_ids)
        score_frame = score.to_frame() 
        score_frame.reset_index(inplace=True)
        score_frame.columns = ['item_id','score']
        sort_by_score = score_frame.sort_values('score',ascending=False)
        sknn_results=[]
        for i in sort_by_score.head(10).item_id:
            sknn_results.append(i)
        return sknn_results
            
    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()

        
        


# In[21]:


# sknn = SKNN()
# sknn.train_data(data)
# ids = data.ItemId.unique()
# sknn.results(session_id = 5,
#                  input_item_id = 174,
#                  predict_for_item_ids = ids)


# In[ ]:


# https://github.com/topics/song-recommender
# https://github.com/caravanuden/spotify_recsys
# https://github.com/topics/music-recommendation
# https://github.com/kartikjagdale/Last.fm-Song-Recommender
# https://github.com/mrthlinh/Spotify-Playlist-Recommender


# # SMF (Session based MF)

# In[84]:


# class SessionMF:
#     def __init__( self, factors=100, batch=50, learn='adagrad_sub', learning_rate=0.001, momentum=0.0, regularization=0.5, 
#                  dropout=0.0, skip=0, samples=2048, activation='linear', objective='bpr_max_org', epochs=10, last_n_days=None, 
#                  session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
#         self.factors = factors
#         self.batch = batch
#         self.learning_rate = learning_rate
#         self.momentum = momentum
#         self.learn = learn
#         self.regularization = regularization
#         self.samples = samples
#         self.dropout = dropout
#         self.skip = skip
#         self.epochs = epochs
#         self.activation = activation
#         self.objective = objective
#         self.last_n_days = last_n_days
#         self.session_key = session_key
#         self.item_key = item_key
#         self.time_key = time_key
        
#         #updated while recommending
#         self.session = -1
#         self.session_items = []
#         self.relevant_sessions = set()

#         # cache relations once at startup
#         self.session_item_map = dict() 
#         self.item_session_map = dict()
#         self.session_time = dict()
        
#         self.item_map = dict()
#         self.item_count = 0
#         self.session_map = dict()
#         self.session_count = 0
        
#         self.floatX = theano.config.floatX
#         self.intX = 'int32'
    
#     def fit(self, data, items=None):
#         self.unique_items = data[self.item_key].unique().astype( self.intX )
#         self.num_items = data[self.item_key].nunique()
#         self.item_list = np.zeros( self.num_items )
        
#         start = time.time()
#         self.init_items(data)
#         print( 'finished init item map in {}'.format(  ( time.time() - start ) ) )
        
#         if self.last_n_days != None:
            
#             max_time = dt.fromtimestamp( data[self.time_key].max() )
#             date_threshold = max_time.date() - td( self.last_n_days )
#             stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
#             train = data[ data[self.time_key] >= stamp ]
        
#         else: 
#             train = data
        
#         self.init_sessions( train )
            
#         start = time.time()
#         self.init_model( train )
#         print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
        
#         start = time.time()
        
#         avg_time = 0
#         avg_count = 0
                   
#         for j in range( self.epochs ):
            
#             loss = 0
#             count = 0
#             hit = 0
            
#             batch_size = set(range(self.batch))
                    
#             ipos = np.zeros( self.batch ).astype( self.intX )
#             #ineg = np.zeros( self.batch ).astype( self.intX )
            
#             finished = False
#             next_sidx = len(batch_size)
#             sidx = np.arange(self.batch)
#             spos = np.ones( self.batch ).astype( self.intX )
#             svec = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
#             smat = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
#             sci = np.zeros( self.batch ).astype( self.intX )
#             scp = np.zeros( self.batch ).astype( self.intX )
                        
#             while not finished:
                
#                 #rand = []
#                 ran = np.random.random_sample()
#                 items = set()
#                 itemsl = None
                
#                 for i in range(self.batch):
                    
#                     item_pos = self.session_map[ self.sessions[ sidx[i] ] ][ spos[i] ]
#                     if ran < self.skip and len(self.session_map[ self.sessions[ sidx[i] ] ]) > spos[i] + 1:
#                         item_pos = self.session_map[ self.sessions[ sidx[i] ] ][ spos[i] + 1 ]
                        
#                     item_current = self.session_map[ self.sessions[ sidx[i] ] ][ spos[i] - 1 ]
#                     #prev = spos[i] - 1 if spos[i] - 1 > 0 else spos[i]
#                     #item_prev = self.session_map[ self.sessions[ sidx[i] ] ][ prev ]
#                     items.update(self.session_map[ self.sessions[ sidx[i] ] ][ :spos[i] ])
#                     #items.extend(self.session_map[ self.sessions[ sidx[i] ] ][ :spos[i] - 1 ])
                    
#                     ipos[i] = item_pos
#                     sci[i] = item_current
#                     #scp[i] = item_current
#                     svec[i][ sci[i] ] = spos[i]
#                     smat[i] = svec[i] / spos[i]
#                     if self.dropout > 0:
#                         itemsl = list(items)
#                         smat[i][itemsl] = smat[i][itemsl] * np.random.choice(2,size=len(itemsl),p=[self.dropout,1-self.dropout])
                        
#                     spos[i] += 1
                
                
#                 if self.samples > 0:
#                     #additional = samples
#                     additional = np.random.randint(self.num_items, size=self.samples).astype( self.intX )
#                     stmp = time.time()
#                     if itemsl is None:
#                         itemsl = list(items)
#                     loss += self.train_model_batch( smat, sci, np.hstack( [ipos, additional] ), itemsl )
#                     avg_time += (time.time() - stmp)
#                     avg_count += 1
#                 else:
#                     loss +=  self.train_model_batch( smat, sci, ipos, scp )
                

#                 if np.isnan(loss):
#                     print(str(j) + ': NaN error!')
#                     self.error_during_train = True
#                     return
                
#                 count += self.batch
                          
#                 for i in range(self.batch):
#                     if len( self.session_map[ self.sessions[ sidx[i] ] ] ) == spos[i]: #session end
#                         if next_sidx < len( self.sessions ):
#                             spos[i] = 1
#                             sidx[i] = next_sidx
#                             svec[i] = np.zeros( self.num_items ).astype( self.floatX )
#                             next_sidx += 1
#                         else:
#                             spos[i] -= 1
#                             batch_size -= set([i])
                    
#                     if len(batch_size) == 0:
#                         finished = True
                            
#             print( 'finished epoch {} with loss {} / hr {} in {}s'.format( j, ( loss / count ), ( hit / count ), ( time.time() - start ) ) )
            
#         print( 'avg_time_fact: ',( avg_time / avg_count ) )
    
#     def init_model(self, train, std=0.01):
        
#         self.I = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I' )
#         self.S = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='S' )
        
#         self.I1 = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I1' )
#         self.I2 = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I2' )

#         self.BS = theano.shared( np.random.normal(0, std, size=(self.num_items,1) ).astype( self.floatX ), name='BS' )
#         self.BI = theano.shared( np.random.normal(0, std, size=(self.num_items,1) ).astype( self.floatX ), name='BI' )
        
#         self.hack_matrix = np.ones((self.batch, self.batch + self.samples), dtype=self.floatX)
#         np.fill_diagonal(self.hack_matrix, 0)
#         self.hack_matrix = theano.shared(self.hack_matrix, borrow=True)
        
#         self._generate_train_model_batch_function()
#         self._generate_predict_function()
#         self._generate_predict_batch_function()
    
#     def init_items(self, train):
        
#         index_item = train.columns.get_loc( self.item_key )
                
#         for row in train.itertuples(index=False):
            
#             ci = row[index_item]
            
#             if not ci in self.item_map: 
#                 self.item_map[ci] = self.item_count
#                 self.item_list[self.item_count] = ci
#                 self.item_count = self.item_count + 1                  
    
#     def init_sessions(self, train):
        
#         index_session = train.columns.get_loc( self.session_key )
#         index_item = train.columns.get_loc( self.item_key )
        
#         self.sessions = []
#         self.session_map = {}
        
#         train.sort_values( [self.session_key,self.time_key], inplace=True )
        
#         prev_session = -1
        
#         for row in train.itertuples(index=False):
            
#             item = self.item_map[ row[index_item] ]
#             session = row[index_session]
            
#             if prev_session != session: 
#                 self.sessions.append(session)
#                 self.session_map[session] = []
            
#             self.session_map[session].append(item)
#             prev_session = session
    
#     def _generate_train_model_batch_function(self):
        
#         s = T.matrix('s', dtype=self.floatX)
#         i = T.vector('i', dtype=self.intX)
#         y = T.vector('y', dtype=self.intX)
#         items = T.vector('items', dtype=self.intX)
        
#         Sit = self.S[items]
#         sit = s.T[items]
        
#         Iy = self.I[y]
#         BSy = self.BS[y]
#         BIy = self.BI[y]
        
#         I1i = self.I1[i]
#         I2y = self.I2[y]
        
#         se = T.dot( Sit.T, sit )
#         predS =  T.dot( Iy, se ).T + BSy.flatten()
        
#         predI = T.dot( I1i, I2y.T ) + BIy.flatten()
        
#         pred = predS + predI
#         pred = getattr(self, self.activation )( pred )
        
#         cost = getattr(self, self.objective )( pred, y )
        
#         #updates = getattr(self, self.learn)(cost, [self.S,self.I,self.IC,self.BI,self.BS], self.learning_rate)
#         updates = getattr(self, self.learn)(cost, [self.S,self.I,self.I1,self.I2,self.BI,self.BS], [Sit,Iy,I1i,I2y,BIy,BSy],[items,y,i,y,y,y], self.learning_rate, momentum=self.momentum)
        
#         self.train_model_batch = theano.function(inputs=[s, i, y, items], outputs=cost, updates=updates  )
    
#     def _generate_predict_function(self):
        
#         s = T.vector('s', dtype=self.floatX)
#         i = T.scalar('i', dtype=self.intX)
        
#         se = T.dot( self.S.T, s.T )
        
#         predS = T.dot( self.I, se ).T + self.BS.flatten()
#         predI = T.dot( self.I1[i], self.I2.T ) + self.BI.flatten()
        
#         pred = predS + predI
#         pred = getattr(self, self.activation )( pred )
        
#         self.predict = theano.function(inputs=[s, i], outputs=pred )
    
#     def _generate_predict_batch_function(self):
        
#         s = T.matrix('s', dtype=self.floatX)
#         i = T.vector('i', dtype=self.intX)
        
#         se = T.dot( self.S.T, s.T )
        
#         predS = T.dot( self.I, se ).T + self.BS
#         predI = T.dot( self.I1[i], self.I2.T ) + self.BI
        
#         pred = predS + predI
        
#         pred = getattr(self, self.activation )( pred )
        
#         self.predict_batch = theano.function(inputs=[s, i], outputs=pred ) #, updates=updates )
        
#     def bpr_max_org(self, pred_mat, y):
#         softmax_scores = self.softmax_neg(pred_mat).T
#         return T.cast(T.mean(-T.log(T.sum(T.nnet.sigmoid(T.diag(pred_mat)-pred_mat.T)*softmax_scores, axis=0)+1e-24)+self.regularization*T.sum((pred_mat.T**2)*softmax_scores, axis=0)), self.floatX)
    
    
#     def softmax_neg(self, X):
#         if hasattr(self, 'hack_matrix'):
#             X = X * self.hack_matrix
#             e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x')) * self.hack_matrix
#         else:
#             e_x = T.fill_diagonal(T.exp(X - X.max(axis=1).dimshuffle(0, 'x')), 0)
#         return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    
#     def adagrad_sub(self, loss, param_list, subparam_list, idx, learning_rate=1.0, epsilon=1e-6, momentum=0.0 ):
        
#         updates = []

#         all_grads = theano.grad(loss, subparam_list)
        
#         for i in range(len(all_grads)):
            
#             grad = all_grads[i]
#             param = param_list[i]
#             index = idx[i]
#             subparam = subparam_list[i]
            
#             accu = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
 
#             accu_s = accu[index]
#             accu_new = accu_s + grad ** 2
#             updates.append( ( accu, T.set_subtensor(accu_s, accu_new) ) )
            
#             delta = learning_rate * grad / T.sqrt(accu_new + epsilon)
            
#             if momentum > 0:
#                 velocity = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
#                 vs = velocity[index]
#                 velocity2 = momentum * vs - delta
#                 updates.append( ( velocity, T.set_subtensor(vs, velocity2) ) )
#                 updates.append( ( param, T.inc_subtensor(subparam, velocity2 ) ) )
#             else:
#                 updates.append( ( param, T.inc_subtensor(subparam, - delta ) ) )
            
#         return updates
    
#     def linear(self, param):
#         return param
    
#     def sigmoid(self, param):
#         return T.nnet.sigmoid( param )
    
#     def predict_next( self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0 ):
#         if( self.session != session_id ): #new session      
                
#             self.session = session_id
#             self.session_items = np.zeros(self.num_items, dtype=np.float32)
#             self.session_count = 0
        
#         if type == 'view':
#             self.session_count += 1
#             self.session_items[ self.item_map[input_item_id] ] = self.session_count
        
#         if skip:
#             return
         
#         predictions = self.predict( self.session_items / self.session_count, self.item_map[input_item_id] )
#         series = pd.Series(data=predictions, index=self.item_list)
#         series = series[predict_for_item_ids]
                
#         return series
    
#     def clear(self):
#         self.I.set_value([[]])
#         self.S.set_value([[]])
#         self.I1.set_value([[]])
#         self.I2.set_value([[]])
#         self.BS.set_value([[]])
#         self.BI.set_value([[]])
#         self.hack_matrix.set_value([[]])
    
    
        


# In[ ]:




