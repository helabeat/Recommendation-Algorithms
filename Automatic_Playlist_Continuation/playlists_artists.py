#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# In[39]:


playlists = pd.read_csv('../Datasets/Copy of explicit_data - Playlist_data.csv')
user_playlist_data = pd.read_csv('../Datasets/Copy of explicit_data - User_playlists.csv')
songs = pd.read_csv('../Datasets/Copy of explicit_data - Songs - All.csv')


# In[40]:


user_playlist_data.columns = ['playlist_id','timestamp','order','song_id']


# In[10]:



def merge_data(user_playlist_data, songs):
    merged_data = pd.merge(user_playlist_data, songs.drop_duplicates(['song_id']), on="song_id", how="left")
    
    merged_data['ratings'] = np.ones((merged_data.shape[0],), dtype=int)
    return merged_data
    
def get_playlist_name(playlist_id):
    df_merge = merge_data(user_playlist_data, songs)
    playlist_name = df_merge.loc[df_merge['playlist_id']==playlist_id]
    return playlist_name


# In[51]:


def train_playlist_songs_with_artists(playlist_id):
    playlist_name = get_playlist_name(playlist_id)
    artist_list = playlist_name.Artist.values
    similarity, vocab = generate_bow(artist_list)
    df=pd.DataFrame(similarity,columns=vocab)
    quantity = df.sum(axis = 0)
    quantity = pd.DataFrame(quantity)
    quantity.reset_index(level=0, inplace=True)
    quantity.columns = ['string','frequency']
    max_artist = quantity['string'].loc[quantity['frequency'] == max(quantity['frequency']) ]
    return [w for w in max_artist]

def get_similar_songs(playlist_id, songs):
    artist_list = songs.Artist.values
    frequent_artist = train_playlist_songs_with_artists(playlist_id)
    #print(frequent_artist[0])
    uniques = []
    for i in artist_list:
        #print(i)
        if(Naive_based_search(frequent_artist[0], i)):
            suggestions = songs['song_id'].loc[songs['Artist'] == i]
            #print(suggestions)
            for idx in suggestions:
                uniques.append(idx)
        else:
            continue
    uniques = list(dict.fromkeys(uniques))
    return uniques

def Naive_based_search(pat, txt): 
        M = len(pat) 
        N = len(txt) 

        # A loop to slide pat[] one by one */ 
        for i in range(N - M + 1): 
            j = 0        
            # For current index i, check  
            # for pattern match */ 
            while(j < M): 
                if (txt[i + j] != pat[j]): 
                    break
                j += 1

            if (j == M):  
                return True
            else:
                return False
    
def word_extraction(sentence):       
    words = re.sub("[^\w]", " ",  sentence).split()    
    cleaned_text = [w for w in words]    
    return cleaned_text

def tokenize(sentences):   
    words = []    
    for sentence in sentences:        
        w = word_extraction(sentence)        
        words.extend(w)            
        words = sorted(list(set(words)))    
        return words

def generate_bow(allsentences):        
    vocab = tokenize(allsentences)    
    # print("Word List for Document \n{0} \n".format(vocab));
    vector_array = []
    for sentence in allsentences:        
        words = word_extraction(sentence)        
        bag_vector = np.zeros(len(vocab))        
        for w in words:            
            for i,word in enumerate(vocab):                
                if word == w:                     
                    bag_vector[i] += 1                            
    #         print("{0}\n{1}\n".format(sentence,np.array(bag_vector)))
        vector_array.append(np.array(bag_vector))
    return vector_array, vocab


# In[ ]:


get_similar_songs(109, songs)


# In[ ]:




