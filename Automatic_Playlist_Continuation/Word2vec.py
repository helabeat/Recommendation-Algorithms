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


# In[2]:


playlists = pd.read_csv('../Datasets/Copy of explicit_data - Playlist_data.csv')
user_playlist_data = pd.read_csv('../Datasets/Copy of explicit_data - User_playlists.csv')
songs = pd.read_csv('../Datasets/Copy of explicit_data - Songs - All.csv')


# In[3]:


user_playlist_data.columns = ['playlist_id','timestamp','order','song_id']
# df_merge = pd.merge(user_playlist_data, songs.drop_duplicates(['song_id']), on="song_id", how="left")

# df_merge['ratings'] = np.ones((df_merge.shape[0],), dtype=int)


# In[4]:


# playlists with particular set of songs using a word2vec model
def get_playlist_name(playlist_id, df_merge):
    playlist_name = df_merge.loc[df_merge['playlist_id']==playlist_id]
    return playlist_name

def get_bag_of_words(playlist_id, df_merge):
    playlist_name = get_playlist_name(playlist_id, df_merge)
    songs_names = playlist_name.Title.values
    song_name_clean = [re.sub(r'[^\w]', ' ', str(item))for item in songs_names]
    song_name_clean = [re.sub(r" \d+", '', str(item.strip())) for item in song_name_clean]
        
    sentences = list()
    bof = []
    for item in song_name_clean:
        sentences.append(item.split())
    unique_sentence = np.unique(sentences)
        
    for i in range (len(unique_sentence)):
        for j in range (len(unique_sentence[i])):
            bof.append(unique_sentence[i][j])
    return unique_sentence, sentences, bof

def word2vec_model_build(playlist_id, df_merge):
    unique_sentence, sentences, bof = get_bag_of_words(playlist_id, df_merge)
    model = Word2Vec(workers=1,             size=50, min_count = 1,             window = 3, sample = 1e-3, sg = 1)
    model.build_vocab(sentences = unique_sentence)
    model.train(sentences = unique_sentence,  total_examples=len(sentences), epochs=10)
    model.init_sims(replace=True)
    return model

def avg_sentence_vector(song_id, playlist_id, df_merge, num_features = 50):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    song = songs['Title'].loc[songs['song_id'] == song_id]
    for i in song:
        words = i.split()
    unique_sentence, sentences, bof = get_bag_of_words(playlist_id, df_merge)
    model = word2vec_model_build(playlist_id, df_merge)
    for word in words:
        if word in bof:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def calculate_cosine_similarity_pairwise(song_id, playlist_id, df_merge):
    playlist_name = get_playlist_name(playlist_id, df_merge)
    first_trac_id = playlist_name.song_id.values[0]
    first_track_vector = avg_sentence_vector(first_trac_id, playlist_id, df_merge)
    suggestion_vectors = avg_sentence_vector(song_id, playlist_id, df_merge)
    
    sen1_sen2_similarity =  cosine_similarity(first_track_vector.reshape(1, -1),suggestion_vectors.reshape(1, -1))
    
    return sen1_sen2_similarity

def make_suggestion(songs, playlist_id, df_merge):
    song_list = songs.song_id.values
    suggestions = []
    for idx in song_list:
        similarity_score = calculate_cosine_similarity_pairwise(idx, playlist_id, df_merge)
        for value in similarity_score:
            if value>0:
                suggestions.append(idx)
            else:
                continue
    return suggestions

# find for similar playlists and find item similarity


# In[5]:


# make_suggestion(songs, 103, df_merge)


# In[ ]:




