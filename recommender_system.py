# -*- coding: utf-8 -*-
"""recommender_system.ipynb
"""

import numpy as np
import pandas as pd
import requests
import json
from sklearn.metrics import mean_squared_error

#!curl -O http://files.grouplens.org/datasets/movielens/ml-100k.zip
#!unzip ml-100k.zip

#!ls

#!unzip ml-100k.zip

#!cd ml-100k && ls -la

df=pd.read_csv('ml-100k/u.data',sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

df.head()

n_users=df.user_id.unique().shape[0]
n_items=df.item_id.unique().shape[0]
print('Number of users:',n_users,'Number of movies:',n_items)

ratings=np.zeros((n_users,n_items))
for row in df.itertuples():
  ratings[row[1]-1,row[2]-1]=row[3]
ratings

def train_test_split(ratings):
  train_data=ratings.copy()
  test_data=np.zeros(ratings.shape)
  for user in range(ratings.shape[0]):
    test_ratings=np.random.choice(ratings[user,:].nonzero()[0],size=10,replace=False)
    train_data[user,test_ratings]=0
    test_data[user,test_ratings]=ratings[user,test_ratings]
  return train_data,test_data

train,test= train_test_split(ratings)

def similarity(ratings, epsilon=1e-9):
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

user_sim = similarity(train)
print(user_sim[:4, :4])

def predict_fast_simple(ratings, similarity):
  return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T



def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

user_prediction = predict_fast_simple(train, user_sim)

print('User-based CF MSE: ' ,get_mse(user_prediction, test))

def predict(ratings,sim,k=40):
  pred = np.zeros(ratings.shape)
  user_bias = ratings.mean(axis=1)
  ratings = (ratings - user_bias[:, np.newaxis]).copy()
  for i in xrange(ratings.shape[0]):
    top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
    for j in xrange(ratings.shape[1]):
      pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
      pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
  pred += user_bias[:, np.newaxis]
  return pred

#!curl -O http://files.grouplens.org/datasets/movielens/ml-100k/u.item

idx_to_movie = {}
with open('u.item', 'r', encoding = "ISO-8859-1") as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[1]
        
def top_k_movies(similarity, mapper, movie_idx, k=6):
    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

idx = 1
movies = top_k_movies(user_sim, idx_to_movie, idx)
names=[]
for movie in movies:
  names.append(movie)

print(names)

