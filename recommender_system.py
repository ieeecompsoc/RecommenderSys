# -*- coding: utf-8 -*-
"""recommender_system.ipynb

Dataset used is Movielens dataset, available at: http://files.grouplens.org/datasets/movielens/
"""

import numpy as np
import pandas as pd
import requests
import json
from sklearn.metrics import mean_squared_error


df=pd.read_csv('ml-100k/u.data',sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

n_users=df.user_id.unique().shape[0]   #number of unique users
n_items=df.item_id.unique().shape[0]   #number of unique items
#print('Number of users:',n_users,'Number of movies:',n_items)

ratings=np.zeros((n_users,n_items))
for row in df.itertuples():
  ratings[row[1]-1,row[2]-1]=row[3]

def train_test_split(ratings):      #function to split training and testing data
  train_data=ratings.copy()
  test_data=np.zeros(ratings.shape)
  for user in range(ratings.shape[0]):
    test_ratings=np.random.choice(ratings[user,:].nonzero()[0],size=10,replace=False)
    train_data[user,test_ratings]=0
    test_data[user,test_ratings]=ratings[user,test_ratings]
  return train_data,test_data

train,test= train_test_split(ratings)

def similarity(ratings, epsilon=1e-9):   #function to calculate similarity between users
  sim = ratings.dot(ratings.T) + epsilon
  norms = np.array([np.sqrt(np.diagonal(sim))])
  return (sim / norms / norms.T)

user_sim = similarity(train)

def predict_fast_simple(ratings, similarity):
  return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T



def get_mse(pred, actual): #function to calculate mean square error
  pred = pred[actual.nonzero()].flatten()
  actual = actual[actual.nonzero()].flatten()
  return mean_squared_error(pred, actual)

user_prediction = predict_fast_simple(train, user_sim)

def predict(ratings,sim,k=40):  #funtion to make prediction
  pred = np.zeros(ratings.shape)
  user_bias = ratings.mean(axis=1)
  ratings = (ratings - user_bias[:, np.newaxis]).copy()
  for i in xrange(ratings.shape[0]):
    top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
    for j in xrange(ratings.shape[1]):
      pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
      pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
  pred+= user_bias[:, np.newaxis]
  return pred

idx_to_movie = {}
with open('u.item', 'r', encoding = "ISO-8859-1") as f:
  for line in f.readlines():
    info = line.split('|')
    idx_to_movie[int(info[0])-1] = info[1]
        
def top_k_movies(similarity, mapper, movie_idx, k=6): #function to return the recommended top k movies 
  return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

def recommend(idx,user_sim,idx_to_movie):  #function to recommend the movies through given movie's id
  idx = 0 #id of movie Toy Story in available dataset
  movies = top_k_movies(user_sim, idx_to_movie, idx)
  names=[]
  for movie in movies:
    names.append(movie)

  print(names)

