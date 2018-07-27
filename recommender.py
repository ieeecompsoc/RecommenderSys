import pandas as pd
import numpy as np
import math

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('Desktop/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

#Reading ratings file:

data_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
data = pd.read_csv('Desktop/ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')

#Reading items file:

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
 
items = pd.read_csv('Desktop/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

utrain = (data.sort_values('user_id'))[:99832]
utest = (data.sort_values('user_id'))[99833:]

utrain=utrain.as_matrix(columns=['user_id','movie_id','rating'])
utest=utest.as_matrix(columns=['user_id','movie_id','rating'])

users_list = []
for i in range(1,943):
    list=[]
    for j in range(0,len(utrain)):
        if utrain[j][0] == i:
            list.append(utrain[j])
        else:
            break
    utrain=utrain[j:]
    users_list.append(list)
    
def EucledianScore(train_user, test_user):
    sum=0
    count=0
    for i in test_user:
        score=0
        for j in train_user:
            if(int(i[1]))==int(j[1]):
                score = ((float(i[2])-float(j[2]))*(float(i[2])-float(j[2])))
                count+=1
            sum+=score
    if (count<4):
        sum = 1000000
    return (math.sqrt(sum))

score_list=[]
for i in range(0,942):
    score_list.append([i+1, EucledianScore(users_list[i],utest)])

score = pd.DataFrame(score_list, columns=['user_id','Eucledian Score'])
score = score.sort_values(by = 'Eucledian Score')
print(score)

score_matrix = score.as_matrix()   


user = int(score_matrix[0][0])
common_list = []
full_list = []
for i in utest:
    for j in users_list[user-1]:
        if(int(i[1]) == int(j[1])):
            common_list.append(int(j[1]))
        full_list.append(int(j[1]))
    
common_list = set(common_list)
full_list = set(full_list)

recommendation = full_list.difference(common_list)
