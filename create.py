# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:54:05 2020

@author: SIDDHARTH
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity

df1 = pd.read_csv('tmdb_5000_credits.csv/tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv/tmdb_5000_movies.csv')

df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on = 'id')
df2.drop(['homepage', 'title_x', 'title_y', 'status','production_countries'], axis=1, inplace=True)

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
    
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 5 elements exist. If yes, return only first five. If no, return entire list.
        if len(names) > 5:
            names = names[:5]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)

df2['overview'] = df2['overview'].fillna('')
df2['soup3'] = df2['soup'] + df2['overview']

count = CountVectorizer(stop_words='english')
count_matrix2 = count.fit_transform(df2['soup3'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim3 = cosine_similarity(count_matrix2, count_matrix2)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', cosine_sim3)

# saving dataframe to csv for later use in main file
df2.to_csv('data.csv',index=False)