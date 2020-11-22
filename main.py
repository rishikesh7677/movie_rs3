# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:32:07 2020

@author: SIDDHARTH
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



#sim = np.load('similarity_matrix.npy')
df2 = pd.read_csv('data.csv')

def create_sim():
    
    # creating a count matrix
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df2['soup3'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return sim

sim = create_sim()
# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['original_title'])

def get_rec2(title, sig=sim):
    if title not in df2['original_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        idx = indices[title]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores, key = lambda x: x[1], reverse=True)
        topn = sig_scores[1:11]
        movie_indices = [i[0] for i in topn]
        lst = []
        df = df2['original_title'].iloc[movie_indices]
        return df.values

print(get_rec2('Spectre'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = get_rec2(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
