from flask import Flask, request, render_template
# Necessary libraries
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For matrix factorization
from scipy.sparse.linalg import svds
import csv 
from csv import DictWriter
from flask import Flask , render_template
import csv
import movieposters as mp
import urllib.request
from PIL import Image
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg
import os
from flask import Flask, render_template, request
from test import recommend_movies,remove_year,suggest_diverse_movies
import requests
app = Flask(__name__,template_folder='template')
field_names = ['userId', 'movieId', 'rating',
               'timestamp']	
@app.route('/')
def index():
    return render_template('nsiyi.html')
@app.route('/recomfilms')
def enteridrecom():
    return render_template('recommand.html')

@app.route('/allfilms/<int:id>')
def showfilms(id):
    movie = pd.read_csv('/home/adel/Desktop/kn/movieswithurl.csv')
    search = request.args.get('search', '')
    if search:
        idd=id
        
        # filter the movie DataFrame based on the search term
        filtered_films = movie[movie.apply(lambda row: search.lower() in row.to_string().lower(), axis=1)]
        page = request.args.get('page', default=1, type=int)
        # slice the filtered DataFrame based on the requested page
        films_per_page = 20
        num_pages = (len(filtered_films) + films_per_page - 1) // films_per_page
        current_page = int(request.args.get('page', '1'))
        start_index = (current_page - 1) * films_per_page
        end_index = start_index + films_per_page
        films_to_display = filtered_films.iloc[start_index:end_index]
        
        return render_template('films.html',movie=films_to_display,idd=idd,num_pages=num_pages,current_page=page, search=search)

    else:
        idd=id
        films_per_page=20
        page = request.args.get('page', default=1, type=int)
        start_index = (page - 1) * films_per_page
        end_index = start_index + films_per_page
        films_to_display = movie[start_index:end_index]
        num_pages = int(len(movie) / films_per_page) + (len(movie) % films_per_page > 0)
        

   
    return render_template('films.html',movie=films_to_display,idd=idd,num_pages=num_pages, current_page=page,search=search)
@app.route('/allfilms/<int:id>/<int:movieId>')
def ratingtemp(id,movieId):
    idd=id
    print(idd)
    movieidd=movieId
    print(movieidd)
    return render_template('ratingg.html', idd=idd,movieidd=movieidd)
@app.route('/allfilms/<int:id>/<int:movieId>/commit', methods=['POST'])
def ratee(id,movieId):
    rating = request.form['rating']
    timestamp = request.form['timestamp']
    # Store the ratings in a database or file
    # For this example, we will just print them to the console
    userId=id 
    movieIdd=movieId
    dict = {'userId':userId, 'movieId':movieIdd, 'rating':rating,
               'timestamp':timestamp}
 

    with open('ratings.csv', 'a') as f_object:

        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(dict)
 
    # Close the file object
    f_object.close()
    print(f"{userId}: {rating}/10")
    print(f"{movieId}: {rating}/10")
    

    return render_template('returnto.html', userId=userId,movieId=movieId)

@app.route('/rate')
def rate_movies():
    rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
    largest_id = rating['userId'].max()
    userId=largest_id+1
    # Store the ratings in a database or file
    # For this example, we will just print them to the console
    
    return render_template('index.html', userId=userId)


@app.route('/recom', methods=['POST'])

def recommand_movies():
    userId = request.form['userId']
    userIdd=int(userId)
    rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
    movie = pd.read_csv('/home/adel/Desktop/kn/movies.csv')
    df = pd.merge(rating, movie, on='movieId')
    eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
    eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    mtrx_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    mtrx = mtrx_df.to_numpy()
    ratings_mean = np.mean(mtrx, axis = 1)
    normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(normalized_mtrx, k = 50)
    sigma = np.diag(sigma)
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
    already_rated, predictions = recommend_movies(preds_df,userIdd, movie, rating, 10)
    
    print(already_rated.head(10))
    print(predictions)
    movie_data = predictions

    poster_urls = []
    api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key

    for prediction in predictions['title']:
        pre=remove_year(prediction)
        url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={pre}'
        response = requests.get(url)
        data = response.json()
        if data['total_results'] > 0:
            poster_path = data['results'][0]['poster_path']
            poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
            print("+1")
            poster_urls.append(poster_url)
        else:
            poster_url = 'https://i.quotev.com/b2gtjqawaaaa.jpg'  # Replace with your actual error image path
            poster_urls.append(poster_url)    
            print("+1 taa errror")

    predictions['urls']=poster_urls        
    print(predictions)
    return render_template('listrecomendations.html', predictions=predictions,id=userId,poster_urls=poster_urls)

@app.route('/diver/<int:idd>')

def diversification(idd):
    idd=idd
    rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
    movie = pd.read_csv('/home/adel/Desktop/kn/movies.csv')
    df = pd.merge(rating, movie, on='movieId')
    eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
    eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    mtrx_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    mtrx = mtrx_df.to_numpy()
    ratings_mean = np.mean(mtrx, axis = 1)
    normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(normalized_mtrx, k = 50)
    sigma = np.diag(sigma)
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
    already_rated, predictions = recommend_movies(preds_df,idd, movie, rating, 20)
    
    diversifie=suggest_diverse_movies(already_rated,predictions)
    diversified=diversifie[1:10]
    print(diversified)

    poster_urls = []
    api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key

    for diversifiedd in diversified['title']:
        pre=remove_year(diversifiedd)
        url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={pre}'
        response = requests.get(url)
        data = response.json()
        if data['total_results'] > 0:
            poster_path = data['results'][0]['poster_path']
            poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
            print("+1")
            poster_urls.append(poster_url)
        else:
            poster_url = 'https://i.quotev.com/b2gtjqawaaaa.jpg'  # Replace with your actual error image path
            poster_urls.append(poster_url)    
            print("+1 taa errror")

    diversified['urls']=poster_urls        
    
    return render_template('diver.html', diversified=diversified,poster_urls=poster_urls)

if __name__ == '__main__':
    app.run(debug=True)