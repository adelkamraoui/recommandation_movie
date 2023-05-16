from flask import Flask, request, render_template
from pandas_profiling import ProfileReport
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
import ydata_profiling
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg
import os
from flask import Flask, render_template, request
from test import get_next_id,recommend_movies,remove_year,suggest_diverse_movies,tuple_list_to_dataframe,dataframe_to_tuple_list,get_movie_title,get_movie_trailer,get_movie_description,get_movie_url_from_csv
import requests
import time
from flask import Flask, render_template, request, session, redirect, url_for
import hashlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
app = Flask(__name__,template_folder='template')
app.secret_key = 'secret_key'
field_names = ['userId', 'movieId', 'rating',
               'timestamp']	


users = {}
with open('output.csv') as f:
       for line in f.readlines():
        user_id, username, password,role = line.strip().split(',')
        users[username] = (user_id, password,role)


@app.route('/login', methods=['GET', 'POST'])
def login():
    with open('output.csv') as f:
        for line in f.readlines():
            user_id, username, password, role = line.strip().split(',')
            users[username] = (user_id, password, role)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and hashlib.sha256(password.encode()).hexdigest() == users[username][1]:
            session['username'] = username
            session['user_id'] = users[username][0]
            
            if users[username][2] == 'admin':
                print('admin salem alikoum')
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('main'))
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)
    else:
        return render_template('login.html')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    with open('output.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        max_id = max(int(row[0]) for row in reader)
    
    if request.method == 'POST':
        # Generate the user ID as the maximum user ID plus one
        user_id = max_id + 1
        username = request.form['username']
        password = request.form['password']
        role='normal'
        # Check if the username is already taken
        with open('output.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if row[1] == username:
                    error = f'Username {username} is already taken'
                    return render_template('signin.html', error=error)
        
        # Add the new user to the CSV file
        with open('output.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, username, hashlib.sha256(password.encode()).hexdigest(),role])
        
        # Redirect the user to the login page
        return redirect(url_for('showfilms', id=user_id))

    else:
        return render_template('signin.html', max_id=max_id+1)

@app.route('/')
def main():
    if 'username' in session:

        user_id = session['user_id']
        userIdd=int(user_id)
        rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
        movie = pd.read_csv('/home/adel/Desktop/kn/movies.csv')
        df = pd.merge(rating, movie, on='movieId')
        eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
        eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
        mtrx_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
        print('avant')
        print(mtrx_df.head(30))
        mtrx_df.head(30)
        


        mtrx = mtrx_df.to_numpy()
        ratings_mean = np.mean(mtrx, axis = 1)
        normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(normalized_mtrx, k = 50)
        sigma = np.diag(sigma)
        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
        print('apres')
        print(preds_df.head(10))
        preds_df.head(10)
        already_rated, predictions = recommend_movies(preds_df,userIdd, movie, rating, 10)
        
        print(already_rated.head(10))
        print(predictions)
        movie_data = predictions

        poster_urls = []
        api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key

        for prediction in predictions['title']:
            poster_url=get_movie_url_from_csv('/home/adel/Desktop/kn/movieswithurl.csv', prediction)
            poster_urls.append(poster_url)

        predictions['urls']=poster_urls        
        print(predictions)
        return render_template('listrecomendations.html', predictions=predictions,id=user_id,poster_urls=poster_urls)

            
    else:
        return redirect(url_for('login'))
#@app.route('/')
#def index():
 #   return render_template('nsiyi.html')

@app.route('/recomfilms')
def enteridrecom():
    return render_template('recommand.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/Factorisation')
def Factorisation():
    rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
    movie = pd.read_csv('/home/adel/Desktop/kn/movies.csv')
    df = pd.merge(rating, movie, on='movieId')
    eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
    eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    mtrx_df = rating.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    page = request.args.get('page', default=1, type=int)
    chunk_size = 10  # Number of rows per page
    start_idx = (page - 1) * chunk_size
    end_idx = start_idx + chunk_size

    mtrx_df_page = mtrx_df.iloc[start_idx:end_idx]
    mtrx_df_html = mtrx_df_page.to_html()

    mtrx = mtrx_df.to_numpy()
    ratings_mean = np.mean(mtrx, axis=1)
    normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(normalized_mtrx, k=50)
    sigma = np.diag(sigma)
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings, columns=mtrx_df.columns)

    preds_df_page = preds_df.iloc[start_idx:end_idx]
    preds_df_html = preds_df_page.to_html()

    num_pages = int(np.ceil(len(mtrx_df) / chunk_size))

    return render_template('factorisation_en_temps_reel.html', mtrx_df_html=mtrx_df_html, preds_df_html=preds_df_html, page=page, num_pages=num_pages)


@app.route('/ajoutfilm', methods=['GET', 'POST'])
def ajoutfilm():
    
    if request.method == 'POST':
        # Récupérer les données du formulaire
        title = request.form['title']
        genres = request.form['genres']

        # Générer l'ID du film
        movie_id = get_next_id()
        print('je suis le maxxxxxx')
        print(movie_id)
        # Ajouter le nouveau film au fichier CSV
        CSV_FILE = '/home/adel/Desktop/kn/movies.csv'
        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([movie_id, title, genres])
    
        return 'Film ajouté avec succès !'
    
    
    return render_template('ajoutfilm.html')
@app.route('/supprimefilm', methods=['GET', 'POST'])
def supprimefilm():
    if request.method == 'POST':
        # Get the film name from the form
        film_name = request.form['film_name']

        # Delete the film from the CSV file
        CSV_FILE = '/home/adel/Desktop/kn/movies.csv'
        updated_rows = []
        with open(CSV_FILE, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] != film_name:  # Exclude the film with the specified name
                    updated_rows.append(row)

        with open(CSV_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)

        return 'Film deleted successfully!'

    return render_template('supprimefilm.html')
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
    api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'
    idd=id
    print(idd)
    title=get_movie_title(movieId)
    pre=remove_year(title)
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={pre}'
    response = requests.get(url)
    data = response.json()
    if data['total_results'] > 0:
        poster_path = data['results'][0]['poster_path']
        poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
        print("+1")
        
    else:
        poster_url = 'https://i.quotev.com/b2gtjqawaaaa.jpg'  # Replace with your actual error image path
            
        print("+1 taa errror")

    movieidd=movieId
    urll=get_movie_trailer('ccfc2af2a0cd4597bf0472fab1af2f02',pre)
    print(movieidd)
    description=get_movie_description(pre,'ccfc2af2a0cd4597bf0472fab1af2f02')
    return render_template('ratingg.html', idd=idd,movieidd=movieidd,title=title,poster_url=poster_url,urll=urll,description=description)
@app.route('/allfilms/<int:id>/<int:movieId>/commit', methods=['POST'])
def ratee(id,movieId):
    rating = request.form['rating']
    
    # ts stores the time in seconds
    ts = time.time()
 
    # print the current timestamp
    print(ts)
    timestamp = ts
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


@app.route('/recom/<int:idd>', methods=['GET'])

def recommand_movies(idd):
    idd=idd
    userIdd=int(idd)
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
    return render_template('listrecomendations.html', predictions=predictions,id=idd,poster_urls=poster_urls)

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
    '''
    already_rated, li1 = recommend_movies(preds_df, idd, movie, rating, 20)
    pertinences = []

    

    recom = dataframe_to_tuple_list(li1)

    # Calcul du score de diversité pour chaque film
    diversity_scores = []
    for i in range(len(recom)):
        genres_i = set(recom[i][2].split("|"))
        diversity_sum = 0
        for j in range(len(recom)):
            if i != j:
                genres_j = set(recom[j][2].split("|"))
                distance = 1 - (len(genres_i.intersection(genres_j)) / len(genres_i.union(genres_j)))
                diversity_sum += distance
            diversity_score = diversity_sum / (len(recom) - 1)
            diversity_scores.append((recom[i][0], recom[i][1], recom[i][2], diversity_score))

        # Trier les films par ordre décroissant de score de diversité et afficher les 10 premiers
        sorted_diversity_scores = sorted(diversity_scores, key=lambda x: x[3], reverse=True)
        diversified = tuple_list_to_dataframe(sorted_diversity_scores)
        diversified = diversified.head(10)
        
        
        # Calculer le nombre de films communs entre la liste initiale et la liste diversifiée
        films_communs = set(li1['title']).intersection(set(diversified10['title']))

        nb_films_communs = len(films_communs)

        # Calculer le pourcentage de pertinence
        pourcentage_pertinence = nb_films_communs / len(li1) * 100
        # Afficher le résultat
        print("La pertinence de la liste diversifiée avec {} films est de : {:.2f}%".format(n, pourcentage_pertinence))

        # Ajouter la pertinence à la liste
        pertinences.append(pourcentage_pertinence)
    
    print(pertinences)
    plt.plot(range(10, 51), pertinences)
    plt.xlabel('Nombre de films recommandés')
    plt.ylabel('Pertinence (%)')
    plt.title('Évolution de la pertinence en fonction du nombre de films recommandés')
    plt.show()    
      '''                                                                                                                      
    already_rated, list1 = recommend_movies(preds_df,idd, movie, rating, 10)
   
    already_rated, predictions = recommend_movies(preds_df,idd, movie, rating, 20)
    recom=dataframe_to_tuple_list(predictions)
    
    # Calcul du score de diversité pour chaque film
    diversity_scores = []
    for i in range(len(recom)):
        genres_i = set(recom[i][2].split("|"))
        diversity_sum = 0
        for j in range(len(recom)):
            if i != j:
                genres_j = set(recom[j][2].split("|"))
                distance = 1 - (len(genres_i.intersection(genres_j)) / len(genres_i.union(genres_j)))
                diversity_sum += distance
        diversity_score = diversity_sum / (len(recom) - 1)
        diversity_scores.append((recom[i][0], recom[i][1], recom[i][2], diversity_score))

    # Trier les films par ordre décroissant de score de diversité et afficher les 10 premiers
    print('ici')
    
    sorted_diversity_scores = sorted(diversity_scores, key=lambda x: x[3], reverse=True)
    
    
    print(tuple_list_to_dataframe(sorted_diversity_scores))
    diversified=tuple_list_to_dataframe(sorted_diversity_scores)
    '''
    print('look adel i made it')
    reco_initiale=list1
    reco_diversifiee=diversified
    films_communs = set(reco_initiale).intersection(set(reco_diversifiee))
    nb_films_communs = len(films_communs)

    # Calculez le pourcentage de pertinence
    pourcentage_pertinence = nb_films_communs / len(reco_initiale) * 100

    print("La pertinence de la liste diversifiée est de : {:.2f}%".format(pourcentage_pertinence))
    '''
    poster_urls = []
    api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key
    diversified = diversified.head(10)
    for diversifiedd in diversified['title']:
            poster_url=get_movie_url_from_csv('/home/adel/Desktop/kn/movieswithurl.csv', diversifiedd)
            poster_urls.append(poster_url)
    

    diversified['urls']=poster_urls        
    
    return render_template('diver.html', diversified=diversified,poster_urls=poster_urls)

if __name__ == '__main__':
    app.run(debug=True)