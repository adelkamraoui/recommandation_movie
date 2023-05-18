from flask import Flask, render_template, request, session, redirect, url_for
import requests, csv
import numpy as np
from final_algo import *

field_names = ['userId', 'movieId', 'rating','timestamp']	

def get_users():
    users = {}
    with open('user_credentials.csv') as f:
        for line in f.readlines():
            user_id, username, password = line.strip().split(',')
            users[username] = (user_id, password)
    return users

def get_app():
    app = Flask(__name__,template_folder='template')
    app.secret_key = 'secret_key'

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        users= get_users()
            
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if username in users and users[username][1] == password:
                session['username'] = username
                session['user_id'] = users[username][0]
                return redirect(url_for('main'))
            else:
                error = 'Invalid username or password'
                return render_template('login.html', error=error)
        else:
            return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        with open('user_credentials.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            max_id = max(int(row[0]) for row in reader)
        
        if request.method == 'POST':
            # Generate the user ID as the maximum user ID plus one
            user_id = max_id + 1
            username = request.form['username']
            password = request.form['password']
            
            # Check if the username is already taken
            with open('user_credentials.csv', 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in reader:
                    if row[1] == username:
                        error = f'Username {username} is already taken'
                        return render_template('signin.html', error=error)
            
            # Add the new user to the CSV file
            with open('user_credentials.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_id, username, password])
            
            # Redirect the user to the login page
            return redirect(url_for('showfilms', id=user_id))

        else:
            return render_template('signin.html', max_id=max_id+1)

    @app.route('/')
    def main():
        if 'username' in session:

            user_id = session['user_id']
            userIdd=int(user_id)
            
            rating, movie= load_data()
            mtrx_df, mtrx_np= get_matrix(rating)
            normalized_mtrx, transform_back= normalize_matrix(mtrx)
            all_predicted_ratings= apply_factorization(normalized_mtrx)
            all_predicted_ratings= transform_back(all_predicted_ratings)
            
            preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
            already_rated, predictions = recommend_movies(preds_df,userIdd, movie, rating, 10)
            
            predictions['urls']= get_posters(predictions['title'])    
            
            return render_template('listrecomendations.html', predictions=predictions,id=user_id,poster_urls=predictions['urls'])
        
        else:
            return redirect(url_for('login'))
    #@app.route('/')
    #def index():
    #   return render_template('nsiyi.html')

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
        return render_template('ratingg.html', idd=idd,movieidd=movieidd,title=title,poster_url=poster_url,urll=urll)

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

        rating, movie= load_data()
        mtrx_df, mtrx_np= get_matrix(rating)
        normalized_mtrx, transform_back= normalize_matrix(mtrx)
        all_predicted_ratings= apply_factorization(normalized_mtrx)
        all_predicted_ratings= transform_back(all_predicted_ratings)
        
        preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
        already_rated, predictions = recommend_movies(preds_df,userIdd, movie, rating, 10)
        
        predictions['urls']= get_posters(predictions['title'])    
        return render_template('listrecomendations.html', predictions=predictions,id=idd,poster_urls=predictions['urls'])


    @app.route('/diver/<int:idd>')
    def diversification(idd):
        idd=idd
        
        rating, movie= load_data()
        mtrx_df, mtrx_np= get_matrix(rating)
        normalized_mtrx, transform_back= normalize_matrix(mtrx)
        all_predicted_ratings= apply_factorization(normalized_mtrx)
        all_predicted_ratings= transform_back(all_predicted_ratings)
        preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)
        already_rated, predictions = recommend_movies(preds_df,idd, movie, rating, 20)
        
        diversity_scores= apply_diversity(predictions)

        # Trier les films par ordre décroissant de score de diversité et afficher les 10 premiers
        
        sorted_diversity_scores = sorted(diversity_scores, key=lambda x: x[3], reverse=True)
        diversified=tuple_list_to_dataframe(sorted_diversity_scores)
        diversified['urls']=get_posters(diversified['title'])     
        
        return render_template('diver.html', diversified=diversified,poster_urls=diversified['urls'])
    
    return app