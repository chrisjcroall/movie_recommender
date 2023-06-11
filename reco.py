import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from operator import itemgetter
import tmdbsimple as tmdb
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import pickle
from PIL import Image
import cv2
from transformers import pipeline
import requests as req
import openai
import streamlit as st
import streamlit_chat as message
import time


#IMAGE/LOGO
image = Image.open("Screenshot 2023-05-10 104047.png")
st.image(image, use_column_width=True, width=200)
# st.title("Movie Recommender")
# st.subheader("For Movies You Want")

#VIDEO
video_file = open("The Ultimate Movie Montage - An Epic Journey.mp4", 'rb')
video_bytes = video_file.read()
st.video(video_bytes, start_time=0)

#THEME

# Primary accent for interactive elements
primaryColor = '#7792E3'

# Background color for the main content area
backgroundColor = '#273346'

# Background color for sidebar and most interactive widgets
secondaryBackgroundColor = '#B9F1C0'

# Color used for almost all text
textColor = '#FFFFFF'

# Font family for all text in the app, except code blocks
# Accepted values (serif | sans serif | monospace) 
font = "monospace"

#LOADING THE DATASETS

#links
links_df = pd.read_csv("C:/Users/chris/OneDrive/Desktop/WBS_Data_Science/8_Recommender_Systems/wbsflix-dataset/ml-latest-small/links.csv")
#movies
movies_df = pd.read_csv("C:/Users/chris/OneDrive/Desktop/WBS_Data_Science/8_Recommender_Systems/wbsflix-dataset/ml-latest-small/movies.csv")
#ratings
ratings_df = pd.read_csv("C:/Users/chris/OneDrive/Desktop/WBS_Data_Science/8_Recommender_Systems/wbsflix-dataset/ml-latest-small/ratings.csv")
#tags
tags_df = pd.read_csv("C:/Users/chris/OneDrive/Desktop/WBS_Data_Science/8_Recommender_Systems/wbsflix-dataset/ml-latest-small/tags.csv")

################ Item-based ###################

# Create pivot table
movies_crosstab = pd.pivot_table(
    data=ratings_df, values='rating', index='userId', columns='movieId')


#FUNCTION 1 - FETCHING POSTER
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=f1eea307d6c66202ebbfffa98e674e2a&language=en-US".format(movie_id)
    data = req.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    st.image(full_path)
    return full_path

#RECOMMENDATION
n = st.sidebar.slider("Select number of recommendations", min_value=5, max_value=50, value=10)

# Item based 1: Top 10 based on correlation
# Input: movie ID
# Output: 10 recommended movies
def sim_mov(mov_id, n):
    # Get ratings from sparse matrix (excluding NaNs)
    mov_ratings = movies_crosstab[mov_id]
    # mov_ratings[mov_ratings>=0] # exclude NaNs
    
    # Get movies correlating with input
    similar_to_mov = movies_crosstab.corrwith(mov_ratings)
    
    # Create df with movies correlating with input
    corr_mov = pd.DataFrame(similar_to_mov, columns=['PearsonR'])
    corr_mov.dropna(inplace=True)
    
    # Create df with rating means and counts
    ratings = pd.DataFrame(ratings_df.groupby('movieId')['rating'].mean())
    ratings['rating_count'] = ratings_df.groupby('movieId')['rating'].count()
    
    # Join df df with restaurants correlating with input and df with rating means and counts
    mov_corr_summary = corr_mov.join(ratings['rating_count'])
    mov_corr_summary.drop(mov_id, inplace=True) # drop input rest from df
    
    # Get top ten restaurants with rating count of at least 10 and highest Pearson corr. coeff.
    top_n = mov_corr_summary[mov_corr_summary['rating_count']>=40].sort_values('PearsonR', ascending=False).head(n)
    
    # Merge top_n df with movies_df (containing title and genre)
    top_n = top_n.merge(movies_df, left_index=True, right_on="movieId")
    output_table = top_n.loc[:,["title","genres"]]
    st.table(output_table)
    #fetching the poster
    list = top_n.movieId.tolist()
    for i in list:
        x = links_df.loc[links_df["movieId"]==i, "tmdbId"].to_list()
        y = int(x[0])
        fetch_poster(y)


#CHANGING THE LAYOUT OF THE POSTERS
user_inputid = st.sidebar.multiselect("Type in user ID:", ratings_df.userId.unique())


users_items = pd.pivot_table(data=ratings_df, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')

# Replace NaN with zero
users_items.fillna(0, inplace=True)

# Compute cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)

#FUNCTION 2 - SIMILAR USERS, USER ID
def sim_user(user_id, n):
    # Compute weights for user_id
    my_weights = (
    user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
          )
    
    # print weights sum (must be 1.0)
    print(f"Sum of weihts: {my_weights.sum()}")
    
    # Create df with movies that the inputed user has not rated
    not_rated_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
    #not_rated_movies.T
    
    # Compute dot product between the not-rated-movies and the weights
    my_weighted_averages = pd.DataFrame(not_rated_movies.T.dot(my_weights), columns=["predicted_rating"])
    #my_weighted_averages.head()
    
    # Get top n movies from rating predictions
    my_recommendations = my_weighted_averages.merge(movies_df, left_index=True, right_on="movieId")
    my_recommendations = my_recommendations.sort_values("predicted_rating", ascending=False).head(n)
    output_table = my_recommendations.loc[:,["title","genres"]]

    st.table(output_table)
    list_2 = my_recommendations.movieId.tolist()
    for i in list_2:
        x = links_df.loc[links_df["movieId"]==i, "tmdbId"].to_list()
        y = int(x[0])
        fetch_poster(y)

for id in user_inputid:
    st.write(f"### Given your recently watched movies, you may also like these:")
    # Get value for current movieId and store as (only element) in a list
    ##current_id = movies_df.loc[movies_df["title"]==id, "movieId"].to_list()

    sim_user(id, n)


#SIDEBAR - USER INPUT
movies_you_like = st.sidebar.multiselect("Select / Type in some movies you like:", movies_df.title)
ids_of_movies_you_like = []


for movie in movies_you_like:
    st.write(f"### Since you like {movie}, you may also like these movies:")
    # Get value for current movieId and store as (only element) in a list
    current_id = movies_df.loc[movies_df["title"]==movie, "movieId"].to_list()

    # Call function passing in first (and only) element in list with current id
    sim_mov(current_id[0], n)

else:
    st.write("Please type in a movie title")

#FETCH POSTER FROM TMDB API
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=f1eea307d6c66202ebbfffa98e674e2a&language=en-US".format(movie_id)
    data = req.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    st.image(full_path)
    return full_path

#DEFINING THE APP
def app():
    # Set Streamlit app title
    st.title("Movie Poster Viewer")

    # Define movie selection dropdown
    movie_options = {
        "The Shawshank Redemption": 278,
        "The Godfather": 238,
        "The Dark Knight": 155,
        "Pulp Fiction": 680,
        "Fight Club": 550
    }
    selected_movie = st.selectbox("Select a movie:", list(movie_options.keys()))

    # Get movie poster URL and display image if available
    movie_id = movie_options[selected_movie]
    poster_url = get_movie_poster_url(movie_id)
    if poster_url:
        st.image(poster_url, caption=selected_movie, use_column_width=True)
        
        # Add link to review website
        review_url = "https://www.themoviedb.org/movie/review?movie={}".format(selected_movie)
        st.markdown("[Review this movie]({})".format(review_url))
    else:
        st.text("Sorry, no poster available.")