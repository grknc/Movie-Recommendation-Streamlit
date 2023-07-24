import pickle
import streamlit as st
import requests
import os
from PIL import Image
from dotenv import load_dotenv
import pandas as pd
from PIL import Image


load_dotenv()
API = os.getenv('MOVIE_API')


st.set_page_config(
   page_title="Film Tavsiye Sistemi",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

image = Image.open('5Week_Recommendation_System/titans_movie_recommend/pic/netflix-cover.jpg')

st.image(image, caption='Recommendation Movie System')



def get_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API}&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    dataframe = dataframe.reset_index()
    indices = pd.Series(dataframe.index, index=dataframe['original_title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:6].index

    movie_names = []
    movie_posters = []

    for i in movie_indices[0:6]:
        # fetch the movie poster
        movie_id = dataframe.iloc[i].id
        movie_posters.append(get_movie_poster(movie_id))
        movie_names.append(movies.iloc[i]['original_title'])

    return movie_names, movie_posters

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)





def improved_recommendations(title,cosine_sim, dataframe):
    dataframe = dataframe.reset_index()
    indices = pd.Series(dataframe.index, index=dataframe['original_title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:10].index

    movies = dataframe.iloc[movie_indices][['original_title', 'vote_count', 'vote_average']]
    vote_counts = dataframe[dataframe['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = dataframe[dataframe['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.50)
    qualified = movies[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    index_qual = qualified.index

    movie_names = []
    movie_posters = []

    for i in index_qual[0:5]:
        # fetch the movie poster
        movie_id = dataframe.iloc[i].id
        movie_posters.append(get_movie_poster(movie_id))
        movie_names.append(dataframe.iloc[i]['original_title'])

    return movie_names, movie_posters

st.title('Movie Recommendation Mini Project')

with open('model/movie_list2.pkl', 'rb') as file:
    movies = pickle.load(file)


vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.50)

cosine_sim = pickle.load(open('model/cosine_sim.pkl', 'rb'))
cosine_sim2 = pickle.load(open('model/cosine_sim2.pkl', 'rb'))
cosine_sim3 = pickle.load(open('model/cosine_sim3.pkl', 'rb'))

movie_list = movies['original_title'].values  # Tüm film isimlerini alma
indices = pd.Series(movies.index, index=movies['original_title'])
indices = indices[~indices.index.duplicated(keep='last')]
indices_copy = indices.reset_index()
index_list = indices_copy["original_title"].values


selected_movie = st.selectbox("Select the movie from the menu", index_list) # Açılır menüde film isimlerini gösterme.

movie_names, movie_posters = content_based_recommender(selected_movie, cosine_sim, movies)


if st.button('See Recommendation'):

    st.subheader("Movie Recommendation based on Description Content")
    st.markdown("We developed a content-based recommendation system for the description of the movie (overview, keywords, genre, director). "
            "We show you the 5 movies closest to the selected movie.")
    movie_names, movie_posters = content_based_recommender(selected_movie, cosine_sim, movies)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(movie_posters[0])
        st.markdown(movie_names[0])

    with col2:
        st.image(movie_posters[1])
        st.markdown(movie_names[1])

    with col3:
        st.image(movie_posters[2])
        st.markdown(movie_names[2])

    with col4:
        st.image(movie_posters[3])
        st.markdown(movie_names[3])

    with col5:
        st.image(movie_posters[4])
        st.markdown(movie_names[4])

    st.subheader("Movie Recommendation based on Director and Cast Contents")
    st.markdown(
        "We developed a content-based recommendation system for the cast and director of the movie "
        "We show you the 5 movies closest to the selected movie.")
    movie_names2, movie_posters2 = content_based_recommender(selected_movie, cosine_sim2, movies)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(movie_posters2[0])
        st.markdown(movie_names2[0])

    with col2:
        st.image(movie_posters2[1])
        st.markdown(movie_names2[1])

    with col3:
        st.image(movie_posters2[2])
        st.markdown(movie_names2[2])

    with col4:
        st.image(movie_posters2[3])
        st.markdown(movie_names2[3])

    with col5:
        st.image(movie_posters2[4])
        st.markdown(movie_names2[4])


    st.subheader("Movie Recommendation based on Genres/Keyword Contents")
    st.markdown(
        "We developed a content-based recommendation system for the genres of the movie "
        "We show you the 5 movies closest to the selected movie.")

    movie_names3, movie_posters3 = content_based_recommender(selected_movie, cosine_sim3, movies)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(movie_posters3[0])
        st.markdown(movie_names3[0])

    with col2:
        st.image(movie_posters3[1])
        st.markdown(movie_names3[1])


    with col3:
        st.image(movie_posters3[2])
        st.markdown(movie_names3[2])

    with col4:
        st.image(movie_posters3[3])
        st.markdown(movie_names3[3])

    with col5:
        st.image(movie_posters3[4])
        st.markdown(movie_names3[4])


    st.subheader("Movie Recommendation based on Vote Average/Counts")
    st.markdown(
        "We developed a content-based recommendation system based on the number and rate of votes. "
        "We show you the 5 movies closest to the selected movie.")

    movie_names4, movie_posters4 = improved_recommendations(selected_movie, cosine_sim, movies)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(movie_posters4[0])
        st.markdown(movie_names4[0])

    with col2:
        st.image(movie_posters4[1])
        st.markdown(movie_names4[1])

    with col3:
        st.image(movie_posters4[2])
        st.markdown(movie_names4[2])

    with col4:
        st.image(movie_posters4[3])
        st.markdown(movie_names4[3])

    with col5:
        st.image(movie_posters4[4])
        st.markdown(movie_names4[4])





