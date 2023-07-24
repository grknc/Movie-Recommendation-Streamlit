import pandas as pd
import json
import pickle
import os
import warnings

import pandas as pd



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/code?datasetId=138&sortBy=voteCount


credits = pd.read_csv('5Week_Recommendation_System/titans_movie_recommend/data/tmdb_5000_credits.csv')
movies = pd.read_csv('5Week_Recommendation_System/titans_movie_recommend/data/tmdb_5000_movies.csv')


def data_preparation(dataframe, column_list):
    for column in column_list:
        dataframe[column] = dataframe[column].apply(json.loads)
        for index, i in dataframe.iterrows():
            column_list_part = [partition['name'] for partition in i[column]]
            dataframe.loc[index, column] = str(column_list_part)

movies.columns

data_preparation(movies, ['genres', 'keywords', 'spoken_languages'])
data_preparation(credits, ['cast'])

# Get Director
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']

credits['crew']=credits['crew'].apply(json.loads)
credits['crew'] = credits['crew'].apply(get_director)
credits.rename(columns={'crew':'director'},inplace=True)

movies = movies.merge(credits,left_on='id',right_on='movie_id',how='left')
movies.columns

df = movies[['id','original_title','overview','genres','cast','vote_average', "vote_count",'director','keywords']]


df['description'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['director']
df["cast_director"] = df['cast'] + df['director']
df["genres_keywords"] = df["genres"] + df["keywords"]


final_data = df[['id', 'original_title', 'description','cast_director','genres_keywords','vote_average','vote_count']]
final_data.dropna(inplace=True)


#################################
# TF-IDF'in Problemimiz için Elde Edilmesi
#################################

tfidf = TfidfVectorizer(stop_words='english')
final_data['description'] = final_data['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(final_data['description'])
tfidf_matrix.shape

#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape



#################################
# TF-IDF'in Problemimiz için Elde Edilmesi
#################################

tfidf = TfidfVectorizer(stop_words='english')
final_data["cast_director"] = final_data["cast_director"].fillna('')
tfidf_matrix2 = tfidf.fit_transform(final_data["cast_director"])


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################
cosine_sim2 = cosine_similarity(tfidf_matrix2, tfidf_matrix2)


# Kategorilere Göre TF-IDF/Cosine Similarity

#################################
# TF-IDF'in Problemimiz için Elde Edilmesi
#################################

tfidf = TfidfVectorizer(stop_words='english')
final_data["genres_keywords"] = final_data["genres_keywords"].fillna('')
tfidf_matrix3 = tfidf.fit_transform(final_data["genres_keywords"])


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################
cosine_sim3= cosine_similarity(tfidf_matrix3, tfidf_matrix3)

indices = pd.Series(movies.index, index=movies['original_title'])
indices = indices[~indices.index.duplicated(keep='last')]
indices_copy = indices.reset_index()
indices_copy["original_title"].values



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
    return dataframe['original_title'].iloc[movie_indices]


content_based_recommender("Toy Story",cosine_sim,final_data)

content_based_recommender("Toy Story",cosine_sim2,final_data)

content_based_recommender("Toy Story",cosine_sim3,final_data)


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


vote_counts = final_data[final_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = final_data[final_data['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.50)


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
        movie_names.append(dataframe.iloc[i]['original_title'])

    return movie_names

improved_recommendations("Avatar",cosine_sim,final_data)


os.makedirs('model', exist_ok=True)


import pickle
import base64

# Writing final_data with binary mode
with open('5Week_Recommendation_System/titans_movie_recommend/model/movie_list.pkl', 'wb') as file:
    pickle.dump(final_data, file)

# Writing cosine_sim with binary mode
with open('5Week_Recommendation_System/titans_movie_recommend/model/cosine_sim.pkl', 'wb') as file:
    pickle.dump(cosine_sim, file)


# Cast Director Simularity
with open('5Week_Recommendation_System/titans_movie_recommend/model/cosine_sim2.pkl', 'wb') as file:
    pickle.dump(cosine_sim2, file)


# Genres_Keywords Simularity
# Writing cosine_sim with binary mode(genres_keywords)
with open('5Week_Recommendation_System/titans_movie_recommend/model/cosine_sim3.pkl', 'wb') as file:
    pickle.dump(cosine_sim3, file)
