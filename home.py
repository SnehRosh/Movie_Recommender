import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import nltk
from nltk.corpus import stopwords
import textblob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


  


# Config
st.set_page_config(
    layout="centered",
    page_title="Movie Recommendation",
    page_icon="🎬",
    initial_sidebar_state='expanded',)

page_bg_img = '''
<style>
body {
background-image: bg.jpg;
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

def load_data(path):
    df = pd.read_csv('C:/Users/Hp/OneDrive/Desktop/PYDS/Movie_recommender/imdb_raw.csv')    
    return df


with st.spinner('Processing Movies Data'):
    df=load_data('C:/Users/Hp/OneDrive/Desktop/PYDS/Movie_recommender/imdb_raw.csv')
with st.container():
    st.title("Movie Remommendation website")

#Top rated movies
c1,c2=st.columns(2)
ft = c1.form('Find the top rated movies here')
btn = ft.form_submit_button(":red[Click here to get the top rated movies]") 
if btn: 
    nd= df[['title','release_year','rating']]
    top_df = nd.head(10)
    stop_df = top_df.sort_values(by='rating',ascending=False)
    top10_mv = stop_df.head(10)
    btn = st.dataframe(data=top10_mv,
                           width=1000,
                           height=400)
    

#Movie recommendation based on selected movie
st.subheader("Enter the name of Movie for recommendation")
c1,c2=st.columns(2)
movie_title = df[['title']]
inx = df.set_index('title')
movie_list=inx.index.tolist()
option = st.selectbox('Select the name of the movie',options = movie_list)
df['data'] = df['title'] + ' ' + df['director'] + ' ' +  df['genre']
    #remove punctuations - anything that is not a word or a space
df['data'] = df['data'].str.replace('[^\w\s]','')
    # Lower case
df['data'] = df['data'].str.lower()
def remove_stopwords(text):
    words = text.split()
    return" ".join(word for word in words if word not in stopwords.words('english'))
df['data'] = df['data'].apply(remove_stopwords)
    #vectorize the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['data']).toarray()
# similarity matrix
similarity = cosine_similarity(X, X)
def get_index_from_title(title):
    try: 
        return df[df.title == title].index[0]
    except:
        return None
def recommend_movie(title, Limit = 10):
    index = get_index_from_title(title)
    if index is None:
        return []
    else:
        movie_scores = []
        for i in range(similarity.shape[0]):
            movie_scores.append((df['title'][i],similarity[index][i]))
        movie_scores.sort(key=lambda x:x[1],reverse = True)
        return movie_scores[1:Limit+1]
if option:    
    rm= recommend_movie(option,5)
    drm=pd.DataFrame(rm)
    drm
