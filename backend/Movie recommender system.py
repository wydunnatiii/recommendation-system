#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(2)


# In[4]:


movies.shape


# In[5]:


credits.head()


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[8]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


import ast 


# In[11]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[12]:


movies.dropna(inplace=True)


# In[13]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[14]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[15]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[16]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[17]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[18]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[19]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[20]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[21]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[22]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[24]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# movies.head()

# In[25]:


movies.head()


# In[26]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[34]:


movies.head()


# In[37]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[38]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[39]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[40]:


movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[41]:


movies.head()


# In[27]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[42]:


movies.head()


# In[43]:


new_df = movies[['movie_id','title','tags']]


# In[46]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[47]:


new_df.head()


# In[70]:


import nltk


# In[72]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer


# In[82]:


def stem(text):
    y = []
    
    for word in text.split():
       y.append(ps.stem(word))
    return" ".join(y)


# In[48]:


new_df['tags'][0]


# In[49]:


new_df['tags']= new_df['tags'].apply(lambda x:x.lower())


# In[52]:


new_df['tags'][0]


# In[53]:


new_df['tags'][1]


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[60]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# 

# In[61]:


vectors


# In[62]:


vectors[0]


# In[85]:


from nltk.stem import PorterStemmer

def stem(text):
    ps = PorterStemmer()
    y = []
    for word in text.split():
        y.append(ps.stem(word))
    return " ".join(y)

stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[87]:


new_df['tags']=new_df['tags'].apply(stem)


# In[90]:


from sklearn.metrics.pairwise import cosine_similarity


# In[92]:


similarity = cosine_similarity(vectors)


# In[112]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[107]:


sorted(similarity[0], reverse=True)


# In[97]:


similarity[0]


# In[119]:


def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
    


# In[121]:


recommend('Batman')


# In[118]:


new_df.iloc[1216].title


# In[101]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[122]:


import pickle


# In[124]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[125]:


new_df['title'].values


# In[127]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[129]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




