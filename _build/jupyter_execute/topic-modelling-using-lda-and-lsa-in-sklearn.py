#!/usr/bin/env python
# coding: utf-8

# ## Topic Modelling using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) in sklearn

# #### IMPORTING MODULES

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

import re
import string
#import nltk
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize,sent_tokenize

# #preprocessing
# from nltk.corpus import stopwords  #stopwords
# from nltk import word_tokenize,sent_tokenize # tokenizing
# from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# # for named entity recognition (NER)
# from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
# stop_words=set(nltk.corpus.stopwords.words('english'))


# #### LOADING THE DATASET

# In[2]:


df=pd.read_csv('./dataset_PTA.csv')


# In[3]:


df.head()


# We will drop the **'publish_date'** column as it is useless for our discussion.

# In[4]:


# drop the publish date.
# df.drop(['Abstraksi', 'Bidang Minat'],axis=1,inplace=True)
df = df[['judul']]


# In[5]:


df.head()


# #### DATA CLEANING & PRE-PROCESSING

# Here I have done the data pre-processing. I have used the lemmatizer and can also use the stemmer. Also the stop words have been used along with the words wit lenght shorter than 3 characters to reduce some stray words.

# In[6]:


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


# In[7]:


def clean_text(text):
  # Mengubah teks menjadi lowercase
  cleaned_text = text.lower()
  # Menghapus angka
  cleaned_text = re.sub(r"\d+", "", cleaned_text)
  # Menghapus white space
  cleaned_text = cleaned_text.strip()
  # Menghapus tanda baca
  cleaned_text = cleaned_text.translate(str.maketrans("","",string.punctuation))
  # Hapus stopword
  cleaned_text = stopword.remove(cleaned_text)
  return cleaned_text
  


# In[8]:


# time taking
df['cleaned_judul'] = df['judul'].apply(clean_text)


# In[9]:


df.head()


# Can see the difference after removal of stopwords and some shorter words. aslo the words have been lemmatized as in **'calls'--->'call'.**

# Now drop the unpre-processed column.

# In[10]:


df.drop(['judul'],axis=1,inplace=True)


# In[11]:


df.head()


# We can also see any particular news headline.

# In[12]:


df['cleaned_judul'][0]


# #### EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )
# 
# In DTM the values are the TFidf values.
# 
# Also I have specified some parameters of the Tfidf vectorizer.
# 
# Some important points:-
# 
# **1) LSA is generally implemented with Tfidf values everywhere and not with the Count Vectorizer.**
# 
# **2) max_features depends on your computing power and also on eval. metric (coherence score is a metric for topic model). Try the value that gives best eval. metric and doesn't limits processing power.**
# 
# **3) Default values for min_df & max_df worked well.**
# 
# **4) Can try different values for ngram_range.**

# In[13]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[59]:


vect =TfidfVectorizer() # to play with. min_df,max_df,max_features etc...


# In[60]:


vect_text=vect.fit_transform(df['cleaned_judul'])


# In[61]:


vect.get_feature_names_out().shape


# #### We can now see the most frequent and rare words in the news headlines based on idf score. The lesser the value; more common is the word in the news headlines.

# In[62]:


print(vect_text.shape)
print(vect_text)


# In[63]:


idf=vect.idf_


# In[65]:


dd=dict(zip(vect.get_feature_names_out(), idf))
l=sorted(dd, key=(dd).get)
# # print(l)
# print(l[0],l[-1])
# print(dd['police'])
# print(dd['forecast'])  # police is most common and forecast is least common among the news headlines.


# In[67]:


print(dd)


# We can therefore see that on the basis of the **idf value** , **'police'** is the **most frequent** word while **'forecast'** is **least frequently** occuring among the news.

# ### TOPIC MODELLING

# In[17]:





# ## Latent Semantic Analysis (LSA)

# The first approach that I have used is the LSA. **LSA is basically singular value decomposition.**
# 
# $$
# A_{mn} = U_{mm} \times S_{mn} \times V^{T}_{nn}
# $$
# 
# $ A_{mn} = $ matriks awal
# 
# **SVD decomposes the original DTM into three matrices S=U.(sigma).(V.T). Here the matrix U denotes the document-topic matrix while (V) is the topic-term matrix.**
# 
# **Each row of the matrix U(document-term matrix) is the vector representation of the corresponding document. The length of these vectors is the number of desired topics. Vector representation for the terms in our data can be found in the matrix V (term-topic matrix).**
# 
# So, SVD gives us vectors for every document and term in our data. The length of each vector would be k. **We can then use these vectors to find similar words and similar documents using the cosine similarity method.**
# 
# We can use the truncatedSVD function to implement LSA. The n_components parameter is the number of topics we wish to extract.
# The model is then fit and transformed on the result given by vectorizer. 
# 
# **Lastly note that LSA and LSI (I for indexing) are the same and the later is just sometimes used in information retrieval contexts.**

# In[68]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[69]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[85]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  


# Similalry for other documents we can do this. However note that values dont add to 1 as in LSA it is not probabiltiy of a topic in a document.

# In[86]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# #### Now e can get a list of the important words for each of the 10 topics as shown. For simplicity here I have shown 10 words for each topic.

# In[88]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
         


# ## Latent Dirichlet Allocation (LDA)  

# LDA is the most popular technique.**The topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.**
# 
# **To understand the maths it seems as if knowledge of Dirichlet distribution (distribution of distributions) is required which is quite intricate and left fior now.**
# 
# To get an inituitive explanation of LDA checkout these blogs: [this](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)  ,  [this](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)  ,[this](https://en.wikipedia.org/wiki/Topic_model)  ,  [this kernel on Kaggle](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)  ,  [this](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) .

# In[89]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[90]:


lda_top=lda_model.fit_transform(vect_text)


# In[91]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[25]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)  


# #### Note that the values in a particular row adds to 1. This is beacuse each value denotes the % of contribution of the corressponding topic in the document.

# $$
# w_{i,j} = tf_{i,j} * log( {{N} \over {df_{j}}} )
# $$

# In[92]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# #### As we can see Topic 7 & 8 are dominantly present in document 0.
# 
#  

# In[93]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# #### Most important words for a topic. (say 10 this time.)

# In[95]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[29]:





# #### To better visualize words in a topic we can see the word cloud. For each topic top 50 words are plotted.

# In[96]:


from wordcloud import WordCloud
# Generate a word cloud image for given topic
def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda_model.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()
 


# In[97]:


# topic 0
draw_word_cloud(0)


# In[98]:


# topic 1
draw_word_cloud(1)  # ...


# In[32]:





# ## THE END !!!

# ## [Please star/upvote in case u liked it. ]

# In[32]:




