#!/usr/bin/env python
# coding: utf-8

# # Topic Modelling using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) in sklearn

# ## Import Library
# 
# ### Library yang digunakan
# 
# - **Pandas**
# 
#     Untuk manipulasi dan membaca data dalam bentuk tabel 
# 
# - **matplotlib**
# 
#     Untuk membuat visualisasi data
# 
# <!-- - **seaborn** -->
# - **PySastrawi**
# 
#     Untuk melakukan text processing
# 
# - **scikit-learn**
# 
#     Untuk menghitung TF dan TF-IDF

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
# import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
# sns.set(style='whitegrid',color_codes=True)

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
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
# stop_words=set(nltk.corpus.stopwords.words('english'))


# In[2]:


# Melakukan setting jumlah kolom maksimal pada output
pd.options.display.max_columns = 10


# ## Membaca Data

# In[3]:


df = pd.read_csv('dataset_pta.csv')


# In[4]:


df.head()


# Data yang digunakan dalam program ini hanya data pada kolom 'judul'. Untuk mengambil kolom 'judul' saja dapat dilakukan dengan inisialisasi ulang df dengan df[['judul']] 

# In[5]:


df = df[['judul']]
df.head()


# ## Pre-processing Data

# Terdapat beberapa tahapan dalam melakukan Pre-processing data, diantaranya *case folding* (Mengubah teks menjadi *lower case*), menghapus angka dan tanda baca, menghapus white space dan *stopword removal*. Semua tahapan *pre-processing* tersebut saya masukkan ke dalam fungsi clean_text, kemudian saya aplikasikan pada data judul pada dataframe dengan method **.apply(clean_text)**. 
# 
# Untuk menghapus stopword saya menggunakan library **PySastrawi**, karena **PySastrawi** memiliki list stopword bahasa indonesia yang lebih lengkap daripada library **nltk**.
# 
# Pada Library **PySastrawi** penghapusan stopword dilakukan dengan membuata objek StopWordRemoverFactory, kemudian buat objek stopword remover dengan method create_stop_word_remover. Objek stopword remover memiliki method remove yang dapat digunakan untuk menghapus stopword dalam sebuah kalimat dengan memasukkan string ke dalam parameter method remove.  

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


# ### Perbedaan data awal dengan data yang telah di-preprocessing

# In[9]:


df.head()


# ### Hapus kolom 'judul'

# In[10]:


df.drop(['judul'],axis=1,inplace=True)


# ### Mengganti nama kolom __cleaned_judul__ dengan __judul__ 

# In[11]:


df.columns = ['judul']


# In[12]:


df.head()


# ### Contoh judul yang telah di lakukan *pre-processing*

# In[13]:


df['judul'][0]


# ## Ekstraksi fitur dan membuat Document Term Matrix (DTM)
# 
# Dalam perhitungan LSA (Latent Semantic Analysis) data yang diperlukan hanya TF-IDF. Sehingga pada program ini tidak perlu mencari nilai TF dari dokumen. Untuk mengetahui nilai TF-IDF dapat dilakukan dengan membuat objek dari kelas TfidfVectorizer yang disediakan library scikit-learn.
# 
# Rumus Term Frequency:
# $$
# tf(t,d) = { f_{ t,d } \over \sum_{t' \in d } f_{t,d}}
# $$
# 
# $ f_{ t,d } \quad\quad\quad\quad$: Jumlah kata t muncul dalam dokumen
# 
# $ \sum_{t' \in d } f_{t,d} \quad\quad$: Jumlah seluruh kata yang ada dalam dokumen
# 
# Rumus Inverse Document Frequency:
# $$
# idf( t,D ) = log { N \over { | \{ d \in D:t \in d \} | } }
# $$
# 
# $ N \quad\quad\quad\quad\quad$ : Jumlah seluruh dokumen
# 
# $ | \{ d \in D:t \in d \} | $ : Jumlah dokumen yang mengandung kata $ t $
# 
# Rumus Inverse Document Frequency:
# $$
# tfidf( t,d,D ) = tf( t,d ) \times idf( t,D )
# $$

# In[14]:


vect = TfidfVectorizer()


# Setelah objek **TfidfVectorizer** dibuat gunakan method **fit_transform** dengan argumen data yang akan dicari nilai **TF-IDF**-nya

# In[15]:


vect_text = vect.fit_transform(df['judul'])


# In[16]:


attr_count = vect.get_feature_names_out().shape[0]
print(f'Jumlah atribut dalam Document-Term Matrix : {attr_count}')


# #### Menyimpan hasil tfidf ke dalam DataFrame

# Hasil tfidf perlu diubah terlebih dahulu menjadi array agar dapat digunakan sebagai data. Kemudian untuk parameter kolom-nya dapat didapatkan menggunakan method get_feature_names_out pada objek TfidfVectorizer.

# In[17]:


tfidf = pd.DataFrame(
    data=vect_text.toarray(),
    columns=vect.get_feature_names_out()
)
tfidf.head()


# Mencari nilai **idf** dengan mengakses atribut **idf_** pada objek **tfidfVectorizer**. Atribut **idf_** hanya terdefinisi apabila parameter **use_idf** saat instansiasi objekk tfidfVectorizer bernilai **True**. Namun, **use_idf** sudah bernilai **True** secara default, sehingga kita dapat perlu menentukannya secara manual. 

# In[18]:


idf = vect.idf_


# In[19]:


dd= dict(zip(vect.get_feature_names_out(), idf))

l = sorted(dd, key = dd.get)


# Kita dapat melihat kata yang paling sering dan paling jarang muncul pada judul tugas akhir berdasarkan nilai idf. Kata yang memiliki nilai lebih kecil, adalah kata yang paling sering muncul dalam judul

# In[20]:


print("5 Kata paling sering muncul:")
for i, word in enumerate(l[:5]):
    print(f"{i+1}. {word}\t(Nilai idf: {dd[word]})")


# In[21]:


print("5 Kata paling jarang muncul:")
for i, word in enumerate(l[:-5:-1]):
    print(f"{i+1}. {word}\t(Nilai idf: {dd[word]})")


# ## TOPIC MODELLING

# ### Latent Semantic Analysis (LSA)

# Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai judul tugas akhir dengan mengkonversikan judul tugas akhir menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term. Secara umum, langkah-langkah LSA dalam penilaian judul tugas akhir adalah sebagai berikut:
# 
# 1. Text Processing
# 2. Document-Term Matrix
# 3. Singular Value Decomposition (SVD)
# 4. Cosine Similarity Measurement

# #### Singular Value Decomposition

# Singular Value Decomposition (SVD) adalah sebuah teknik untuk mereduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan Document-Term Matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari Document-Term Matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu Matriks ortogonal U, Matriks diagonal S, Transpose dari matriks ortogonal V.

# $$
# A_{mn} = U_{mm} \times S_{mn} \times V^{T}_{nn}
# $$
# 
# $ A_{mn} $ : matriks awal
# 
# $ U_{mm} $ : matriks ortogonal
# 
# $ S_{mn} $ : matriks diagonal
# 
# $ V^{T}_{nn} $ : Transpose matriks ortogonal V

# Setiap baris dari matriks $ U $ (Document-Term Matrix) adalah bentuk vektor dari dokumen. Panjang dari vektor-vektor tersebut adalah jumlah topik. Sedangkan matriks $ V $ (Term-Topic Matrix) berisi kata-kata dari data.
# 
# SVD akan memberikan vektor untuk setiap dokumen dan kata dalam data. Kita dapat menggunakan vektor-vektor tersebut untuk mencari kata dan dokumen serupa menggunakan metode **Cosine Similarity**.
# 
# Dalam mengimplementasikan LSA, dapat menggunakan fungsi TruncatedSVD. parameter n_components digunakan untuk menentukan jumlah topik yang akan diekstrak.
# 
# 

# In[22]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[23]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print(f"Topic {i} : {topic*100}")


# In[24]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang kita dapat mendapatkan daftar kata yang penting untuk setiap topik. Jumlah kata yang akan ditampilkan hanya 10. Untuk melakukan sorting dapat menggunakan fungsi sorted, lalu slicing dengan menambahkan \[:10\] agar data yang diambil hanya 10 data pertama. Slicing dilakukan berdasarkan nilai pada indeks 1 karena nilai dari nilai lsa.

# In[25]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)

    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print(f"Topic {i}: ")
    print(" ".join([ item[0] for item in sorted_words ]))
    
    print("\n")
         


# ## Latent Dirichlet Allocation (LDA)  

# LDA is the most popular technique.**The topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.**
# 
# **To understand the maths it seems as if knowledge of Dirichlet distribution (distribution of distributions) is required which is quite intricate and left fior now.**
# 
# To get an inituitive explanation of LDA checkout these blogs: [this](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)  ,  [this](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)  ,[this](https://en.wikipedia.org/wiki/Topic_model)  ,  [this kernel on Kaggle](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)  ,  [this](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) .

# In[26]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[27]:


lda_top=lda_model.fit_transform(vect_text)


# In[28]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[29]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)  


# #### Note that the values in a particular row adds to 1. This is beacuse each value denotes the % of contribution of the corressponding topic in the document.

# $$
# w_{i,j} = tf_{i,j} * log( {{N} \over {df_{j}}} )
# $$

# In[30]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# #### As we can see Topic 7 & 8 are dominantly present in document 0.
# 
#  

# In[31]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# #### Most important words for a topic. (say 10 this time.)

# In[32]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:





# #### To better visualize words in a topic we can see the word cloud. For each topic top 50 words are plotted.

# In[33]:


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
 


# In[34]:


# topic 0
draw_word_cloud(0)


# In[35]:


# topic 1
draw_word_cloud(1)  # ...

