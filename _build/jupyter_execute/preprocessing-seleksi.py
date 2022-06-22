#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


# Install library Sastrawi
get_ipython().system('pip install PySastrawi')


# In[2]:


import pandas as pd  
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np


# # Membaca data hasil crawling

# In[3]:


data = pd.read_csv('dataset_PTA.csv')
data.head()


# In[4]:


data.shape


# In[5]:


# Inisialisasi label (Bidang Minat) ke dalam array
y = [ y[0] for y in data.iloc[:, [2]].values ]


# In[6]:


print(y)


# In[7]:


# Inisialisasi abstraksi ke dalam array
x = [ w[0] for w in data.iloc[:, [1]].values ]


# In[8]:


x[0]


# # Preprocessing Data Teks

# ## Case Folding dan menghapus karakter yang tidak diperlukan

# In[9]:


for i in range(len(x)):
    # Mengubah teks menjadi lowercase
    x[i] = x[i].lower()
    
    # Menghapus angka
    x[i] = re.sub(r"\d+", "", x[i])
    
    # Menghapus white space
    x[i] = x[i].strip()
    
    # Menghapus tanda baca
    x[i] = x[i].translate(str.maketrans("","",string.punctuation))


# In[10]:


x[0]


# ## Stemming dan stopwords removal menggunakan Sastrawi

# In[11]:


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopwordremover
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

for i in range(len(x)):
    # Stemming
    x[i] = stemmer.stem(x[i])
    
    # Hapus Stopword
    x[i] = stopword.remove(x[i])


# In[12]:


x[0]


# In[13]:


vect = CountVectorizer()

data_tf = vect.fit_transform(x)
data_tf = data_tf.toarray()

x_dtm = pd.DataFrame(data_tf, columns = vect.get_feature_names_out())
x_dtm.head()


# # Seleksi Fitur

# ## Menghitung nilai chi-square manual

# In[14]:


# binarisasi kolom output
y_binarized = LabelBinarizer().fit_transform(y)
print(y_binarized.T)

# baris menunjukkan jumlah pengamatan setiap kelas
# dan kolom menunjukkan setiap fitur
observed = np.dot(y_binarized.T, x_dtm)
print(observed)


# In[15]:


# menghitung probabilitas setiap kelas dan jumlah fitur; 
# simpan keduanya sebagai array dua dimensi dengan menggunakan reshape
class_prob = y_binarized.mean(axis = 0).reshape(1, -1)
feature_count = x_dtm.to_numpy().sum(axis = 0).reshape(1, -1)
expected = np.dot(class_prob.T, feature_count)

print(expected)


# In[16]:


chisq = (observed - expected) ** 2 / expected
chisq_score = chisq.sum(axis = 0)
print(chisq_score)


# ## Menghitung nilai chi-square menggunakan fungsi dari sklearn

# In[17]:


# Menghitung nilai chi-square menggunakan fungsi chi2
chi2score = chi2(x_dtm, y)

skorchi= np.reshape(chi2score[1], (1, chi2score[0].shape[0]))


# In[18]:


# Menampilkan nilai chi-square dalam bentuk DataFrame
data_skorchi = pd.DataFrame(skorchi, columns = vect.get_feature_names_out())
data_skorchi


# In[19]:


# Mencari 50 kata yang paling relevan dalam menentukan bidang minat
kbest = SelectKBest(score_func = chi2, k = 10)

X_dtm_kbest = kbest.fit_transform(x_dtm, y)
mask = kbest.get_support()

X_dtm_kbest = x_dtm.columns[mask]
X_dtm_kbest


# In[20]:


# Menampilkan 10 term paling relevan dan menambahkan kolom "Bidang Minat"
result = pd.concat([x_dtm[X_dtm_kbest], data["Bidang Minat"]], axis=1)
result.head()


# In[21]:


result.to_csv("Hasil_Seleksi_Fitur.csv", index=False)

