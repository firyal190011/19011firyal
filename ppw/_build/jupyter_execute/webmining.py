#!/usr/bin/env python
# coding: utf-8

# # LSA

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# 

# In[9]:


df=pd.read_csv('jurnal.csv', usecols =['Abstrak_indo'])


# In[11]:


df.head(10)


# In[14]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[15]:


# time taking
df['Abstrak_indo_cleaned']=df['Abstrak_indo'].apply(clean_text)


# In[16]:


df.head()


# In[17]:


df.drop(['Abstrak_indo'],axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


df['Abstrak_indo_cleaned'][0]


# In[20]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) 
# to play with. min_df,max_df,max_features etc...


# In[21]:


vect_text=vect.fit_transform(df['Abstrak_indo_cleaned'])


# In[22]:


print(vect_text.shape)
print(vect_text)


# In[23]:


idf=vect.idf_


# In[25]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['yang'])
print(dd['wajah'])  # police is most common and forecast is least common among the news headlines.


# In[26]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[27]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[28]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[29]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[30]:


lda_top=lda_model.fit_transform(vect_text)


# In[31]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[32]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)  


# In[33]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# In[34]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# In[35]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[37]:


pip install wordcloud


# In[38]:


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
 


# In[39]:


# topic 0
draw_word_cloud(0)


# In[40]:


# topic 1
draw_word_cloud(1)


# In[ ]:




