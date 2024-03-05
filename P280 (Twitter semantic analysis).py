#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspellchecker')
get_ipython().system('pip install unidecode')


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import spacy
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')


# In[ ]:


import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[2]:


tweet=pd.read_csv("/content/sample_data/tweet.csv")
tweet.head()


# In[ ]:


tweet.shape


# In[ ]:


tweet.info()


# In[ ]:


df=pd.DataFrame(tweet[tweet.duplicated])
df.to_csv("duplicate.csv")


# In[ ]:


len(df)


# In[3]:


tweet[tweet.duplicated]
#here are the same tweet but belong to different class


# In[ ]:


[tweet['tweets'].duplicated]


# In[ ]:


len(tweet[tweet.duplicated])


# In[4]:


#we have 49 row with duplicate value we can drop them
data=tweet.copy()
data=data.drop_duplicates(keep='first',ignore_index=True)
data.tail(10)


# In[5]:


data[data.duplicated]


# In[6]:


data.to_csv("tweets_one.csv")


# In[ ]:


data['class'].value_counts()


# In[ ]:


#plt.plot(data['class'].value_counts())
data['class'].value_counts().plot(kind='bar')


# In[ ]:


data['tweets'][data['class']=='figurative']


# In[ ]:


#lets construct word cloude to see which word is being used in particular class
all_tweet=''.join(data['tweets'][data['class']=='figurative'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("figurative tweet")
plt.show()


# In[ ]:


#most frequent word in figurative class
word_freq=pd.Series(''.join(data['tweets'][data['class']=='figurative']).split()).value_counts()[100:200]
# barchart
plt.figure(figsize=(10,15))
sns.barplot(x=word_freq.values,y=word_freq.index)
plt.show()


# In[ ]:


all_tweet=''.join(data['tweets'][data['class']=='irony'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("Irony tweet")
plt.show()


# In[ ]:


all_tweet=''.join(data['tweets'][data['class']=='sarcasm'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("Sarcasm tweet")
plt.show()


# In[ ]:


all_tweet=''.join(data['tweets'][data['class']=='regular'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("regular tweet")
plt.show()


# In[ ]:


#find the length
data['text_len']=[len(x.split(" ")) for x in data.tweets]
data.head()


# In[ ]:


data['text_len']


# In[ ]:


fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(15,5))
#fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))#
tweet_len=tweet[tweet['class']=='figurative']['tweets'].str.len()
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title("figurative")

tweet_len=tweet[tweet['class']=='irony']['tweets'].str.len()
ax2.hist(tweet_len,color='#17C37B')
ax2.set_title("irony")

tweet_len=tweet[tweet['class']=='sarcasm']['tweets'].str.len()
ax3.hist(tweet_len,color='#17C37B')
ax3.set_title("sarcasm")

tweet_len=tweet[tweet['class']=='regular']['tweets'].str.len()
ax4.hist(tweet_len,color='#17C37B')
ax4.set_title("regular")
plt.show()


# In[ ]:


fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(15,5))
#fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))#
tweet_len=data[data['class']=='figurative']['tweets'].str.len()
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title("figurative")

tweet_len=data[data['class']=='irony']['tweets'].str.len()
ax2.hist(tweet_len,color='#17C37B')
ax2.set_title("irony")

tweet_len=data[data['class']=='sarcasm']['tweets'].str.len()
ax3.hist(tweet_len,color='#17C37B')
ax3.set_title("sarcasm")

tweet_len=data[data['class']=='regular']['tweets'].str.len()
ax4.hist(tweet_len,color='#17C37B')
ax4.set_title("regular")
plt.show()


# In[ ]:


#old
tweet['class'].value_counts()


# In[ ]:


#new
data['class'].value_counts()


# In[ ]:


max(data.text_len)


# In[ ]:


print(data[data.text_len==67].tweets)


# In[ ]:


#removing leading or ending spaces
data['clean_data']=[x.strip() for x in data.tweets]
data.head()


# In[ ]:


#removal of @ word
def remove_pattern(text,pattern_regex):
  r=re.findall(pattern_regex,text)
  for i in r:
    text=re.sub(re.escape(i),'',text)
  return text


# In[ ]:


data['clean_data']=np.vectorize(remove_pattern)(data['clean_data'],"@[\w]*")
data.head()


# In[ ]:


#removal of https
data['clean_data']=np.vectorize(remove_pattern)(data['clean_data'],"http[\S]*")
data.head()


# In[ ]:


#data.to_csv('test_csv.csv')


# In[ ]:


data.isnull().sum()


# In[ ]:


#removal of "#"
data['clean_data']=np.vectorize(remove_pattern)(data['clean_data'],"#")
data.head()


# In[ ]:


#data.to_csv('test_csv.csv')


# In[ ]:


data.head(10)


# In[ ]:


for i in range(0,len(data['clean_data'])):
  text=data['clean_data'][i]
  no_pun=[char for char in text if char not in string.punctuation]
  no_pun=''.join(no_pun)
  data['clean_data'][i]=no_pun


# In[ ]:


data.head()


# #we need to do lemmatization to put all words into root words

# In[ ]:


from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
lemmatizer=WordNetLemmatizer()
spell = SpellChecker()


# In[ ]:


data['lem_tweet']=data['clean_data'].apply(lambda x:' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data['lem_tweet']=data['lem_tweet'].apply(lambda x:' '.join([word.lower() for word in x.split()]))
#lemm_df['lema']=clean_df['cleaned_rev.apply(lambda x : " ".join([word_lemmatizer.lemmatize(word) for word in x.split()]))


# In[ ]:


data.head()


# # #lets see after removing punctuation ,hash tag,@ and https

# In[ ]:


#lets construct word cloude to see which word is being used in particular class
all_tweet=''.join(data['lem_tweet'][data['class']=='figurative'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("figurative tweet")
plt.show()


# In[ ]:


#word_frequency
word_freq=pd.Series(''.join(data['lem_tweet'][data['class']=='figurative']).split()).value_counts()[0:50]
word_fq_df=pd.DataFrame({'word':word_freq.index,'frequency':word_freq.values})
word_fq_df


# In[ ]:


#lets construct word cloude to see which word is being used in particular class
all_tweet=''.join(data['lem_tweet'][data['class']=='irony'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("irony tweet")
plt.show()


# In[ ]:


all_tweet=''.join(data['lem_tweet'][data['class']=='sarcasm'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("sarcasm tweet")
plt.show()


# In[ ]:


all_tweet=''.join(data['clean_data'][data['class']=='regular'])
word_cloud=WordCloud(width=1500,height=1000,background_color='black').generate(all_tweet)
plt.imshow(word_cloud)
plt.title("regular tweet")
plt.axis('off')
plt.show()


# In[ ]:


all_tweet=''.join(data['lem_tweet'])
word_cloud=WordCloud(width=2000,height=1000,background_color='black').generate(all_tweet)
plt.figure(figsize=(12,6))
plt.imshow(word_cloud)
plt.title("all tweet")
plt.show()


# In[ ]:


data.to_csv('test.csv')


# In[ ]:


data.lem_tweet[10000]


# In[ ]:


#lets count stopwords present in each class and print in bar chart
from nltk.corpus import stopwords
nltk.download('stopwords')
my_stop_word=stopwords.words('english')


# In[ ]:


def find_hash(text):
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)
data['hash']=data['tweets'].apply(lambda x:find_hash(x.lower()))


# In[ ]:


data.head()


# In[ ]:


#data['hash'].value_counts()
#sns.barplot(x="Hashtag",y="count", data = data['hash'])
temp=data['hash'].value_counts()[:][1:11]
plt.figure(figsize=(18,5))
temp= temp.to_frame().reset_index().rename(columns={'index':'Hashtag','hash':'count'})
sns.barplot(x="Hashtag",y="count", data = temp)
plt.show()


# In[ ]:




