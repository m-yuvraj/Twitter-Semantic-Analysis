#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import string
import numpy as np
import spacy
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# In[2]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:


old_data=pd.read_csv("tweet.csv")
new_data=pd.read_csv("twt_classs_nlp.csv")


# In[4]:


old_data.head()


# In[5]:


old_data=old_data.drop(old_data[old_data.tweets.duplicated()].index)


# In[6]:


old_data.tail(10)


# In[7]:


old_data.head()


# In[8]:


def separte_hash(txt):
  temp=txt.split(" ")
  remove_word=['sarcasm','irony','ironic','sarcastic']
  temp=[word for word in temp if not word in remove_word]
  temp=' '.join(temp)
  return temp


# In[9]:


new_data.isnull().sum()


# In[10]:


new_data=new_data.dropna()
new_data=new_data.reset_index()
new_data.head()


# In[11]:


new_data['after_removal']=""


# In[12]:


for i in range(0,len(new_data['clean_tweet'])):
  if(new_data['class'][i]=='figurative'):
    new_data['after_removal'][i]=separte_hash(new_data['clean_tweet'][i])
  else:
    new_data['after_removal'][i]=new_data['clean_tweet'][i]
new_data['after_removal']


# In[13]:


#for i in range(0,len(new_data['clean_tweet'])):
#  if(new_data['class'][i]=='figurative'):
#    text=new_data['after_removal'][i].split(" ")
#    new_data['after_removal'][i]=separte_hash(new_data['clean_tweet'][i])
#  else:
#    new_data['after_removal'][i]=new_data['clean_tweet'][i]
#new_data['after_removal']


# In[14]:


new_data.isnull().sum()


# In[15]:


data_exp=pd.DataFrame()
data_exp['old_tweet']=new_data['clean_tweet']
data_exp['new_tweet']=new_data['after_removal']
data_exp['twt_class']=new_data['class']
data_exp.head()


# In[16]:


nlp=spacy.load('en_core_web_sm')
my_stop_word=stopwords.words('english')
for i in range(0,len(data_exp['new_tweet'])):
  text_tokens=(data_exp['new_tweet'][i]).split(" ")
  doc=nlp(' '.join(text_tokens))
  txt=[token.lemma_ for token in doc]
  data_exp['new_tweet'][i]=' '.join(txt)

#data1.head()


# In[17]:


my_stop_word=nltk.corpus.stopwords.words('english')
my_stop_word.append('m')
my_stop_word.append('s')
for i in range(0,len(data_exp['new_tweet'])):
  text_tokens=(data_exp['new_tweet'][i]).split(" ")
  txt=[word for word in text_tokens if not word in my_stop_word]
  txt=' '.join(txt)
  data_exp['new_tweet'][i]=txt
data_exp.head()


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


tweet_all=data_exp['new_tweet']
tweet_all=data_exp[data_exp['twt_class']=='figurative']['new_tweet']
cnt_vect_fig=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 650)
X=cnt_vect_fig.fit_transform(tweet_all)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cnt_vect_fig.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x:x[1] ,reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
#wd_df.to_csv("figurative_ngram_cntvect.csv")
wd_df.head(10)


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
#figurative
tweet_all=data_exp['new_tweet']
tfidf_fig=TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 1000)
X=tfidf_fig.fit_transform(tweet_all)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in tfidf_fig.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x:x[1] ,reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','weight']
#wd_df.to_csv("figurative_ngram_tfidf.csv")
wd_df.head(10)


# In[21]:


X.toarray()


# In[22]:


data_exp['lbl_class']=data_exp['twt_class'].map({'figurative':0,'sarcasm':1,'irony':2,'regular':3})
data_exp


# In[23]:


y=data_exp['lbl_class']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=43)


# In[25]:


model_xg = XGBClassifier(n_estimators=500,max_depth=3,gamma=2, max_leaves=10, learning_rate=0.01)
model_xg.fit(x_train, y_train)
y_pred_train=model_xg.predict(x_train)
y_pred_test=model_xg.predict(x_test)
print("train_score",metrics.accuracy_score(y_pred_train,y_train))
print("test_score",metrics.accuracy_score(y_pred_test,y_test))


# In[26]:


print(metrics.classification_report(y_pred_train,y_train))

