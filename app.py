import streamlit as st
import pickle
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pickled model and TfidfVectorizer
model = pickle.load(open('xgb_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Define a preprocess function to clean the input text
def preprocess(text):
    # Remove HTTP and HTTPS links
    text = re.sub(r'http\S+|https\S+', '', text)
    
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove the hashtag symbol (#)
    text = text.replace('#', '')
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

def predict_sentiment(tweets):
    # Preprocess the input tweet (e.g., tokenization, lowercasing, etc.)
    processed_tweet = preprocess(tweets)
    
    # Vectorize the tweet using TfidfVectorizer
    tweet_vectorized = tfidf_vectorizer.transform([processed_tweet])
    
    # Make the sentiment prediction using your model
    prediction = model.predict(tweet_vectorized)
    
    # Return the sentiment class
    return prediction[0]

def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a tweet to classify its sentiment:")
    
    user_input = st.text_input("Input your tweet:")
    
    if st.button("Analyze"):
        sentiment = predict_sentiment(user_input)
        if (sentiment==0):
            pred="figurative"
        elif(sentiment==1):
            pred="sarcasm"
        elif (sentiment==2):
            pred="irony"
        else:
            pred="regular"
        st.write(f"Sentiment Class: {pred}")

if __name__ == "__main__":
    main()
