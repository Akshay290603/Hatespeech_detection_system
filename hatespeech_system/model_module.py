import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def clean_text(text):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

def predict_and_plot(input_text):
    df = pd.read_csv("twitter_data.csv")
    df["cleaned_tweet"] = df["tweet"].apply(clean_text)

    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_tweet"])
    y = df["class"]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    cleaned_input = clean_text(input_text)
    input_vectorized = vectorizer.transform([cleaned_input])
    prediction = clf.predict(input_vectorized)

    labels = ['Hate Speech', 'Offensive Language', 'Neither']
    percentages = np.zeros(3)
    percentages[prediction[0] - 1] = 100  

    plt.bar(labels, percentages, color=['red', 'blue', 'green'])
    plt.xlabel('Speech Type')
    plt.ylabel('Percentage')
    plt.title('Percentage of Different Speech Types')
    graph_url = ('/hatespeech_system/static/images/graph.png')
    plt.savefig('.' + graph_url)
    plt.close()

    if prediction[0] == 1:
        result = "Hate Speech Detected"
    elif prediction[0] == 2:
        result = "Offensive Language Detected"
    else:
        result = "Neither Hate Speech nor Offensive Language"

    return result, graph_url
