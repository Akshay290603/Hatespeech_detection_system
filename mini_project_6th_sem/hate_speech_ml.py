import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stopwords = set(stopwords.words("english"))

df = pd.read_csv(r"C:\Users\Akshay\Downloads\twitter_data.csv")
# print(df.head())

df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language detected", 3: "No hate and offensive speech"})

df.dropna(subset=['class'], inplace=True)

df = df[['tweet', 'labels']]

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)
# print(df.head())

# Calculate hate speech percentage
hate_speech_count = df['labels'].value_counts()['Hate Speech Detected']
total_samples = df.shape[0]
hate_speech_percentage = (hate_speech_count / total_samples) * 100

# Plot the graph
labels = ['Non-Hate Speech', 'Hate Speech']
percentages = [hate_speech_percentage, 100 - hate_speech_percentage]

plt.bar(labels, percentages, color=['blue', 'red'])
plt.xlabel('Speech Type')
plt.ylabel('Percentage')
plt.title('Percentage of Hate Speech')
plt.show()

x = np.array(df["tweet"])
y = np.array(df["labels"])
# Check for NaN values in X_train and y_train
cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

nan_indices = np.isnan(X_train.toarray()).any(axis=1) | pd.isnull(y_train)
X_train = X_train[~nan_indices]
y_train = y_train[~nan_indices]

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

test_data = "I will kill you"
df = cv.transform([test_data]).toarray()
print(test_data,":is ")
print(clf.predict(df))