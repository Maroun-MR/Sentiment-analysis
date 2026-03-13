# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:29:23 2026

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Reviews.csv')

# First look
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Keep only Score and Text columns
df = df[['Score', 'Text']].dropna()

# Use only 50,000 rows to keep it fast
df = df.sample(50000, random_state=42)
print("Working dataset shape:", df.shape)

# Convert scores to sentiment labels
def get_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Score'].apply(get_sentiment)
print("\nSentiment distribution:")
print(df['Sentiment'].value_counts())

# Clean the text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['Clean_Text'] = df['Text'].apply(clean_text)
print("\nOriginal review:")
print(df['Text'].iloc[0])
print("\nCleaned review:")
print(df['Clean_Text'].iloc[0])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data
X = df['Clean_Text']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers using TF-IDF
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
print("\nDetailed Results:")
print(classification_report(y_test, y_pred))

# Test with your own review!
def predict_sentiment(review):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    result = model.predict(vectorized)[0]
    return result

print("\n--- Test Your Own Reviews ---")
print(predict_sentiment("This product is absolutely amazing!"))
print(predict_sentiment("Terrible quality, waste of money!"))
print(predict_sentiment("It was okay, nothing special"))

from wordcloud import WordCloud

# 1. Sentiment Distribution
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='Sentiment', 
              order=['Positive', 'Neutral', 'Negative'], 
              hue='Sentiment', palette='Blues', legend=False)
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()

# 2. Wordcloud for Positive reviews
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Clean_Text'])
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      colormap='Blues').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Positive Reviews')
plt.tight_layout()
plt.savefig('positive_wordcloud.png')
plt.show()

# 3. Wordcloud for Negative reviews
negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Clean_Text'])
wordcloud_neg = WordCloud(width=800, height=400, 
                          background_color='white', 
                          colormap='Reds').generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Negative Reviews')
plt.tight_layout()
plt.savefig('negative_wordcloud.png')
plt.show()



