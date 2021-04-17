#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:31:37 2021

@author: hou
"""
#library 
import os
import pandas as pd
%matplotlib inline
import pprint
pp = pprint.PrettyPrinter(indent=4)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

#api_news_key
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=('Xxxxxxxxxxxxxxxxxxxxxxxxxxxx'))
print(newsapi)

#functions

def get_sentiment_scores(text, date, source, url):
    sentiment_scores = {}

    # Sentiment scoring with VADER
    text_sentiment = analyzer.polarity_scores(text)
    sentiment_scores["date"] = date
    sentiment_scores["text"] = text
    sentiment_scores["source"] = source
    sentiment_scores["url"] = url
    sentiment_scores["compound"] = text_sentiment["compound"]
    sentiment_scores["positive"] = text_sentiment["pos"]
    sentiment_scores["neutral"] = text_sentiment["neu"]
    sentiment_scores["negative"] = text_sentiment["neg"]
    if text_sentiment["compound"] >= 0.05:  # Positive
        sentiment_scores["normalized"] = 1
    elif text_sentiment["compound"] <= -0.05:  # Negative
        sentiment_scores["normalized"] = -1
    else:
        sentiment_scores["normalized"] = 0  # Neutral

    return sentiment_scores
#domains=
def get_sentiments_on_topic(topic):
    """ We loke documentation"""
    sentiments_data = []


    # Loop through all the news articles
    for article in newsapi.get_everything(q=topic, language="en",from_param='2021-03-05',domains='nytimes.com', page_size=100,)["articles"]:
        try:
            # Get sentiment scoring using the get_sentiment_score() function
            sentiments_data.append(
                get_sentiment_scores(
                    article["content"],
                    article["publishedAt"][:10],
                    article["source"]["name"],
                    article["url"],
                )
            )

        except AttributeError:
            pass

    return sentiments_data

def sentiment_to_df(sentiments):
    
    # Create a DataFrame with the news articles' data and their sentiment scoring results
    news_df = pd.DataFrame(sentiments)

    # Sort the DataFrame rows by date
    news_df = news_df.sort_values(by="date")

    # Define the date column as the DataFrame's index
    news_df.set_index("date", inplace=True)
    return news_df




#get_article datas

topics = ['Bitcoin']
#domains =['Reuters','TechCrunch','Entrepreneur','Wired','Business Insider','Harvard Business Review','The Verge']
btc_sentiment = get_sentiments_on_topic(topics[0])
btc_df_ft = sentiment_to_df(btc_sentiment)

#combine_all_data_and_sort_on_date
btc_df_all = pd.concat([btc_df_bbc,btc_df_businessinsider,btc_df_entrepreneur,btc_df_hbr,btc_df_Reuters,btc_df_TechCrunch,btc_df_theverge,btc_df_wired])
btc_df = btc_df_all.sort_values(by="date")

display(btc_df.head())
display("btc_df.describe()")
display(btc_df.describe())


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re

#tokenize
#get_negative_list_of_words
nltk.download('wordnet')
nltk.download('stopwords')

# Complete the tokenizer function
def tokenizer(text):
    """returns a list of words that is lemmatized, stopworded, tokenized, and free of any non-letter characters. """
    # Create a list of the words
    # Convert the words to lowercase
    # Remove the punctuation
    # Remove the stop words
    # Lemmatize Words into root words
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words('english'))
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)
    words = word_tokenize(re_clean)
    return [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in set(stopwords.words('english'))]

btc_df["tokens"] = btc_df["text"].apply(tokenizer)
btc_df.head()






#NGrams and Frequency Analysis
from collections import Counter
from nltk import ngrams

flat_btc_tokens = [item for sublist in btc_df.tokens.to_list() for item in sublist]
bigram_counts = Counter(ngrams(flat_btc_tokens, n=2))
bigram_counts.most_common(20)


# Generate the Ethereum N-grams where N=2
flat_eth_tokens = [item for sublist in eth_df.tokens.to_list() for item in sublist]
eth_bigram_counts = Counter(ngrams(flat_eth_tokens, n=2))
eth_bigram_counts.most_common(20)

# Use the token_count function to generate the top 10 words from each coin
def token_count(tokens, N=20):
    """Returns the top N tokens from the frequency count"""
    return Counter(tokens).most_common(N)

token_count(flat_btc_tokens)






#plot_scatter
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

btc_df_all['x1']= btc_df_all.index
btc_df_all.plot(kind='scatter', x='x1',y='compound', rot=70)
plt.title("Compound X Date")
plt.show()


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [20.0, 10.0]





# Generate the Bitcoin word cloud
wordcloud = WordCloud(colormap="RdYlBu").generate(" ".join(flat_btc_tokens))
plt.imshow(wordcloud)
plt.axis("off")
fontdict = {"fontsize": 50, "fontweight": "bold"}
plt.title("Bitcoin Word Cloud", fontdict=fontdict)
plt.show()

btc_df_all.to_csv('data_articles.csv')

#demonstrate
import re
import nltk
from collections import Counter
#from wordcloud import WordCloud # using python 3.7
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

NEGWORDS = ["not", "no", "none", "neither", "never", "nobody", "n't", 'nor']
# STOPWORDS = ["an", "a", "the"] + NEGWORDS
STOPWORDS = ["an", "a", "the", "or", "and", "thou", "must", "that", "this", "self", "unless", "behind", "for", "which",
             "whose", "can", "else", "some", "will", "so", "from", "to", "by", "within", "of", "upon", "th", "with",
             "it","ha","reuters","stafffile","Charsby","wa","charsmarch","charsby","charsthe"]

def _remove_stopwords(txt):1
    """Delete from txt all words contained in STOPWORDS."""
    words = txt.split()
    # words = txt.split(" ")
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))

with open('/Users/text.txt', 'r', encoding='utf-8') as shakespeare_read:
    # read(n) method will put n characters into a string
    shakespeare_string = shakespeare_read.read()

shakespeare_split = str.split(shakespeare_string, sep=',')
print(shakespeare_split)
len(shakespeare_split)

doc_out = []
for k in shakespeare_split:
    cleantextprep = str(k)
        # Regex cleaning
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  # apply regex
    cleantext = cleantextCAP.lower()  # lower case
    cleantext = _remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    doc_out.append(bound) 

print(doc_out)
print(shakespeare_split)

for line in doc_out:
    print(line)

ndct = ''
with open('/Users/bl_negative.csv', 'r', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        ndct = ndct + line

# create a list of negative words
ndct = ndct.split('\n')
# ndct = [entry for entry in ndct]
len(ndct)

pdct = ''
with open('/Users/bl_positive.csv', 'r', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        pdct = pdct + line

pdct = pdct.split('\n')
# pdct = [entry for entry in pdct]
len(pdct)

#count words being collected in the lexicon
def decompose_word(doc):
    txt = []
    for word in doc:
        txt.extend(word.split())
    return txt

def wordcount(words, dct):
    counting = Counter(words)
    count = []
    for key, value in counting.items():
        if key in dct:
            count.append([key, value])
    return count

tokens = decompose_word(doc_out)
# decompose a list of sentences into words from NLTK module
tokens_nltk = nltk.word_tokenize(str(doc_out))

#wordcloud
comment_words = ' '
for token in tokens:
    comment_words = comment_words + token + ' '

print(comment_words)

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(comment_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("wordcloud.png",format='png',dpi=200)
plt.show()


#number of words inarticle
nwords = len(tokens)

nwc = wordcount(tokens, ndct)
pwc = wordcount(tokens, pdct)

ntot, ptot = 0, 0
for i in range(len(nwc)):
    ntot += nwc[i][1]

for i in range(len(pwc)):
    ptot += pwc[i][1]
    
print('Positive words:')
for i in range(len(pwc)):
    print(str(pwc[i][0]) + ': ' + str(pwc[i][1]))
print('Total number of positive words: ' + str(ptot))
print('\n')
print('Percentage of positive words: ' + str(round(ptot / nwords, 4)))
print('\n')
print('Negative words:')
for i in range(len(nwc)):
    print(str(nwc[i][0]) + ': ' + str(nwc[i][1]))
print('Total number of negative words: ' + str(ntot))
print('\n')
print('Percentage of negative words: ' + str(round(ntot / nwords, 4)))

#correlation
from numpy import cov

import yfinance as yf

BTC_USD = yf.Ticker("BTC-USD")

hist1D = BTC_USD.history(period="max",interval="1d",start="2021-03-04",end="2021-04-04")

hist1D["Pct"] = hist1D["Close"].pct_change(1) * 100


import matplotlib.pyplot as plt

plt.plot(hist1D["Close"])
plt.title('btc price')
plt.xlabel('price')
plt.ylabel('date')
plt.show()

#correlation on close_price
covariance_a =cov(btc_df_businessinsider['compound'],hist1D["Close"])

#correlation on close_price
cor_all= btc_df_all['compound'].corr(hist1D["Close"])
cor_businessinsider = btc_df_businessinsider['compound'].corr(hist1D["Close"])
cor_bbc= btc_df_bbc['compound'].corr(hist1D["Close"])
cor_entrepreneur= btc_df_entrepreneur['compound'].corr(hist1D["Close"])
cor_reuters= btc_df_Reuters['compound'].corr(hist1D["Close"])
cor_techcrunch= btc_df_TechCrunch['compound'].corr(hist1D["Close"])
cor_theverge= btc_df_theverge['compound'].corr(hist1D["Close"])
cor_wired= btc_df_wired['compound'].corr(hist1D["Close"])
