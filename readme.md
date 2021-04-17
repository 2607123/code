Context and goal of the project
Considering the prevalent discussion of cryptocurrency and bitcoin in public and social media in a verbal or paper way, this project is trying to figure out the relationship between the sentiment of major media and bitcoin price based on the data obtained from previous month. The goal of this project is so analyse the most postive and negative words, a N-grams analyse and the correlation between the news and bitcoin price. The code is write in a file and separate with tab to seperate the different section. 

Requirement
To run this code it is recommanded to have python 3.6 or above and was run on spyder and pycharm.

installment
To run the code.py file the following package need to be installed:
- newsapi 1.1.0,
- pandas  2.0.0,
- nltk.sentiment.vader 3.2.1.1,
- nltk 3.6.1,
- pyplot-themes 0.2.2,
- yfinance 0.1,59,
- wordcloud 3.5.2

Data sources and quantity
NewsAPI for analysis due to its convenience to use in terms of the accessibility to more than 30,000 news from all over the world. Besides, the API allows users to search published articles and journals with various combinations among keywords and phrases such as publishing dates, languages and sources. In order to study the correlation between attitudes from major media and float of bitcoin prices, we selected “Bitcoin” as the keyword and extracted data from 8 popular newspapers and magazines, including New York Times, BBC News, Reuters, TechCrunch, Entrepreneur, Wired, Business Insider, Harvard Business Revie and The Verge. The data extracted was the source, date and text. 

Bitcoin price
The bitcoin price was extracted via yahoo finance to get OCLHV (Open price, Close price, Low price, High price and Volume), ranging from 4 March 2021 to 4 April 2021. 

Sentiment analysis - vaderSentiment
For the sentimental analysis the Vader algorithm was applied. There is a great implementation document related in Python called vaderSentiment. https://github.com/cjhutto/vaderSentiment.

About the Scoring

The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.

It is also useful for researchers who would like to set standardized thresholds for classifying sentences as either positive, neutral, or negative. Typical threshold values (used in the literature cited on this page) are:

positive sentiment: compound score >= 0.05
neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
negative sentiment: compound score <= -0.05
NOTE: The compound score is the one most commonly used for sentiment analysis by most researchers, including the authors.
The pos, neu, and neg scores are ratios for proportions of text that fall in each category (so these should all add up to be 1... or close to it with float operation). These are the most useful metrics if you want to analyze the context & presentation of how sentiment is conveyed or embedded in rhetoric for a given sentence. For example, different writing styles may embed strongly positive or negative sentiment within varying proportions of neutral text -- i.e., some writing styles may reflect a penchant for strongly flavored rhetoric, whereas other styles may use a great deal of neutral text while still conveying a similar overall (compound) sentiment. As another example: researchers analyzing information presentation in journalistic or editorical news might desire to establish whether the proportions of text (associated with a topic or named entity, for example) are balanced with similar amounts of positively and negatively framed text versus being "biased" towards one polarity or the other for the topic/entity.

IMPORTANTLY: these proportions represent the "raw categorization" of each lexical item (e.g., words, emoticons/emojis, or initialisms) into positve, negative, or neutral classes; they do not account for the VADER rule-based enhancements such as word-order sensitivity for sentiment-laden multi-word phrases, degree modifiers, word-shape amplifiers, punctuation amplifiers, negation polarity switches, or contrastive conjunction sensitivity.

CONFIGURATION
-------------

The module has no menu or modifiable settings. There is no configuration. When
enabled, the module will prevent the links from appearing. To get the links
back, disable the module and clear caches.

TROUBLESHOOTING
---------------

 * If wordcloud doesn't pip install:

   - use pycharm to install this package and run the code, as wordcloud doens't support 3.6.4 python or above degree the python version in pycharm and install a later version of wordcloud package. 

 * If newsapi line give errors:

- try pip install news-api, otherwise check the site package folder to see if there are multiple folder with the same name. If there is delete those and re-install the package.


