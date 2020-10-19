---
layout: post
title: Twitter Sentiment Analysis
date: 2020-05-24 00:00:00 +0300
description: In this blog, I will discuss linguistic features for detecting the sentiment of Twitter messages. (optional)
img: sentiment.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ZUBI_ASH, PROJECT, NLP, Twitter, linguistic, sentiment] # add tag
---

##  Project Description
In this blog, I will discuss linguistic features for detecting the sentiment of Twitter messages. I take a supervised approach to the problem, but I removed hashtags in the Twitter data for building training data.


{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_2.jpg)

## Introduction

In the past few years, there has been a huge growth in the use of `microblogging` platforms such as `Twitter`. Spurred by that growth, companies and media organizations are increasingly seeking ways to mine `Twitter` for information about what people think and feel about their products and services. Companies such as `Twitratr (twitrratr.com)`, `tweetfeel (www.tweetfeel.com)`, and `Social Mention (www.socialmention.com)` are just a few who advertise `Twitter` sentiment analysis as one of their services. While there has been a fair amount of research on how `sentiments` are expressed in genres such as online reviews and news articles, how sentiments are expressed given the informal language and message-length constraints of microblogging has been much less studied.

In this project, I will train a Naive Bayes classifier to predict sentiment from thousands of Twitter tweets. The process could be done automatically without having humans manually review thousands of `twitter` and customer `reviews`.

### 1. Problem Statement 

The objective of this task is to detect `sentiments` of the speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a `racist` or `sexist` sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label `1` denotes the tweet is racist/sexist and label `0` denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

`data source: https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech`

### 2. IMPORT LIBRARIES AND LOAD DATASETS

{% highlight ruby %}
#=> Importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
{% endhighlight %}

{% highlight ruby %}
tweets_df = pd.read_csv('twitter.csv')
#=> Printing out the head of the dataset
print(tweets_df.head())
{% endhighlight %}

| index | id | label | tweet | 
|---------------:|----------------:|-----------:|--:|
| 0 | 1 | 0 | thank you @user for you follow |
| 1 | 2 | 0 | @user #sikh #temple vandalised in in #calgary,... |
| 2 | 3 | 0 | listening to sad songs on a monday morning otw... |
| 3 | 4 | 0  | to see nina turner on the airwaves trying to... |
| 4 | 5 | 0 | factsguide: society now #motivation |


Getting a concise `summary` of the dataframe tweets_df.

{% highlight ruby %}
tweets_df.info()
{% endhighlight %}

OUTPUT.

{% highlight ruby %}
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 31962 entries, 0 to 31961
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      31962 non-null  int64 
 1   label   31962 non-null  int64 
 2   tweet   31962 non-null  object
dtypes: int64(2), object(1)
memory usage: 749.2+ KB
{% endhighlight %}

Getting `statistical summary` of data frame tweets_df.

{% highlight ruby %}
tweets_df.describe()
{% endhighlight %}

OUTPUT.

|       | id | label |
|---------------:|----------------:|-----------:|
| count | 31962.000000 | 31962.000000 |
| mean | 15981.500000 | 0.070146 |
| std | 9226.778988 | 0.255397 |
| min | 1.000000 | 0.000000 |
| 25% | 7991.250000 | 0.000000 |
| 50% | 15981.500000 | 0.000000 |
| 75% | 23971.750000 | 0.000000 |
| max | 31962.000000 | 1.000000 |

peeking in dataframe tweets_df.

{% highlight ruby %}
tweets_df['tweet']
{% endhighlight %}


RESULT:
{% highlight ruby %}
0         @user when a father is dysfunctional and is s...
1        @user @user thanks for #lyft credit i can't us...
2                                      bihday your majesty
3        #model   i love u take with u all the time in ...
4                   factsguide: society now    #motivation
                               ...                        
31957    ate @user isz that youuu?ðððððð...
31958      to see nina turner on the airwaves trying to...
31959    listening to sad songs on a monday morning otw...
31960    @user #sikh #temple vandalised in in #calgary,...
31961                     thank you @user for you follow  
Name: tweet, Length: 31962, dtype: object
{% endhighlight %}

We can see that there is a column named `id` which are just ids no use of it so we can remove it.

{% highlight ruby %}
tweets_df = tweets_df.drop(['id'], axis = 1)
{% endhighlight %}


### 3. EXPLORE DATASET

We want to sure there are no `null` elements in our data. for this we going to check null value in `seaborn heatmap`

If any `null` element it will show in the chart

{% highlight ruby %}
sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
{% endhighlight %}

RESULT: 
{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_3.JPG)

Checking other than `1` or `0` values in label column.

{% highlight ruby %}
 sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
{% endhighlight %}


RESULT: 
{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_4.JPG)

We can see in the plot that majority of labels are `0`.

Let's get the `length` of the `messages`.

{% highlight ruby %}
tweets_df['length'] = tweets_df['tweet'].apply(len)
{% endhighlight %}


OUTPUT.

| index | label | label | tweet | 
|---------------:|----------------:|-----------:|--:|
| 0 | 0 | @user when a father is dysfunctional and is s... | 102 |
| 1 | 0 | @user @user thanks for #lyft credit i can't us...	 |122 |
| 2 | 0 | bihday your majesty | 21 |
| 3 | 0 | #model i love u take with u all the time in ...  | 86 |
| 4 | 0 | to see nina turner on the airwaves trying to...	 | 131 |


Let's check the destribution of `length` of `tweets`.

{% highlight ruby %}
tweets_df['length'].plot(bins=100, kind='hist') 
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_5.JPG)


Let's see the `shortest message`

{% highlight ruby %}
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0]
{% endhighlight %}

{% highlight ruby %}
'i love you '
{% endhighlight %}

`Source` and `target` are the two `nodes` that are linked by an `edge`. A network can have directed or undirected edges and in this network all the edges are undirected. The weight attribute of every edge tells us the number of `interactions that the characters` have had over the book, and the book column tells us the book number.

Once we have the data loaded as a pandas DataFrame, it's time to create a network. We will use networkx, a network analysis library, and create a graph object for the first book.

{% highlight ruby %}
#=> Importing networkx module
import networkx as nx 
#=> Creating an empty graph object
G_book1 = nx.Graph()
{% endhighlight %}

Let's see the `positive messages`

{% highlight ruby %}
positive = tweets_df[tweets_df['label']==0]
{% endhighlight %}


OUTPUT:

| index | label | tweet | length | 
|----:|------:|--------------------------------------------:|--:|
| 0 | 0 | @user when a father is dysfunctional and is s... | 102 |
| 1 | 0 | @user @user thanks for #lyft credit i can't us...	 |122 |
| 2 | 0 | bihday your majesty | 21 |
| 3 | 0 | #model i love u take with u all the time in ...  | 86 |
| 4 | 0 | to see nina turner on the airwaves trying to...	 | 131 |


Let's see the `negative messages`

{% highlight ruby %}
negative = tweets_df[tweets_df['label']==1]
{% endhighlight %}


OUTPUT:

| index | label | tweet | length | 
|---------------:|----------------:|--------------------------------------------:|--:|
| 0 | 1 | @user #cnn calls #michigan middle school 'buil... | 74 |
| 1 | 1 | no comment! in #australia #opkillingbay #se...		 |101 |
| 2 | 1 | @user @user lumpy says i am a . prove it lumpy. | 22 |
| 3 | 1 | it's unbelievable that in the 21st century we'...  | 47 |
| 4 | 1 | lady banned from kentucky mall. @user #jcpenn..	 | 104 |


### 3. Lets Plot The WORDCLOUD

A `WORDCLOUD` is an image made of words that together resemble a cloudy shape. The size of a word shows how important it is e.g. how often it appears in a text — its frequency.

People typically use word clouds to easily produce a summary of large documents (reports, speeches), to create art on a topic (gifts, displays) or to visualise data (tables, surveys).

For it wae have to collect all tweets into list

{% highlight ruby %}
sentences = tweets_df['tweet'].tolist()
{% endhighlight %}

{% highlight ruby %}
sentences
{% endhighlight %}

{% highlight ruby %}
[' @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run',
 "@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked",
 '  bihday your majesty',
 '#model   i love u take with u all the time in urð\x9f\x93±!!! ð\x9f\x98\x99ð\x9f\x98\x8eð\x9f\x91\x84ð\x9f\x91\x85ð\x9f\x92¦ð\x9f\x92¦ð\x9f\x92¦  ',
 ' factsguide: society now    #motivation',
 '[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo  ',
 ' @user camping tomorrow @user @user @user @user @user @user @user dannyâ\x80¦',
 "the next school year is the year for exams.ð\x9f\x98¯ can't think about that ð\x9f\x98\xad #school #exams   #hate #imagine #actorslife #revolutionschool #girl",
 'we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â\x80¦ ',
 " @user @user welcome here !  i'm   it's so #gr8 ! ",
 ' â\x86\x9d #ireland consumer price index (mom) climbed from previous 0.2% to 0.5% in may   #blog #silver #gold #forex',
 'we are so selfish. #orlando #standwithorlando #pulseshooting #orlandoshooting #biggerproblems #selfish #heabreaking   #values #love #',
 'i get to see my daddy today!!   #80days #gettingfed',
 "@user #cnn calls #michigan middle school 'build the wall' chant '' #tcot  ",
 'no comment!  in #australia   #opkillingbay #seashepherd #helpcovedolphins #thecove  #helpcovedolphins',
 'ouch...junior is angryð\x9f\x98\x90#got7 #junior #yugyoem   #omg ',
 'i am thankful for having a paner. #thankful #positive     ',
 'retweet if you agree! ',
 'its #friday! ð\x9f\x98\x80 smiles all around via ig user: @user #cookies make people   ',
 'as we all know, essential oils are not made of chemicals. ',
 '#euro2016 people blaming ha for conceded goal was it fat rooney who gave away free kick knowing bale can hit them from there.  ',
 'sad little dude..   #badday #coneofshame #cats #pissed #funny #laughs ',
 "product of the day: happy man #wine tool  who's   it's the #weekend? time to open up &amp; drink up!",
 '@user @user lumpy says i am a . prove it lumpy.',
 ' @user #tgif   #ff to my #gamedev #indiedev #indiegamedev #squad! @user @user @user @user @user',
 'beautiful sign by vendor 80 for $45.00!! #upsideofflorida #shopalyssas   #love ',
 ' @user all #smiles when #media is   !! ð\x9f\x98\x9cð\x9f\x98\x88 #pressconference in #antalya #turkey ! sunday #throwback  love! ð\x9f\x98\x8að\x9f\x98\x98â\x9d¤ï¸\x8f '
{% endhighlight %}

Joining all `sentences`.

{% highlight ruby %}
sentences_as_one_string = " ".join(sentences)
{% endhighlight %}

`WordCloud` installation

{% highlight ruby %}
!pip install WordCloud
{% endhighlight %}

Importing `WordCloud` library

{% highlight ruby %}
from wordcloud import WordCloud
{% endhighlight %}

`Visualizing` the `WordCloud`

{% highlight ruby %}
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_6.JPG)



### 4. PERFORM DATA CLEANING - REMOVE PUNCTUATION FROM TEXT & REMOVE STOPWORDS

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_8.jpeg)

Firstly, we have to remove punctuation then followed by removing stop words from the text.

Import string and check import string punctuation

{% highlight ruby %}
import string
string.punctuation
{% endhighlight %}

RESULT:
{% highlight ruby %}
`'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
{% endhighlight %}

We wanted to remove aforementioned result from out text.

Looking at actual text.

{% highlight ruby %}
Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
{% endhighlight %}
 
Let go and clear our test data

{% highlight ruby %}
test_punc_removed = [ char for char in Test if char not in string.punctuation ]
{% endhighlight %}
 
{% highlight ruby %}
#=> Join the characters again to form the string.
test_punc_removed
test_punc_removed_join = ''.join(test_punc_removed)
test_punc_removed_join
{% endhighlight %}

{% highlight ruby %}
'Good morning beautiful people  I am having fun learning Machine learning and AI'
{% endhighlight %}

Importing Natural Language tool kit

{% highlight ruby %}
#=> import nltk # Natural Language tool kit 
nltk.download('stopwords')
{% endhighlight %}

{% highlight ruby %}
#=># You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')
{% endhighlight %}


{% highlight ruby %}
['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]
{% endhighlight %}

Removing stoppage words

{% highlight ruby %}
test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
{% endhighlight %}

{% highlight ruby %}
print(test_punc_removed_join_clean)
{% endhighlight %}

RESULT:

{% highlight ruby %}
['Good',
 'morning',
 'beautiful',
 'people',
 'fun',
 'learning',
 'Machine',
 'learning',
 'AI']
{% endhighlight %}

### 4. NLP - TOKENIZATION

Features in machine learning is basically numerical attributes from which anyone can perform some mathematical operation such as matrix factorisation, dot product etc. But there are various scenario when dataset does not contain numerical attribute for example- sentimental analysis of `Twitter/Facebook user`, `Amazon customer review`, `IMDB/Netflix movie recommendation`. In all the above cases dataset contain numerical value, string value, character value, categorical value, connection (one user connected to another user). Conversion of these types of feature into numerical feature is called featurization.

Lets suppose we have folowing four strings
 
STRING1: This is the first paper.
STRING2: This paper is the second paper.
STRING3: And this is the thirt one.
STRING4: Is this the first paper?

we wanted to convert them in bunch of numbers. We do pick every single word from t he s entences and place them in a table as column (each word each column). And as rows we put training samples. At the end we add zeros and ones corresponding to locations of every single word if it exist every single instinct and frequency in the sentence.

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_9.JPG)

we will use CountVectorizer which convert a collection of text documents to a matrix of token counts.

{% highlight ruby %}
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sample_data)
{% endhighlight %}

{% highlight ruby %}
print(vectorizer.get_feature_names())
{% endhighlight %}

{% highlight ruby %}
['and', 'document', 'first', 'is', 'one', 'paper', 'second', 'the', 'third', 'this']
{% endhighlight %}

{% highlight ruby %}
print(X.toarray())
{% endhighlight %}

{% highlight ruby %}
[[0 0 1 1 0 1 0 1 0 1]
 [0 1 0 1 0 1 1 1 0 1]
 [1 0 0 1 1 0 0 1 1 1]
 [0 0 1 1 0 1 0 1 0 1]]
 {% endhighlight %}


### 4. CREATE A PIPELINE TO REMOVE PUNCTUATIONS, STOPWORDS AND PERFORM COUNT VECTORIZATION

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_10.jpg)

Let's define a pipeline to clean up all the messages. The pipeline performs the following: 
(1) remove punctuation 
(2) remove stopwords

{% highlight ruby %}
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
{% endhighlight %}

Applying function on the data.

{% highlight ruby %}
#=> Let's test the newly added function
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
{% endhighlight %}

show the cleaned up version

{% highlight ruby %}
print(tweets_df_clean[5])
{% endhighlight %}

RESULT:
{% highlight ruby %}
['22', 'huge', 'fan', 'fare', 'big', 'talking', 'leave', 'chaos', 'pay', 'disputes', 'get', 'allshowandnogo']
{% endhighlight %}

show the original version

{% highlight ruby %}
print(tweets_df['tweet'][5])
{% endhighlight %}

RESULT:
{% highlight ruby %}
[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo
{% endhighlight %}

Define the cleaning pipeline we defined earlier

{% highlight ruby %}
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning,  dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()
{% endhighlight %}

{% highlight ruby %}
tweets = pd.DataFrame(tweets_countvectorizer)
X = tweets
{% endhighlight %}


{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_11.JPG)

### 4. TRAIN A NAIVE BAYES CLASSIFIER MODEL

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_12.JPG)

Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_13.JPG)

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_14.JPG)

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_15.JPG)

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_16.JPG)

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_17.JPG)

{: .center}
![GOT]({{site.baseurl}}/assets/img/sentiment_18.JPG)

Train dataset.

{% highlight ruby %}
X.shape
{% endhighlight %}

RESULT:
`(31962, 47386)`

{% highlight ruby %}
y.shape
{% endhighlight %}

RESULT:
`(31962,)`

{% highlight ruby %}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
{% endhighlight %}



{% highlight ruby %}
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
{% endhighlight %}



 ### 6. What's up with Stannis Baratheon?

We can see that the importance of `Eddard Stark` dies off as the book series progresses. With `Jon Snow`, there is a drop in the fourth book but a sudden rise in the fifth book.

Now let's look at various other measures like betweenness centrality and PageRank to find important characters in our `Game of Thrones` character co-occurrence network and see if we can uncover some more interesting facts about this network. Let's plot the evolution of betweenness centrality of this network over the five books. We will take the evolution of the top four characters of every book and plot it.

{% highlight ruby %}
=># Creating a list of betweenness centrality of all the books just like we did for degree centrality
evol = [nx.betweenness_centrality(book, weight='weight') for book in books]

=># Making a DataFrame from the list
betweenness_evol_df = pd.DataFrame.from_records(evol)

#=> Finding the top 4 characters in every book
set_of_char = set()
for i in range(5):
    set_of_char |= set(list(betweenness_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

#=> Plotting the evolution of the top characters
betweenness_evol_df[list_of_char].plot(figsize=(13, 7))
{% endhighlight %}


{: .center}
![tree2]({{site.baseurl}}/assets/img/graph_GOT2.png)

### 7. What does the Google PageRank algorithm tell us about Game of Thrones?

We see a peculiar rise in the importance of `Stannis Baratheon` over the books. In the fifth book, he is significantly more important than other characters in the network, even though he is the third most important character according to `degree centrality`.

`PageRank` was the initial way Google ranked web pages. It evaluates the inlinks and outlinks of webpages in the world wide web, which is, essentially, a directed network. Let's look at the importance of characters in the Game of Thrones network according to `PageRank`.

{% highlight ruby %}
#=> Creating a list of pagerank of all the characters in all the books
evol = [nx.pagerank(book) for book in books]

#=> Making a DataFrame from the list
pagerank_evol_df = pd.DataFrame.from_records(evol)

#=> Finding the top 4 characters in every book
set_of_char = set()
for i in range(5):
    set_of_char |= set(list(pagerank_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

#=> Plotting the top characters
pagerank_evol_df[list_of_char].plot(figsize=(13, 7))
{% endhighlight %}

{: .center}
![tree]({{site.baseurl}}/assets/img/graph_GOT3.png)


### 8. Correlation between different measures

`Stannis`, `Jon Snow`, and `Daenerys` are the most important characters in the fifth book according to `PageRank`. `Eddard Stark` follows a similar curve but for `degree centrality` and `betweenness centrality`: He is important in the first book but dies into oblivion over the book series.

We have seen three different measures to calculate the importance of a node in a network, and all of them tells us something about the characters and their importance in the co-occurrence network. We see some names pop up in all three measures so maybe there is a strong correlation between them?

Let's look at the correlation between PageRank, betweenness centrality and degree centrality for the fifth book using Pearson correlation.

{% highlight ruby %}
#=> Creating a list of pagerank, betweenness centrality, degree centrality
#=> of all the characters in the fifth book.
measures = [nx.pagerank(books[4]), 
            nx.betweenness_centrality(books[4], weight='weight'), 
            nx.degree_centrality(books[4])]

#=> Creating the correlation DataFrame
cor = pd.DataFrame.from_records(measures)

#=> Calculating the correlation
cor.T.corr()
{% endhighlight %}

OUTPUT:

| 0 | 1 | 2 |
|---------:|---------:|--------:|
| 1.000000 | 0.793372 | 0.971493|
| 0.793372 | 1.000000 | 0.833816|
| 0.971493 | 0.833816 | 1.000000|

### 9. Conclusion

We see a high correlation between these three measures for our character `co-occurrence network`.

So we've been looking at different ways to find the important characters in the Game of Thrones `co-occurrence network`. According to `degree centrality`, `Eddard Stark` is the most important `character initially` in the books. But who is/are the most important character(s) in the fifth book according to these three measures.

{% highlight ruby %}
#=> Finding the most important character in the fifth book,  
#=> according to degree centrality, betweenness centrality and pagerank.
p_rank, b_cent, d_cent = cor.idxmax(axis=1)
{% endhighlight %}

`p_rank = 'Jon-Snow'`

`b_cent = 'Stannis-Baratheon'`

`d_cent = 'Jon-Snow'`

