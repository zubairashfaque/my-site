---
layout: post
title: Fake New Detection
date: 2020-06-25 00:00:00 +0300
description: In this blog, I will analyze thousands of news text to detect if it is fake or not. (optional)
img: fake_2.jpeg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ZUBI_ASH, PROJECT, NLP, FAKE_NEWS ] # add tag
---

##  Project Description

The goal of this hands-on project is to detect fake news using RNN (recurrent neural network). I will train a Bidirectional Neural Network and LSTM based deep learning model to detect fake news from a given news corpus. 

{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_1.jpeg)

## Introduction

The spread of false information has emerged as a pervasive force in recent years — upsetting elections, disrupting democratic societies, and further dividing people into fractious groups, stubbornly entrenched in destructive “us-versus-them” ideologies. With an estimated 20% to 38% of news stories shared on social media platforms being deemed as bogus, disinformation has, unfortunately, become the new norm, and it’s getter harder and harder to discern the truth from the bits of “fake news” floating around — whether it’s read in the written word, or seen in photographs or moving images.
To counter the problem, experts have come up with a variety of AI-powered “fake news” detectors.***

### 1. IMPORT LIBRARIES AND DATASETS

Let’s start with loading all the libraries and dependencies.


{% highlight ruby %}
!pip install plotly
!pip install --upgrade nbformat
!pip install nltk
!pip install spacy # spaCy is an open-source software library for advanced natural language processing
!pip install WordCloud
!pip install gensim # Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
{% endhighlight %}

load the data

{% highlight ruby %}
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")
{% endhighlight %}

We have two file one fake amd other true.


### 2. PERFORM EXPLORATORY DATA ANALYSIS

{% highlight ruby %}
df_true.head()
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_3.JPG)

{% highlight ruby %}
df_fake.head()
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_4.JPG)

Let's add a target class column to indicate whether the news is real or fake.

{% highlight ruby %}
df_true['isfake'] = 1
df_true.head()
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_5.JPG)

{% highlight ruby %}
df_fake['isfake'] = 0
df_fake.head()
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_6.JPG)

Concatenate Real and Fake News

{% highlight ruby %}
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_7.JPG)

We can see that there is a column named `date` which are just ids no use of it so we can remove it.

{% highlight ruby %}
df.drop(columns = ['date'], inplace = True)
{% endhighlight %}

Let's combine title and text together

{% highlight ruby %}
df['original'] = df['title'] + ' ' + df['text']
df.head()
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_8.JPG)


{% highlight ruby %}
df['original'][0]
{% endhighlight %}


{% highlight ruby %}
'As U.S. budget fight looms, Republicans flip their fiscal script WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary” spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump) administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” Meadows, chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats are saying that’s not enough, we need to give the government a pay raise of 10 to 11 percent. '
{% endhighlight %}


### 3. PERFORM DATA CLEANING

Let's download `stopwords`.

{% highlight ruby %}
nltk.download("stopwords")
{% endhighlight %}

Obtain additional stopwords from nltk

{% highlight ruby %}
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
{% endhighlight %}

We have to remove stopwords and remove words with 2 or less characters.

{% highlight ruby %}
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result
{% endhighlight %}

Apply the function to the dataframe

{% highlight ruby %}
df['clean'] = df['original'].apply(preprocess)
{% endhighlight %}

Show original news

{% highlight ruby %}
df['original'][0]
{% endhighlight %}


RESULT:
{% highlight ruby %}
'As U.S. budget fight looms, Republicans flip their fiscal script WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary” spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump) administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” '
{% endhighlight %}


Show cleaned up news after removing stopwords

{% highlight ruby %}
print(df['clean'][0])
{% endhighlight %}

{% highlight ruby %}
['budget', 'fight', 'looms', 'republicans', 'flip', 'fiscal', 'script', 'washington', 'reuters', 'head', 'conservative', 'republican', 'faction', 'congress', 'voted', 'month', 'huge', 'expansion', 'national', 'debt', 'cuts', 'called', 'fiscal', 'conservative', 'sunday', 'urged', 'budget', 'restraint', 'keeping', 'sharp', 'pivot', 'republicans', 'representative', 'mark', 'meadows', 'speaking', 'face', 'nation', 'drew', 'hard', 'line', 'federal', 'spending', 'lawmakers', 'bracing', 'battle', 'january', 'return', 'holidays', 'wednesday', 'lawmakers', 'begin', 'trying', 'pass', 'federal', 'budget', 'fight', 'likely', 'linked', 'issues', 'immigration', 'policy', 'november', 'congressional', 'election', 'campaigns', 'approach', 'republicans', 'seek', 'control', 'congress', 'president', 'donald', 'trump', 'republicans', 'want', 'budget', 'increase', 'military', 'spending', 'democrats', 'want', 'proportional', 'increases', 'defense', 'discretionary', 'spending', 'programs', 'support', 'education', 'scientific', 'research', 'infrastructure', 'public', 'health', 'environmental', 'protection', 'trump', 'administration', 'willing', 'going', 'increase', 'defense', 'discretionary', 'spending', 'percent', 'meadows', 'chairman', 'small', 'influential', 'house', 'freedom', 'caucus', 'said', 'program', 'democrats', 'saying', 'need', 'government', 'raise', 'percent', 'fiscal', 'conservative', 'rationale', 'eventually', 'people', 'money', 'said', 'meadows', 'republicans', 'voted', 'late', 'december', 'party', 'debt', 'financed', 'overhaul', 'expected', 'balloon', 'federal', 'budget', 'deficit', 'trillion', 'years']
{% endhighlight %}


{% highlight ruby %}
df
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_9.JPG)


Obtain the total words present in the dataset

{% highlight ruby %}
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
{% endhighlight %}


{% highlight ruby %}
len(list_of_words)
{% endhighlight %}

result:`9276947`

Obtain the total number of unique words

{% highlight ruby %}
total_words = len(list(set(list_of_words)))
total_words
{% endhighlight %}

result:`108704`

join the words into a string

{% highlight ruby %}
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_10.JPG)

{% highlight ruby %}
df['clean_joined'][0]
{% endhighlight %}


{% highlight ruby %}
'budget fight looms republicans flip fiscal script washington reuters head conservative republican faction congress voted month huge expansion national debt cuts called fiscal conservative sunday urged budget restraint keeping sharp pivot republicans representative mark meadows speaking face nation drew hard line federal spending lawmakers bracing battle january return holidays wednesday lawmakers begin trying pass federal budget fight likely linked issues immigration policy november congressional election campaigns approach republicans seek control congress president donald trump republicans want budget increase military spending democrats want proportional increases defense discretionary spending programs support education scientific research infrastructure public health environmental protection trump administration willing going increase defense discretionary spending percent meadows chairman small influential house freedom caucus said program democrats saying need government raise percent fiscal conservative rationale eventually people money said meadows republicans voted late december party debt financed overhaul expected balloon federal budget deficit trillion years trillion national debt interesting hear mark talk fiscal responsibility democratic representative joseph crowley said crowley said republican require united states borrow trillion paid future generations finance cuts corporations rich fiscally responsible bills seen passed history house representatives think going paying years come crowley said republicans insist package biggest overhaul years boost economy growth house speaker paul ryan supported recently went meadows making clear'
{% endhighlight %}

### 4. VISUALIZE CLEANED UP DATASET

Plot the number of samples in 'subject'

{% highlight ruby %}
plt.figure(figsize = (8, 8))
sns.countplot(y = "subject", data = df)
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_11.JPG)


Plot the count plot for fake vs. true news

{% highlight ruby %}
plt.figure(figsize = (8, 8))
sns.countplot(y = "subject", data = df)
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_12.JPG)


Plot the word cloud for text that is Real

{% highlight ruby %}
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_13.JPG)

Plot the word cloud for text that is Fake

{% highlight ruby %}
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
{% endhighlight %}

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_14.JPG)

Length of maximum document will be needed to create word embeddings 

{% highlight ruby %}
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)
{% endhighlight %}

{% highlight ruby %}
The maximum number of words in any document is = 4405
{% endhighlight %}

#### 5: PREPARE THE DATA BY PERFORMING TOKENIZATION AND PADDING

RESULT:
{: .center}
![GOT]({{site.baseurl}}/assets/img/fake_15.JPG)

Split data into test and train 

{% highlight ruby %}
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)
{% endhighlight %}


{% highlight ruby %}
from nltk import word_tokenize
{% endhighlight %}

Create a tokenizer to tokenize the words and create sequences of tokenized words

{% highlight ruby %}
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)
{% endhighlight %}

{% highlight ruby %}
print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])
{% endhighlight %}

{% highlight ruby %}
The encoding for document
 budget fight looms republicans flip fiscal script washington reuters head conservative republican faction congress voted month huge expansion national debt cuts called fiscal conservative sunday urged budget restraint keeping sharp pivot republicans representative mark meadows speaking face nation drew hard line federal spending lawmakers bracing battle january return holidays wednesday lawmakers begin trying pass federal budget fight likely linked issues immigration policy november congressional election campaigns approach republicans seek control congress president donald trump republicans want budget increase military spending democrats want proportional increases defense discretionary spending programs support education scientific research infrastructure public health environmental protection trump administration willing going increase defense discretionary spending percent meadows chairman small influential house freedom caucus said program democrats saying need government raise percent fiscal conservative rationale eventually people money said meadows republicans voted late december party debt financed overhaul expected balloon federal budget deficit trillion years trillion national debt interesting hear mark talk fiscal responsibility democratic representative joseph crowley said crowley said republican require united states borrow trillion paid future generations finance cuts corporations rich fiscally responsible bills seen passed history house representatives think going paying years come crowley said republicans insist package biggest overhaul years boost economy growth house speaker paul ryan supported recently went meadows making clear radio interview welfare entitlement reform party calls republican priority republican parlance entitlement programs mean food stamps housing assistance medicare medicaid health insurance elderly poor disabled programs created washington assist needy democrats seized ryan early december remarks saying showed republicans overhaul seeking spending cuts social programs goals house republicans seat senate votes democrats needed approve budget prevent government shutdown democrats leverage senate republicans narrowly control defend discretionary defense programs social spending tackling issue dreamers people brought illegally country children trump september march expiration date deferred action childhood arrivals daca program protects young immigrants deportation provides work permits president said recent twitter messages wants funding proposed mexican border wall immigration changes exchange agreeing help dreamers representative debbie dingell told favor linking issue policy objectives wall funding need daca clean said wednesday trump aides meet congressional leaders discuss issues followed weekend strategy sessions trump republican leaders white house said trump scheduled meet sunday florida republican governor rick scott wants emergency house passed billion package hurricanes florida texas puerto rico wildfires california package exceeded billion requested trump administration senate voted 
 is :  [2365, 558, 332, 2311, 2716, 42, 972, 27, 11043, 950, 513, 120, 258, 57, 30, 558, 332, 6402, 972]
 {% endhighlight %}

Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results

{% highlight ruby %}
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post') 
{% endhighlight %}

{% highlight ruby %}
for i,doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)
{% endhighlight %}

{% highlight ruby %}
The padded encoding for document 1  is :  [ 2365   558   332  2311  2716    42   972    27 11043   950   513   120  258    57    30   558   332  6402   972     0     0     0     0     0  0     0     0     0     0     0     0     0     0     0     0     0   0     0     0     0]
{% endhighlight %}

{% highlight ruby %}
The padded encoding for document 2  is :  [   49   183     5  3537   231    75  4423    20   877   694   751  4037 16    12    20   278   316   694   751   838    38   204 23342   844 1023   694 49568  9060     4  4423   348  4631   352    98    45    20 521   694   751   355]
{% endhighlight %}


#### 6: BUILD AND TRAIN THE MODEL 

{% highlight ruby %}
=># Sequential Model
model = Sequential()

=># embeddidng layer
model.add(Embedding(total_words, output_dim = 128))
=># model.add(Embedding(total_words, output_dim = 240))


=># Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

=># Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
{% endhighlight %}


{% highlight ruby %}
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, None, 128)         13914112  
_________________________________________________________________
bidirectional_3 (Bidirection (None, 256)               263168    
_________________________________________________________________
dense_6 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 129       
=================================================================
Total params: 14,210,305
Trainable params: 14,210,305
Non-trainable params: 0
_________________________________________________________________
{% endhighlight %}

{% highlight ruby %}
y_train = np.asarray(y_train)
{% endhighlight %}

Train the model

{% highlight ruby %}
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)
{% endhighlight %}

{% highlight ruby %}
Train on 32326 samples, validate on 3592 samples
Epoch 1/2
32326/32326 [==============================] - 321s 10ms/sample - loss: 0.0421 - acc: 0.9815 - val_loss: 0.0073 - val_acc: 0.9992
Epoch 2/2
32326/32326 [==============================] - 316s 10ms/sample - loss: 0.0016 - acc: 0.9997 - val_loss: 0.0096 - val_acc: 0.9981
<tensorflow.python.keras.callbacks.History at 0x229c16c58c8>
{% endhighlight %}

#### 7. ASSESS TRAINED MODEL PERFORMANCE

Let's make prediction

{% highlight ruby %}
pred = model.predict(padded_test)
{% endhighlight %}

If the predicted value is >0.5 it is real else it is fake

{% highlight ruby %}
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)
{% endhighlight %}     

Getting the accuracy

{% highlight ruby %}
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)
{% endhighlight %}   

`Model Accuracy :  0.9968819599109131`

#### 9. Please find below Fake News Classification jupyter notebook on github

[link to github!](https://github.com/zubairashfaque/NLP/tree/main/Fake%20News%20Classification)


`***` Taken from: https://thenewstack.io/mits-new-ai-tackles-loopholes-in-fake-news-detection-tools/
