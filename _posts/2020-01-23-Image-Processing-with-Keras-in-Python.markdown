---
layout: post
title: Image Processing With Neural Networks using Keras
date: 2020-01-23 00:00:00 +0300
description: In this blog, I will discuss Image Processing With Neural Networks using Keras. (optional)
img: Keras-Basic.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ZUBI_ASH, Keras_Basic, DATACAMP, Keras, CNN, convolutional_neural_network] # add tag
---

In this blog, I will discuss Image Processing with Neural Networks using Keras.

##  Keras_Basic 

### Images as data: visualizations

To display image data, you will rely on Python's Matplotlib library, and specifically use matplotlib's pyplot sub-module, that contains many plotting commands. Some of these commands allow you to display the content of images stored in arrays.

{% highlight ruby %}
#=> Import matplotlib
import matplotlib.pyplot as plt

#=> Load the image
data = plt.imread('bricks.png')

#=> Display the image
plt.imshow(data)
plt.show()
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/k_img_1.JPG)

### Images as data: changing images

To modify an image, you can modify the existing numbers in the array. In a color image, you can change the values in one of the color channels without affecting the other colors, by indexing on the last dimension of the array.


{% highlight ruby %}
#=> Set the red channel in this part of the image to 1
data[0:10, 0:10, 0] = 1

#=> Set the green channel in this part of the image to 0
data[0:10, 0:10, 2] = 0

#=> Set the blue channel in this part of the image to 0
data[0:10, 0:10, 1] = 0

#=> Visualize the result
plt.imshow(data)
plt.show()
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/k_img_2.JPG)


### create a one-hot encoding

Neural networks expect the labels of classes in a dataset to be organized in a one-hot encoded manner: each row in the array contains zeros in all columns, except the column corresponding to a unique label, which is set to 1.

{% highlight ruby %}
#=> The number of image categories
n_categories = 3

#=> The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

#=> Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

#=> Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1
{% endhighlight %}

`Game of Thrones` is a TV series based on the novel `A Song Of Ice And Fire` written by `George RR Martin`. Most Important characters are `Jon Snow`, `Daenerys Targaryen`, or `Tyrion Lannister`. Let us find out who is the most `important character` in `Game of Thrones`? The importance of character could be analyzed by their `co-occurrence network` and its `evolution` over the five books in R.R. Martin's. We will look at how the importance of the characters changes over the books using different centrality measures.

In this project we will use `networkx` package and different network centrality measures.In this project uses a dataset parsed by `Andrew J. Beveridge` and `Jie Shan` which is available [here](https://github.com/mathbeveridge/asoiaf). For more information on this dataset have a look at the [Network of Thrones blog](https://networkofthrones.wordpress.com/).


## Introduction

Decision tree are supervised learning models used in data mining. In other words, these models help us in finding new information in a provided dataset to solve problems involving `classification` and `regression`. Tree based learning algorithms are considered to be one of the best in terms of `accuracy`, `stability`, `ease of understanding` and `flexibility`. As discussed, these models have a high flexibility but that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to over fitting (memorizing the noise present in a dataset).

### 1. Let's load and explore the data? 

This dataset constitutes a network and is given as a text file describing the edges between characters, with some attributes attached to each edge. Let's start by loading in the data for the first book A Game of Thrones and inspect it.

{% highlight ruby %}
#=> Importing modules
import pandas as pd
#=>  Reading in book1.csv
book1 = pd.read_csv("book1.csv")
#=> Printing out the head of the dataset
print(book1.head())
{% endhighlight %}

OUTPUT.

| Source | Target | Type | weight | book |
|---------------:|----------------:|-----------:|--:|--:|
| Addam-Marbrand | Jaime-Lannister | Undirected | 3 | 1 |
| Addam-Marbrand | Tywin-Lannister | Undirected | 6 | 1 |
| Aegon-I-Targaryen | Daenerys-Targaryen | Undirected | 5 | 1 |
| Aegon-I-Targaryen | Eddard-Stark | Undirected  | 4 | 1 |
| Aemon-Targaryen-(Maester-Aemon) | Alliser-Thorne | Undirected | 4 | 1 | 

### 2. Time to find Network of Thrones? 

The resulting DataFrame book1 has 5 columns: `Source`, `Target`, `Type`, `weight`, and `book`.
Before diving into details we have to understand the concept of the Nodes and Edges concept in NetworkX which could be depicted in the following picture.

{: .center}
![NetworkX]({{site.baseurl}}/assets/img/Node_edge.JPG)

`Source` and `target` are the two `nodes` that are linked by an `edge`. A network can have directed or undirected edges and in this network all the edges are undirected. The weight attribute of every edge tells us the number of `interactions that the characters` have had over the book, and the book column tells us the book number.

Once we have the data loaded as a pandas DataFrame, it's time to create a network. We will use networkx, a network analysis library, and create a graph object for the first book.

{% highlight ruby %}
#=> Importing networkx module
import networkx as nx 
#=> Creating an empty graph object
G_book1 = nx.Graph()
{% endhighlight %}

### 3. Populate the network with the DataFrame

Currently, the graph object `G_book1` is empty. Let's now populate it with the `edges` from `book1`. And while we're at it, let's load in the rest of the books too!

{% highlight ruby %}
=># Iterating through the DataFrame to add edges
for index, edge in book1.iterrows():
    G_book1.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
=># Creating a list of networks for all the books
books = [G_book1]
book_fnames = ['book2.csv', 'book3.csv', 'book4.csv', 'book5.csv']
for book_fname in book_fnames:
    book = pd.read_csv(book_fname)
    G_book = nx.Graph()
    for index, edge in book.iterrows():
        G_book.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
    books.append(G_book)
{% endhighlight %}
 

 ### 4. Time to find the most important character in Game of Thrones
 
 Is it `Jon Snow`, `Tyrion`, `Daenerys`, or someone else? Let's see! Network Science offers us many different `metrics` to measure the importance of a node in a network. Note that there is no "correct" way of calculating the most important `node` in a network, every `metric` has a different meaning.

First, let's measure the importance of a node in a network by looking at the number of neighbors it has, that is, the number of nodes it is connected to. For example, an `influential` account on `Twitter`, where the follower-followee relationship forms the network, is an account which has a high number of followers. This measure of importance is called `degree centrality`.

Using this measure, let's extract the top ten important characters from the first book `(book[0])` and the fifth book `(book[4])`.

{% highlight ruby %}
=># Calculating the degree centrality of book 1
deg_cen_book1 = nx.degree_centrality(books[0])

=># Calculating the degree centrality of book 5
deg_cen_book5 = nx.degree_centrality(books[4])

sorted_deg_cen_book1 = sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:10]
sorted_deg_cen_book5 = sorted(deg_cen_book5.items(), key=lambda x:x[1], reverse=True)[0:10]

=># Printing out the top 10 of book1 and book5
print(sorted_deg_cen_book1)
print(sorted_deg_cen_book5)
{% endhighlight %}

OUTPUT1: `[('Eddard-Stark', 0.3548387096774194), ('Robert-Baratheon', 0.2688172043010753), ('Tyrion-Lannister', 0.24731182795698928), ('Catelyn-Stark', 0.23118279569892475), ('Jon-Snow', 0.19892473118279572), ('Robb-Stark', 0.18817204301075272), ('Sansa-Stark', 0.18817204301075272), ('Bran-Stark', 0.17204301075268819), ('Cersei-Lannister', 0.16129032258064518), ('Joffrey-Baratheon', 0.16129032258064518)]`

OUTPUT2: `[('Jon-Snow', 0.1962025316455696), ('Daenerys-Targaryen', 0.18354430379746836), ('Stannis-Baratheon', 0.14873417721518986), ('Tyrion-Lannister', 0.10443037974683544), ('Theon-Greyjoy', 0.10443037974683544), ('Cersei-Lannister', 0.08860759493670886), ('Barristan-Selmy', 0.07911392405063292), ('Hizdahr-zo-Loraq', 0.06962025316455696), ('Asha-Greyjoy', 0.056962025316455694), ('Melisandre', 0.05379746835443038)]`

 ### 5. Evolution of importance of characters over the books

According to `degree centrality`, the most important character in the first book is Eddard Stark but he is not even in the top 10 of the fifth book. The importance of characters changes over the course of five books because, you know, stuff happens... ;)

Let's look at the evolution of degree centrality of a couple of characters like `Eddard Stark`, `Jon Snow`, and `Tyrion`, which showed up in the top 10 of `degree centrality` in the `first book`.

{% highlight ruby %}
%matplotlib inline

=># Creating a list of degree centrality of all the books
evol = [nx.degree_centrality(book) for book in books]
 
=># Creating a DataFrame from the list of degree centralities in all the books
degree_evol_df = pd.DataFrame.from_records(evol)

=># Plotting the degree centrality evolution of Eddard-Stark, Tyrion-Lannister and Jon-Snow
degree_evol_df[['Eddard-Stark', 'Tyrion-Lannister', 'Jon-Snow']].plot()
{% endhighlight %}

{: .center}
![tree1]({{site.baseurl}}/assets/img/graph_GOT.png)

 ### 6. What's up with Stannis Baratheon?

We can see that the importance of `Eddard Stark` dies off as the book series progresses. With `Jon Snow`, there is a drop in the fourth book but a sudden rise in the fifth book.

Now let's look at various other measures like betweenness centrality and PageRank to find important characters in our `Game of Thrones` character co-occurrence network and see if we can uncover some more interesting facts about this network. Let's plot the evolution of betweenness centrality of this network over the five books. We will take the evolution of the top four characters of every book and plot it.

{% highlight ruby %}
=># Creating a list of betweenness centrality of all the books just like we did for degree centrality
evol = [nx.betweenness_centrality(book, weight='weight') for book in books]

=># Making a DataFrame from the list
betweenness_evol_df = pd.DataFrame.from_records(evol)

=># Finding the top 4 characters in every book
set_of_char = set()
for i in range(5):
    set_of_char |= set(list(betweenness_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

=># Plotting the evolution of the top characters
betweenness_evol_df[list_of_char].plot(figsize=(13, 7))
{% endhighlight %}


{: .center}
![tree2]({{site.baseurl}}/assets/img/graph_GOT2.png)

### 7. What does the Google PageRank algorithm tell us about Game of Thrones?

We see a peculiar rise in the importance of `Stannis Baratheon` over the books. In the fifth book, he is significantly more important than other characters in the network, even though he is the third most important character according to `degree centrality`.

`PageRank` was the initial way Google ranked web pages. It evaluates the inlinks and outlinks of webpages in the world wide web, which is, essentially, a directed network. Let's look at the importance of characters in the Game of Thrones network according to `PageRank`.

{% highlight ruby %}
=># Creating a list of pagerank of all the characters in all the books
evol = [nx.pagerank(book) for book in books]

=># Making a DataFrame from the list
pagerank_evol_df = pd.DataFrame.from_records(evol)

=># Finding the top 4 characters in every book
set_of_char = set()
for i in range(5):
    set_of_char |= set(list(pagerank_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

=># Plotting the top characters
pagerank_evol_df[list_of_char].plot(figsize=(13, 7))
{% endhighlight %}

{: .center}
![tree]({{site.baseurl}}/assets/img/graph_GOT3.png)


### 8. Correlation between different measures

`Stannis`, `Jon Snow`, and `Daenerys` are the most important characters in the fifth book according to `PageRank`. `Eddard Stark` follows a similar curve but for `degree centrality` and `betweenness centrality`: He is important in the first book but dies into oblivion over the book series.

We have seen three different measures to calculate the importance of a node in a network, and all of them tells us something about the characters and their importance in the co-occurrence network. We see some names pop up in all three measures so maybe there is a strong correlation between them?

Let's look at the correlation between PageRank, betweenness centrality and degree centrality for the fifth book using Pearson correlation.

{% highlight ruby %}
=># Creating a list of pagerank, betweenness centrality, degree centrality
=># of all the characters in the fifth book.
measures = [nx.pagerank(books[4]), 
            nx.betweenness_centrality(books[4], weight='weight'), 
            nx.degree_centrality(books[4])]

=># Creating the correlation DataFrame
cor = pd.DataFrame.from_records(measures)

=># Calculating the correlation
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
=># Finding the most important character in the fifth book,  
=># according to degree centrality, betweenness centrality and pagerank.
p_rank, b_cent, d_cent = cor.idxmax(axis=1)
{% endhighlight %}

`p_rank = 'Jon-Snow'`

`b_cent = 'Stannis-Baratheon'`

`d_cent = 'Jon-Snow'`


