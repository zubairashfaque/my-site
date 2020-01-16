---
layout: post
title: Tree Based Modeling from Scratch (Python)
date: 2020-01-16 00:00:00 +0300
description: In this blog, I will cover in detail my understanding of tree-based modes. This tutorial is for beginners to learn tree-based modeling from scratch. (optional)
img: Decision-Tree-Algorithm.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ZUBI_ASH, BLOG, DATACAMP, Decision_Tree_Algorithm] # add tag
---

In this blog, I will cover in detail my understanding of tree-based modes. This tutorial is for beginners to learn tree-based modeling from scratch.

###  Overview
*	Explaining of tree based models for classification and regression from scratch with their advantages and disadvantages.
*	Learn supervised machine learning concepts like Classification and Regression Trees (CART), Bagging and Random Forests, Boosting random forest, and ensemble methods.
*	Implementation of these tree based machine-learning algorithms in Python.

### Introduction

Decision tree are supervised learning models used in data mining. In other words, these models help us in finding new information in a provided dataset to solve problems involving classification and regression. Tree based learning algorithms are considered to be one of the best in terms of accuracy, stability, ease of understanding and flexibility. As discussed, these models have a high flexibility but that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to over fitting (memorizing the noise present in a dataset).

### 1. What is Decision Tree? 

A decision tree is like a flowchart. To put it another way, a decision-tree is data-structure consisting of hierarchy of individual units called nodes. Each internal node represents a “question” or “prediction” (e.g., whether a coin flip comes up heads or tails). 
There are three kinds of nodes.

* Root
* Internal node
* Leaf

The root is parent node or starting of a flowchart, a question-giving rise to two children nodes. An internal node having one parent node, question-giving rise to two children nodes. Leaf having one parent node with no children node and involving no questions; it is where prediction is made.
A decision tree is a tree in which each internal node is labeled with an input feature. The branch coming from a node labeled with an input feature are labeled with each of the possible values of the output feature or in other words the branch leads to a secondary decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes, telling that the data set has been classified by the tree either into a specific class, or into a particular probability distribution.

{: .center}
![I and My friends]({{site.baseurl}}/assets/img/tree-1.jpg)

{: .center}
Figure: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

In above tree diagram, consider the case where an instance move back and forth the tree to reach the leaf in left. In this leaf, there are 257 instances classified as benign and 7 instances classified as malignant. As a result, the tree's prediction for this instance would be benign.

#### Types of Decision Trees

Types of decision trees are based on the type of target variable we have. It can be of two types:
1.  **Classification Decision Tree**: A decision tree, which has categorical target variable then it called as classification decision tree also called variable decision tree. 
2.	**Regression Decision Tree**: A decision tree, which has continuous target variable then it is called as regression decision tree also continuous variable decision Tree.

In order to understand how a classification tree produces purest leafs we have to understand definition of information gain.

#### Information Gain

The nodes of a classification tree are grown recursively; in other words, the restraint to grow of an internal node of leaf depends on the state of its ancestor’s node. 

{: .center}
![tree]({{site.baseurl}}/assets/img/tree-2.jpg)

To produce the purist leaves possible, at each node, a tree asks the question involving one feature f and split-point sp. Now, million dollar question is that how does it know which feature and which spit-point to pick? It does so by maximizing information gain. The tree considers that every node contains information and aims at maximizing the information gain obtained after each split. Consider the case where a Node with N samples is split into a left-node with Nleft samples and a right-node with Nright samples. 

The information gain for such split is given by the formula shows below.

![information gain]({{site.baseurl}}/assets/img/tree-3.jpg)

A question that you may have in your mind here is: What is impurity and what is criterion is used to measure the impurity of a node? Look at the image below and think which node can be described easily. I am sure, your answer is C because it requires less information as all values are similar. On the other hand, B requires more information to describe it and A requires the maximum information. In other words, we can say that C is a Pure node, B is less Impure and A is more impure.

{: .center}
![tree]({{site.baseurl}}/assets/img/tree-4.jpg)

Now, we can build a conclusion that less impure node requires less information to describe it. And, more impure node requires more information. There are different criteria you can use to measure impurities of a node among which are the gini-index and entropy. 

#### What is Gini Index?

Gini index or Gini impurity measures the degree or probability of a particular variable being wrongly classified when it is randomly chosen. But what is actually meant by ‘impurity’? If all the elements belong to a single class, then it can be called pure. The degree of Gini index varies between 0 and 1, where 0 denotes that all elements belong to a certain class or if there exists only one class, and 1 denotes that the elements are randomly distributed across various classes. A Gini Index of 0.5 denotes equally distributed elements into some classes.



Emphasis, aka italics, with *asterisks* or _underscores_.

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

