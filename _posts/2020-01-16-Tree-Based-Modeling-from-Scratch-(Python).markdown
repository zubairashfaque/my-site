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

####  Overview
•	Explaining of tree based models for classification and regression from scratch with their advantages and disadvantages.

•	Learn supervised machine learning concepts like Classification and Regression Trees (CART), Bagging and Random Forests, Boosting random forest, and ensemble methods.

•	Implementation of these tree based machine-learning algorithms in Python.

### Introduction

Decision tree are supervised learning models used in data mining. In other words, these models help us in finding new information in a provided dataset to solve problems involving classification and regression. Tree based learning algorithms are considered to be one of the best in terms of accuracy, stability, ease of understanding and flexibility. As discussed, these models have a high flexibility but that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to over fitting (memorizing the noise present in a dataset).

### 1. What is Decision Tree? 

A decision tree is like a flowchart. To put it another way, a decision-tree is data-structure consisting of hierarchy of individual units called nodes. Each internal node represents a “question” or “prediction” (e.g., whether a coin flip comes up heads or tails). 
There are three kinds of nodes 
I.	Root
II.	Internal node
III.	Leaf
The root is parent node or starting of a flowchart, a question-giving rise to two children nodes. An internal node having one parent node, question-giving rise to two children nodes. Leaf having one parent node with no children node and involving no questions; it is where prediction is made.
A decision tree is a tree in which each internal node is labeled with an input feature. The branch coming from a node labeled with an input feature are labeled with each of the possible values of the output feature or in other words the branch leads to a secondary decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes, telling that the data set has been classified by the tree either into a specific class, or into a particular probability distribution.


![Breast Cancer Wisconsin (Diagnostic) Data Set]({{site.baseurl}}/assets/img/tree-1.jpg)

Man bun umami keytar 90's lomo drinking vinegar synth everyday carry +1 bitters kinfolk raclette meggings street art heirloom. Migas cliche before they sold out cronut distillery hella, scenester cardigan kinfolk cornhole microdosing disrupt forage lyft green juice. Tofu deep v food truck live-edge edison bulb vice. Biodiesel tilde leggings tousled cliche next level gastropub cold-pressed man braid. Lyft humblebrag squid viral, vegan chicharrones vice kinfolk. Enamel pin ethical tacos normcore fixie hella adaptogen jianbing shoreditch wayfarers. Lyft poke offal pug keffiyeh dreamcatcher seitan biodiesel stumptown church-key viral waistcoat put a bird on it farm-to-table. Meggings pitchfork master cleanse pickled venmo. Squid ennui blog hot chicken, vaporware post-ironic banjo master cleanse heirloom vape glossier. Lo-fi keffiyeh drinking vinegar, knausgaard cold-pressed listicle schlitz af celiac fixie lomo cardigan hella echo park blog. Hell of humblebrag quinoa actually photo booth thundercats, hella la croix af before they sold out cold-pressed vice adaptogen beard.

### Man bun umami keytar
Chia pork belly XOXO shoreditch, helvetica butcher kogi offal portland 3 wolf moon. Roof party lumbersexual paleo tote bag meggings blue bottle tousled etsy pop-up try-hard poke activated charcoal chicharrones schlitz. Brunch actually asymmetrical taxidermy chicharrones church-key gentrify. Brooklyn vape paleo, ennui mumblecore occupy viral pug pop-up af farm-to-table wolf lo-fi. Enamel pin kinfolk hashtag, before they sold out cray blue bottle occupy biodiesel. Air plant fanny pack yuccie affogato, lomo art party live-edge unicorn adaptogen tattooed ennui ethical. Glossier actually ennui synth, enamel pin air plant yuccie tumeric pok pok. Ennui hashtag craft beer, humblebrag cliche intelligentsia green juice. Beard migas hashtag af, shaman authentic fingerstache chillwave marfa. Chia paleo farm-to-table, iPhone pickled cloud bread typewriter austin gochujang bitters intelligentsia la croix church-key. Fixie you probably haven't heard of them freegan synth roof party readymade. Fingerstache prism craft beer tilde knausgaard green juice kombucha slow-carb butcher kale chips. Snackwave organic tbh ennui XOXO. Hell of woke blue bottle, tofu roof party food truck pok pok thundercats. Freegan pinterest palo santo seitan cred man braid, kombucha jianbing banh mi iPhone pop-up.

>Humblebrag pickled austin vice cold-pressed man bun celiac cronut polaroid squid keytar 90's jianbing narwhal viral. Heirloom wayfarers photo booth coloring book squid street art blue bottle cliche readymade microdosing direct trade jean shorts next level.

Selvage messenger bag meh godard. Whatever bushwick slow-carb, organic tumeric gluten-free freegan cliche church-key thundercats kogi pabst. Hammock deep v everyday carry intelligentsia hell of helvetica. Occupy affogato pop-up bicycle rights paleo. Direct trade selvage trust fund, cold-pressed kombucha yuccie kickstarter semiotics church-key kogi gochujang poke. Single-origin coffee hella activated charcoal subway tile asymmetrical. Adaptogen normcore wayfarers pickled lomo. Ethical edison bulb shaman wayfarers cold-pressed woke. Helvetica selfies blue bottle deep v. Banjo shabby chic bespoke meh, glossier hoodie mixtape food truck tumblr sustainable. Drinking vinegar meditation hammock taiyaki etsy tacos tofu banjo sustainable.

Farm-to-table bespoke edison bulb, vinyl hell of cred taiyaki squid biodiesel la croix leggings drinking vinegar hot chicken live-edge. Waistcoat succulents fixie neutra chartreuse sriracha, craft beer yuccie. Ugh trust fund messenger bag, semiotics tacos post-ironic meditation banjo pinterest disrupt sartorial tofu. Meh health goth art party retro skateboard, pug vaporware shaman. Meh whatever microdosing cornhole. Hella salvia pinterest four loko shabby chic yr. Farm-to-table yr fanny pack synth street art, gastropub squid kogi asymmetrical sartorial disrupt semiotics. Kombucha copper mug vice sriracha +1. Tacos hashtag PBR&B taiyaki franzen cornhole. Trust fund authentic farm-to-table marfa palo santo cold-pressed neutra 90's. VHS artisan drinking vinegar readymade yr. Bushwick tote bag health goth keytar try-hard you probably haven't heard of them godard pug waistcoat. Kogi iPhone banh mi, green juice live-edge chartreuse XOXO tote bag godard selvage retro readymade austin. Leggings ramps tacos iceland raw denim semiotics woke hell of lomo. Brooklyn woke adaptogen normcore pitchfork skateboard.

Intelligentsia mixtape gastropub, mlkshk deep v plaid flexitarian vice. Succulents keytar craft beer shabby chic. Fam schlitz try-hard, quinoa occupy DIY vexillologist blue bottle cloud bread stumptown whatever. Sustainable cloud bread beard fanny pack vexillologist health goth. Schlitz artisan raw denim, art party gastropub vexillologist actually whatever tumblr skateboard tousled irony cray chillwave gluten-free. Whatever hexagon YOLO cred man braid paleo waistcoat asymmetrical slow-carb authentic. Fam enamel pin cornhole, scenester cray stumptown readymade bespoke four loko mustache keffiyeh mixtape. Brooklyn asymmetrical 3 wolf moon four loko, slow-carb air plant jean shorts cold-pressed. Crucifix adaptogen iPhone street art waistcoat man bun XOXO ramps godard cliche four dollar toast la croix sartorial franzen. Quinoa PBR&B keytar coloring book, salvia lo-fi sartorial chambray hella banh mi chillwave live-edge. Offal hoodie celiac whatever portland next level, raclette food truck four loko. Craft beer kale chips banjo humblebrag brunch ugh. Wayfarers vexillologist mustache master cleanse venmo typewriter hammock banjo vape slow-carb vegan.
