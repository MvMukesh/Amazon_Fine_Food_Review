# `Amazon Food Reviews (Analysis & Modelling) Using Various Machine Learning Models`
## `Learnings`
* `Text Processing and Natural Language Processing`
* `SQLLite`
* `sklearn`
* `nltk` (natural language processing toolkit)
* `Handling Imbalanced Classification Problem`

![image](https://user-images.githubusercontent.com/26667491/144200308-f1629271-94e5-4139-a82c-588bdbcae994.png)

# `Problem Faced and Tackled`
1. `How to convert text(words/sentences) into numerical Vectors of d-dimension?`
   * Most important Features are Review and Text which are in simple English language, aimed to convert them into Vectors then applying `Linear Algebra` to it
      * Say draw a plane(need to find it) between d-dimensional datapoints, this plane divides data points in two parts, say -ve and +ve
      * assume a normal which is perpendicular to plane
      * data points in direction of normal are positive and on opposit direction are negative (this can be given to any model)
2. `How to find a good plane to seperate data points?` 
3. `What Rules this conversion(text to d-dimension) will follow?`
   * Say we have 1,2,3 reviews and `Sementic Similarity`(Englist Similarity) between (1,2) is greater then (1,3) i.e. if we finds out that (1,2) are more similar reviews then (1,3)
   * for review 1,2,3 we have vectors v1,v2,v3 then
      *  distance between (v1,v2) is smaller then (v1,v3) i.e. if they are more similar then there distance will be less
      *  Geomatrically distance b/w (v1,v2) = (v1-v2)
   * Can say if review(1,2) are more similar then there distance(v1,v2) will be less, `or similar points are closer`
   * **`If all +ve and -ve points will be in there closer boundery it will become easy to draw plane between them and seprate them in two groups`**
4. `How to find a Method which takes Text as input and convert it into d-dimensional vactor, such that similar text must be closer Geometrically(distance must be least between similar reviews)?`
   * Bag-of-Words (simplest technique to change words to vectors) going down this list complexity increases
   * Tf-idf (term frequency-inverse document frequency)
   * Word2vec (technique for natural language processing)
   * Averag Word2vec
   * Tf-idf and Word2Vec

### `How Bag-of-Words works?`
**`set of all Text document is Called a Corpus` => Corpus means collection of documents(each Review is called as a Document in NLP)** 
1. It constructs a dictionary(not similar to python) inside this dictionary Set of all unique words in Review(`called as a document in NLP`) will be collected
   * Say Review is 'This is good product' => constructed by BOW as {'This', 'is', 'good', 'product'} <= set of unique words taken from reviews
     * Assume we have `n` number of reviews(`called as a document in NLP`) and then we have `d-unique words` across all my reviews(document) and `d-vectors` which are vector representation of d-unique words. for example: {v1,v2,v3,v4} will correspond to {'This', 'is', 'good', 'product'} 
       * `Keep in mind each Word is a different dimension of d-dimension_vectors`
     * Now update every word count(corpus) in Review and assign that number to its vector. for example: how many time This comes say 1 time or 2 time assign this number to say v1 which was associated with word `This`, as all other words in d-dimensions are taken in considration so there corresponding vector values will be filled with 0.
     * Due to this Vectors for particular sentence we take are generaly `very sparse`(sparse vactor means most of the values are 0 only some are non-0)

2. 
---
### `AIM(Framing it into ML Problem)`:
Given a text review, determine sentiment of review whether its positive or negative

### `Business Relevance`:
Say a business have a product, only by undestanding product reviews, they can understand what to add,modify or remove in that product. +ve review tells Business what to consider while making new product and opposit goes for negative reviews

### `How to Determine if Review is +ve or -ve`
Score/Rating can be used
* Rating which are 4 or 5 could be considered as +ve_rating
* Rating which are 1 or 2 could be considered as -ve_rating
* Rating which are 3 could be considered as neutral or ignored
This can be seen as a approximate or proxy way of determining polarity or review

#### `Data Source`: [Amazon Fine Foods Review](https://www.kaggle.com/snap/amazon-fine-food-reviews)

### `Data Includes`:
Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon
* Number of `reviews`: 568,454
* Number of `users`: 256,059
* Number of `products`: 74,258
* 260 users with > 50 reviews
* `Timespan`: Oct 1999 - Oct 2012 (13 years dataset)
* Number of `Attributes/Columns` in data: 10

### `Attribute Information`:
1. Id
2. ProductId - unique identifier for product
3. UserId - unqiue identifier for the user
4. ProfileName - profile name of user
5. HelpfulnessNumerator - number of users who found review helpful => Users who said review is helpfull say 2000people, we dont take people who said review is not Usefull in this counting
6. HelpfulnessDenominator - number of users who indicated whether they found review helpful or not => we add both users who said review is usefull and not usefull say, people number who find review usefull=2000 and notusefull=100, so HelpfulnessDenominator in this case will be (2000+100)
7. Score - rating between 1 and 5
8. Time - timestamp when review was given
9. Summary - brief summary of the review (written at the top of the review)
10. Text - text of review


### `Task Bird Eye View`:
* Data Cleaning
* Data Visualization
* Text Featurization(BOW, tfidf, Word2Vec)
* Building several Machine Learning models like 
  * Logistic Regression
  * KNN
  * SVM
  * Naive Bayes
  * Random Forest
  * GBDT
  * LSTM(RNNs) etc
----
----
# `Analysis Task`

##### amazon-fine-foods-review-analysis
![image](https://user-images.githubusercontent.com/26667491/143383141-41cf4f75-cae8-41f9-b770-8c2abd20b22f.png)

## Introduction

Given longitudinal data, one should be able to understand how things change over time. McAuley and Leskovec published a paper in 2013 detailing how they used Amazon's gourmet food section to build a recommendation classifier which builds upon experience of a reviewer. Using this longitudinal dataset, there should be many things that could be understood from looking into data. 
For instnace, we could potentially see trends of food over years and maybe even capture cupcake craze of 2011

## `Goals` of this analysis:  

* Understanding evolution of reviewers over time
* Understanding variation of helpfulness of reviews for products over time
* Visualize changes in reviews over 10 year period to understand what trends were important that year

# `Results`
Several results gathered through analysis notebook  

* Review lengths over time become longer  
* Semantic prediction of summary and review text weakly but significantly correlated according to pearson correlation  
* Summary of reviews also get slightly longer over time  
* Older a product is, more variation for review scores   
* Helpfulness_Ratio generally increases overtime for a product  
* Scatter_text plots showed evolution of Amazon platform and shows transition from movies to foods
 * Earlier positive ratings stemmed from specific products but slowly shifted towards more sentiment based relationship.  

## `Analysis Conclusion`
This dataset for Fine Foods Reviews shows several trends
* As writers become more experienced, length of their reviews get longer
* In addition, Summary of reviews also seem to get longer
* Additionally, Older_Products have more variation in Review_Scores
*  Finally, using scattertext some of text features which are associated with higher or lower ratings for product which eventually shifts to a more semnatic based relationship can be seen

----
----
____
`As this work will progress I will try to update this framework along with Proper Conclusions`
____
![image](https://user-images.githubusercontent.com/26667491/143769781-a9982064-5336-4896-a677-3c89949e7cc8.png)
____

# `Modelling Task`
## `1`-`Amazon Food Reviews EDA, NLP, Text Preprocessing and Visualization using ``t-SNE` `i.e `t-distributed stochastic neighbor embedding`
#### Steps:
1. Define `Problem Statement`
2. Performe Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset plotted `Word Clouds`, `Distplots`, `Histograms` etc.
3. Performe `Data Cleaning` & `Data Preprocessing` by removing unneccesary and duplicates rows and for text reviews `removed html tags`, punctuations, `Stopwords` and `Stemmed words` using `Porter Stemmer`
4. Document concepts clearly
5. `Plot t-SNE` plots for Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
----
----
## `2`-`Logistic Regression`
#### Steps:
1. Apply Logistic Regression on Different Featurization of Data viz. `BOW(uni-gram)`, `tfidf`, `Avg-Word2Vec` and `tf-idf-Word2Vec`
2. Use both `Grid Search` & `Randomized Search` Cross Validation
3. Evaluate test data on various performance metrics like `accuracy`, `f1-score`, `precision`, `recall` etc. also plot `Confusion matrix` using seaborne
4. Showe how `Sparsity` increases as we increase `lambda` or decrease `C` when `L1 Regularizer` is used for each `featurization`
5. Do `Pertubation test` to check whether features are `multi-collinear` or not

#### `Conclusions`:
* -
* -

----
----

## `3`-`K-nearest neighbors`
#### Steps:
1. Apply K-Nearest Neighbour on Different Featurization of Data viz. `BOW(uni-gram)`, `tfidf`, `Avg-Word2Vec` and `tf-idf-Word2Vec`
2. Use both `brute` & `kd-tree` implementation of KNN
3. Evaluate test data on various performance metrics like `accuracy` also plotted `Confusion matrix` using `seaborne`

#### `Conclusions`:
* -
* -

----
----

## `4`-`Naive Bayes`
#### Steps:
1. Apply Naive Bayes using `Bernoulli NB` and `Multinomial NB` on Different Featurization of Data viz. BOW(uni-gram), tfidf.
2. Evaluate test data on various performance metrics like `accuracy`, `f1-score`, `precision`, `recall` etc. also plotted `Confusion Matrix` using seaborne
3. Print Top 25 Important Features for both Negative and Positive Reviews

#### `Conclusions`:
* -
* -

----
----

## `5`-`SVM`
#### Steps:
1. Apply SVM with `rbf`(radial basis function) kernel on different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use both Grid Search & Randomized Search Cross Validation
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Evaluate `SGDClassifier` on best resulting featurization

#### `Conclusions`:
* -
* -

----
----

## `6`-`Decision Trees`
#### Steps:
1. Apply Decision Trees on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use Grid Search with random 30 points for getting best max_depth
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Plot Feature Importance recieved from Decision Tree Classifier

#### `Conclusions`:
* -
* -

----
----
## `7`-`Ensembles(RF&GBDT)`
#### Steps:
1. Apply `Random Forest` on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use Grid Search with random 30 points for getting best max_depth, learning rate and n_estimators
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Plot `World Cloud` of feature importance recieved from `RF and GBDT Classifier`

#### `Conclusions`:
* -
* -


----
## Analysis References  
J. McAuley and J. Leskovec. [From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews.](http://i.stanford.edu/~julian/pdfs/www13.pdf). WWW, 2013.
