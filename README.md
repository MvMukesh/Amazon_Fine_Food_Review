# `Amazon Food Reviews (Analysis & Modelling) Using Various Machine Learning Models`
## `Learnings`
* `Text Processing and Natural Language Processing`
* `SQLLite`
* `sklearn`
* `nltk` (natural language processing toolkit)
![image](https://user-images.githubusercontent.com/26667491/144200308-f1629271-94e5-4139-a82c-588bdbcae994.png)

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
