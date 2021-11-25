*# Amazon-Food-Reviews-Analysis-and-Modelling Using Various Machine Learning Models

### `Task Bird Eye View`:
* Exploratory Data Analysis
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
### `AIM`:
Given a text review, determine sentiment of review whether its positive or negative

#### `Data Source`: https://www.kaggle.com/snap/amazon-fine-food-reviews

### `Data Includes`:
Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon
* Number of `reviews`: 568,454
* Number of `users`: 256,059
* Number of `products`: 74,258
* 260 users with > 50 reviews
* `Timespan`: Oct 1999 - Oct 2012
* Number of `Attributes/Columns` in data: 10

### `Attribute Information`:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review
----
----
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
* Sparsity increases as we decrease C (increase lambda) when we use L1 Regularizer for regularization
* TF_IDF Featurization performs best with F1_score of 0.967 and Accuracy of 91.39
* Features are multi-collinear with different featurization
* Logistic Regression is faster algorithm

----
----

## `3`-`K-nearest neighbors`
#### Steps:
1. Apply K-Nearest Neighbour on Different Featurization of Data viz. `BOW(uni-gram)`, `tfidf`, `Avg-Word2Vec` and `tf-idf-Word2Vec`
2. Use both `brute` & `kd-tree` implementation of KNN
3. Evaluate test data on various performance metrics like `accuracy` also plotted `Confusion matrix` using `seaborne`

#### `Conclusions`:
* KNN is a very slow Algorithm takes very long time to train
* Best Accuracy is achieved by Avg Word2Vec Featurization which is of 89.38%
* Both kd-tree and brute algorithms of KNN gives comparatively similar results
* Overall KNN is not that good for this dataset
----
----

## `4`-`Naive Bayes`
#### Steps:
1. Apply Naive Bayes using `Bernoulli NB` and `Multinomial NB` on Different Featurization of Data viz. BOW(uni-gram), tfidf.
2. Evaluate test data on various performance metrics like `accuracy`, `f1-score`, `precision`, `recall` etc. also plotted `Confusion Matrix` using seaborne
3. Print Top 25 Important Features for both Negative and Positive Reviews

#### `Conclusions`:
* Naive Bayes is much faster algorithm than KNN
* Performance of Bernoulli NB is way much more better than Multinomial NB
* Best F1 score is acheived by BOW featurization which is appx 0.9342

----
----

## `5`-`SVM`
#### Steps:
1. Apply SVM with `rbf`(radial basis function) kernel on different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use both Grid Search & Randomized Search Cross Validation
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Evaluate `SGDClassifier` on best resulting featurization

#### `Conclusions`:
* BOW Featurization with linear kernel with grid search gave best results with F1-score of 0.9201
* `Using SGDClasiifier takes very less time to train`

----
----

## `6`-`Decision Trees`
#### Steps:
1. Apply Decision Trees on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use Grid Search with random 30 points for getting best max_depth
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Plot Feature Importance recieved from Decision Tree Classifier

#### `Conclusions`:
* BOW Featurization(max_depth=8) gave best results with accuracy of 85.8% and F1-score of 0.858
* `Decision Trees on BOW and tfidf would have taken forever if had taken all dimensions as it had huge dimension and hence tried with max 8 as max_depth`

----
----
## `7`-`Ensembles(RF&GBDT)`
#### Steps:
1. Apply `Random Forest` on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2. Use Grid Search with random 30 points for getting best max_depth, learning rate and n_estimators
3. Evaluate test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plot Confusion matrix using seaborne
4. Plot `World Cloud` of feature importance recieved from `RF and GBDT Classifier`

#### `Conclusions`:
* TFIDF Featurization in Random Forest (BASE-LEARNERS=10) with random search gave best results with F1-score of 0.857
* TFIDF Featurization in GBDT (BASE-LEARNERS=275, DEPTH=10) gave best results with F1-score of 0.8708
