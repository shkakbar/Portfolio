# Starbucks Capstone project

## Project Overview

Customer satisfaction drives business success and data analytics provides insight into what customers think. Each person has some hidden traits that make the people act differently to offers while a few converts to sales. Since marketing is a cost, hence business likes to study how each segment of customer responds to different offer and how they can target/tailor the offer to the individual to improve RoI by converting a potential customer to smiling customer.
The Starbucks Udacity Data Scientist Nanodegree Capstone challenge data set is a simulation of customer behavior on the Starbucks rewards mobile application. Periodically, Starbucks sends offers to users that may be an advertisement, discount, or buy one get on free (BOGO). An important characteristic regarding this dataset is that not all users receive the same offer.
This data set contains three files. The first file describes the characteristics of each offer, including its duration and the amount a customer needs to spend to complete it (difficulty). The second file contains customer demographic data including their age, gender, income, and when they created an account on the Starbucks rewards mobile application. The third file describes customer purchases and when they received, viewed, and completed an offer. An offer is only successful when a customer both views an offer and meets or exceeds its difficulty within the offer's duration.


------------------------------------------------------------------------------------------------------------------
## Problem Statement / Metrics

The problem that I chose to solve is to build a model that predicts whether a customer will respond to an offer. My strategy for solving this problem has following steps. 
First, clean and combine offer portfolio, customer profile, and transaction data. Each row of this combined dataset will describe an offer's attributes, customer demographic data, and whether the offer was successful. 
Second, apply a different machine learning model and assess different accuracy and F1-score. The 'naive model' assumes all offers were successful. This provides me a baseline for evaluating the performance of models that I construct. Accuracy measures how well a model correctly predicts whether an offer is successful. However, if the percentage of successful or unsuccessful offers is very low, accuracy is not a good measure of model performance. For this situation, evaluating a model's precision and recall provides better insight into its performance. Hence, I chose the F1-score metric because it is "a weighted average of the precision and recall metrics". 
Third, compare the performance of logistic regression, random forest, and gradient boosting models. 
And finally, refine the parameters of the model that has the highest accuracy and F1-score.

## Installation
In order to be able to execute your own python statements it should be noted that scripts are only tested on anaconda distribution 4.5.11 in combination with python 3.7 or above. The scripts don't require additional python libraries.
Python Libraries used are as follows>
Progressbar
joblib 
Pandas
Numpy
Matplotlib
Seaborn
sklearn (train_test_split, RandomizedSearchCV, GridSearchCV, MinMaxScaler, accuracy_score, f1_score, fbeta_score, make_scorer, LogisticRegression, GradientBoostingClassifier, RandomForestClassifier)

## Motivation
This project is done as part Udacity Data Scientist Nanodegree program - Starbucks Capstone challenge, this is a final project submission of my certification.

## Analysis 
Starbucks, periodically send the offers to users that may be an advertisement, discount or buy one get one free offer this data set is a simulation of customer behavior on the Starbucks rewards mobile application. An important characteristic of this dataset is that not all users receive the same offer. Hence, identifying the right individuals for targeted marketing who will respond positively to an offer
Machine Learning Model used as part of the project 
Logistic Regression Model
random forest
gradient boosting
naive predictor
This model will be compared to the highest training data accuracy and F1-score. Bias and variance are two characteristics of a machine learning model.
Bias refers to inherent model assumptions regarding the decision boundary between different classes.
On the other hand, variance refers to a model’s sensitivity to changes in its inputs.
Both random forest and gradient boosting models are a combination of multiple decision trees. A random forest classifier randomly samples the training data with replacement to construct a set of decision trees that are combined using majority voting. In contrast, gradient boosting iteratively constructs a set of decision trees with the goal of reducing the number of misclassified training data samples from the previous iteration. A consequence of these model construction strategies is that the depth of decision trees generated during random forest model training is typically greater than gradient boosting weak learner depth to minimize model variance. Typically, gradient boosting performs better than a random forest classifier. However, gradient boosting may overfit the training data and requires additional effort to tune. A random forest classifier is less prone to overfitting because it constructs decision trees from random training data samples. Also, a random forest classifier’s hyperparameters are easier to optimize.
Random Forest model was the best model on training dataset hence best fit parameters were refined further to test on 'test dataset', the performance of the model accuracy and F1-score improved as compared with training set and overfitting signs were observed.


## File descriptions
Within the download you'll find the following directories and files.
Portfolio / Capstone Starbuck/ ├── Starbucks_Capstone_notebook.ipynb

## Creator

Akbarali Shaikh

https://github.com/shkakbar

https://medium.com/@sk_akbar/starbucks-capstone-challenge-predicting-marketing-offer-success-9e69ae06dc86

Thanks to Starbucks and Udacity 
