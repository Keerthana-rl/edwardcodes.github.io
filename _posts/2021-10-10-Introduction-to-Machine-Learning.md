---
title:  "Chapter-1 Introduction to Machine Learning"
permalink: /posts/mlbook-1/
excerpt: "This covers the chapter-1 from Hands-on Machine Learning with Scikit-learn and Tensorflow book"
last_modified_at: 2021-10-10T16:00:11-04:00
image: assets/images/love-to-learn.jpg
categories:
- tutorial
tags:
- book-notes
toc: true
toc_sticky: true
#classes: wide
---

## What is Machine Learning?

Machine Learning is the science (and art) of programming computers so they can learn from data.

## Why use Machine Learning?

In Traditional programming, it is difficult to maintain rules based on new patterns as we have to add new patterns in our system whenever arises.

In Machine Learning, it automatically learns the new patterns by comparing to the earlier data. The program is shorter, easier to maintain and most likely more accurate.

**Data Mining** - Discover patterns that are not immediately apparent in large amount of data by applying ML techniques.

In general, ML is great for,
- Problems for which the current solutions require a long list of rules
- Complex problems that cannot be solved using traditional approaches
- Fluctuating new environments
- Getting insights about complex problems and large amounts of data

## Type of ML systems

### Based on Human Supervision

- **Supervised** - training dataset includes desired solutions or labels

  - Classification - For Category prediction (Eg.,spam/ham)
  - Regression - For predicting Numeric values (Eg., Price of the house)

  Most used supervised algorithms - KNN, Linear Regression, Logistic Regression, SVM, Decision tree & RF, Neural Networks.

- **Unsupervised** - training data is unlabeled.

  - Clustering - Detecting similar groups without any help. Gives more information by subdividing each groups into smaller groups. Popular Clustering algorithms - K-Means, DBSCAN, Hierarchical Cluster Analysis
  
  - Anomaly detection and novelty detection - It helps to find unusal patterns or to remove outliers from datasets automatically and finds anomalies in new instances where *novelty detection* doesn't train with outliers in training datsets and it finds the anomalies based on training data. Popular Algorithms - One-class SVM, Isolation Forest.
  
  - Visualization and dimensionality reduction - *Visualization algorithm* converts the complex and unlabeled data into 2D or 3D representation and preserve the structure as much they can, where *Dimensionality Reduction* merges the correlated features into one and helps in preserving the data without losing information. Popular algorithms - PCA, Kernel PCA, Locally-Linear Embedding, t-SNE.

  - Association Rule learning - Dig into large amounts of data and discover interesting relations between features (Eg., People who buy this, can buy this also). Apriori and Eclat are the algorithms used.

- **Semi-Supervised** - Data with partially labeled data or solutions. Combination of unsupervised and supervised algorithms. Example - Photo hosting services where it identifies person in the images if we label persons in one photo and it identifies the same person in other photos.

- **Reinforcement Learning** - The model, called an agent, observe the environment itself, select and perform actions, get *rewards* in return. The model learns the best strategy itself over the time and gets the most reward. Example: DeepMind's AlphaGo

### Based on the capacity to learn incrementally

- **Batch Learning** - The system cannot learn incrementally, must be trained using all the available data. To avoid using lot of resources and time, it's trained offline and deployed in production. It applies whatever it has learned. To train the new data, we have to merge new and old data, and train again from scratch for the whole dataset.

- **Online Learning** - the system is trained incrementally by feeding it data instances
sequentially, either individually or by small groups called *mini-batches*. Great for systems that receive data as continuous flow and have limited computing resources. Used to train on huge datasets that cannot fit in machine's memory. *Learning rate* is an important parameter, which helps in adapting to changing data. The biggest challenge in online learning is about intake data quality. Bad data would decrease the performance of the existing model and show wrong predictions to clients if it's running in live system. Monitoring the input data and removing outliers or unwanted data are the solutions.

### Based on Generalizing Capability

- **Instance-based Learning** - the system learns the examples by heart, then generalizes to new cases by comparing them to the learned examples (or a subset of them), using a similarity measure.

- **Model-based Learning** - the model is trained with set of examples and then use that model to make predictions. After model is trained, the model applied on a performance measure to evaluate how well the model is working. If model performance is good, apply it to make predictions on new cases which is called as *inference*.

## Main Challenges of ML

- **Insufficient Quantity of Training Data** - requires a lot of data to train even a simple problem.

- **Nonrepresentative Training Data** - choosing the sampling data to represent our model.

- **Poor Quality Data**  - Poor data (Outliers, missing data) leads to poor model.

- **Irrelevant Features** - Features not relevant to our problem. This can be avoided by adding relevant features, selecting important features and combining weak features into one

- **Overfitting Training Data** - the model performs well on the training data, but it does not generalize well. This can be avoided by,
  - reducing the number of attributes in the training data
  - gather more training data
  - reduce the noise in the training data

  Constraining a model to make it simpler and reduce the risk of overfitting is called *regularization*. It helps to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.

- **Underfitting the training data** - opposite of overfitting and it occurs when your
model is too simple to learn the underlying structure of the data. The way to fix the issue is,
  - more parameters
  - better features to the learning algorithm
  - reducing the regularization hyper‐parameter in the model

## Testing and Validating

- **Generalization Error** - The error rate on new cases, which evaluates the model on test set and tells how well our model will perform on instances it has never seen before.

- **Holdout Validation** - hold out part of the training set (validation set) to evaluate several candidate models and select the best one. The goal is to train multiple models with various hyperparameters on the reduced training set, and select the model that performs best on the validation set. After this, train the best model in full training set (train + validation) and evaluate this final model on the test set to get an estimate of the generalization error.

- **Cross-validation** - Each model is evaluated once per validation set, after it is trained on the rest of the data. By averaging out all the evaluations of a model, we get a much more accurate measure of its performance. But the disadvantage is, the training time is multiplied by the number of validation sets.

- **Data Mismatch** - hold out some of the traning set in a new set called *train-dev set*, allowing to evaluate if the model is performing bad because overfitting on training data or due to a mismatch between training and production data.
