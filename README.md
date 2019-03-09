# Predicting Yelp business Star Ratings using Neural Nets

## Summary

This is part of a project for Artificial Intelligent Class at California State University, Sacramento

On Yelp, a business can have hundreds, or thousands of reviews, and it is difficult for users to read and analyze them all. As a result, the Star Rating is usually the most important factor for user to choose a business. As an attempt to solve this problem, I'm using a fully-connected Neural Network to predict star ratings from reviews.

## Implementation
1. **Dataset**: We will be using the dataset from Yelp: https://www.yelp.com/dataset/. 

2. **Data Preprocessing**: I convert the file from json to tsv, and parse the file into Python. I also include tips.json in the list of features, which is the short review about each business. I use sklearn's tf-idf Vectorizer to transform the reviews and tips into matrices of TF-IDF features. To avoid outliers, I only filter businesses that have more than 45 reviews_count.

3. **Model**: The model is a simple Dense, fully-connected Neural Network. The data is split into training and testing set. I also use EarlyStopping and ModelCheckPoint to save the best model and avoid overfitting. The main framework of this project is Tensorflow and Keras. I also compare the performance of the Neural Network Models with sklearn's models including: Linear Regression, Logistic Regression, SVM, KNN, and Multinomial Naive Bayes

## Result:
Our best model achieved a 73% accuracy. Preparing the data correctly played an important role in boosting the accuracy of our model. Further tuning is necessary in order to achieve better accuracy for the model
