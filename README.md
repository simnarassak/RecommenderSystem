# RecommenderSystem
A simple movie recommendation system with collaborative filter using python code
Recommendation or Recommender system is one of the most used application data science. The system employs statistical algorithm that help to predict using user rating, review or view etc.The system assume that, it is highly likely for users to have similar kind of review/rating for a set of entities. Netflix, Amazon, Facebook, YouTube etc. uses recommender system in one way or the other way to increase their customer base or the products.

In this project a simple recommendation system is developed using movie rating data from MovieLens. I have used the latest dataset "ml-latest". It has 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. Includes tag genome data with 14 million relevance scores across 1,100 tags. 

import all the packages required for the project

### Python Packages
```python
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```
### Model Training

The are two major recommendor approaches-
1. Content based filtering

This method uses the similarity based on the attributes and products. 
In the case of movies content based filter uses genres, production house, directors, actors

2. Collaborative filtering

It is based on the users past behaviour and similar decision or choices made by other users

Singular Value Decomposition (SVD)

SVD is used as a collaborative filtering technique. It is a method from linear algebra that has been generally used as a dimensionality reduction technique in machine learning. SVD is a matrix factorisation technique.


```python
#Read the ratings and movie data into data frame
ratings=pd.read_csv("/Users/ml-latest/ratings.csv")
movies=pd.read_csv("/Users/ml-latest/movies.csv")
del ratings['timestamp'] #Removing timestamp as it is not used in this project at this moment
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algorithm1=SVD()

```

Cross-validation is primarily used in machine learning to estimate model performance. It gives a less biased or less optimistic estimate of the model performance. In this project only top 100k data is used for cross validation

```python

cross_validate(algorithm1,d,measures=['RMSE','MAE'],cv=5,verbose=True)

```

25% the data is used as test and remaining as the training set to generate a model

```python
traindata,testdata=train_test_split(dataset,test_size=0.25)
```
The model is developed using train data. Then a prediction test is done on the 25% test data. The test results shows that the model has an accuracy of 79.8%. 

```python
algorithm1.fit(traindata)
predict=algorithm1.test(testdata)
accuracy.rmse(predict)

```
### Recommendation
Now let use this model for some real recommendation.

First I give an option to select user by entering user id. Then the id is used to review the user's previous movie rating patterns. From the previous history of the user, a new set of movies are recommended. An estimated score from the model using customer behaviour is used to give the recommendation.

```python
#Select a user by giving a user Id
Idu=input("Enter the user Id:")
user_Id=int(Idu)
```
Now we will look into historical data for identifying the user behaviour 

```python
#user previous views
user=ratings[(ratings['userId']==user_Id)&(ratings['rating']<=4)]
user=user.set_index('movieId')
user=user.join(movies)['title']
print(user)

```
Now the model knows a pattern of the given user's move review, thus it can give a recommendation for the user

```python

#Recommendation for user
user=movies.copy()
user=user.reset_index()
user['Estimate Score']=user['movieId'].apply(lambda x:algorithm1.predict(user_Id,x).est)
user=user.drop('movieId',axis=1)
user=user.sort_values('Estimate Score',ascending=False)
print(user.head(5))

```
