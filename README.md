## RecommenderSystem
A simple movie recommendation system with collaborative filter using python code
Recommendation or Recommender system is one of the most used application data science. The system employs statistical algorithm that help to predict using user rating, review or view etc.The system assume that, it is highly likely for users to have similar kind of review/rating for a set of entities. Netflix, Amazon, Facebook, YouTube etc. uses recommender system in one way or the other way to increase the customer base for their products.

In this project a simple recommendation system is developed using movie rating data from MovieLens. I have used the latest dataset "ml-latest". It has 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. Includes tag genome data with 14 million relevance scores across 1,100 tags. 

import all the packages required for the project

# Python Packages
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
