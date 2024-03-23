# amazon_products_sentiment_analysis
Sentiment analysis of amazon products and products recommendation

## Project Summary


## Table of Contents
- [ETL](#ETL)
  - [Data Source](#data-source)
  - [Data Sets](#data-sets)
  - [Data Cleaning](#data-cleaning)
  - [Data Loading](#data-loading)
  - [Data Segregation](#data-segregation)
  - [Data Processing](#data-processing)
  - [Data Prediction](#data-prediction)
- [Sentiment Analysis](#sentiment-analysis)
- [Statitical Analysis](#statitical-analysis)
- [Models Comparison](#models-comparison) 
- [Results and Conclusions](#results-and-conclusions)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [References](#references)

# ETL
## Data Source
In this project data is obtained from Home credit default risk at Kaggle.com 
## Data Sets

## Data Cleaning

## Data Loading


``` bash

```


``` bash

```
## Data Segregation

``` bash

```
## Data Processing


``` bash

```

``` bash


```

## Data Prediction  



```

```

# Sentiment Analysis  


# Statistical Analysis  

| Feature            | Result (values)   |
| :----------------- | :----------------: 
|        count       |   33332.000000    |
|         mean       |   0.364646        |  
|         std        |   0.297416        |
|        min         |  -1.000000        |
|       25%          |   0.150000        |
|        50%         |   0.350000        |
|        75%         |   0.562500        |
|         max        |   1.000000        |           

# Models Comparison  







## Results and Conclusions

  
## Dependencies

This project involves using various dependences listed below for data cleaning and predictions.

```bash
# import Denpendencies
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

```
## Future Work


## Acknowledgments


## Author

Muhammad Kashif 

## References
[1] [Amazon Products data]([(https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products?resource=download])
