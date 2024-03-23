# amazon_products_sentiment_analysis
Sentiment analysis of amazon products and products recommendation

## Project Summary


## Table of Contents
- [Extract](#extract)
  - [Data Source](#data-source)
  - [Data Sets](#data-sets)
- [Transform](#transform)
  - [Data Cleaning](#data-cleaning)
  - [Data Loading](#data-loading)
  - [Data Segregation](#data-segregation)
  - [Data Processing](#data-processing)
  - [Data Prediction](#data-prediction)
      - [Supervised Learning](#supervised-learning)
      - [UnSupervised Learning](#unsupervised-learning)   
  - [Models Comparison](#models-comparison)  
- [Load](#load)
  - [Data Export](#data-export)
- [Results and Conclusions](#results-and-conclusions)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [References](#references)

## Data Source
In this project data is obtained from Home credit default risk at Kaggle.com 
## Data Sets


# Transform

## Data Cleaning
The process of data cleaning is divided into three stages below as per techniquie followed to better understand segregate , process the numerial data and then make the predictions for categorical data and put the predicted values into missing fields.

### Data Loading
Data is loaded from csv files in this project using absolute path method. since Data files are big size which can not be accomodated in the github. so for sake of project they were kept in different folder to avoid large size file errors during commit stage. Code for loading data using absolute path is below.

``` bash
# Dictionary to hold file names and their paths
file_paths = {
    'application_train': '../Resources/application_train.csv',
    'bureau': '../Resources/bureau.csv',
    'bureau_balance': '../Resources/bureau_balance.csv',
    'credit_card_balance': '../Resources/credit_card_balance.csv',
    'POS_CASH_balance': '../Resources/POS_CASH_balance.csv',
    'previous_application': '../Resources/previous_application.csv',
    'installments_payments': '../Resources/installments_payments.csv'
}

# Dictionary to hold the loaded data
data_frames = {}

# Loop through the file_paths dictionary to load each file
for file_name, relative_path in file_paths.items():
    # Construct the absolute path
    file_path = os.path.abspath(relative_path)
    
    # Attempt to load the CSV file into a DataFrame
    try:
        data_frames[file_name] = pd.read_csv(file_path)
        print(f"{file_name} data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")

```
The data is present is dictionary  data_frames each data set can be accessed using code below. 

``` bash

```

### Data Segregation

``` bash

```

### Data Processing


``` bash

```

``` bash


```


### Data Prediction  
#### Supervised Learning  



``` bash

```


```

```

#### UnSupervised Learning  

``` bash

```
 
```


```

## Models Comparison


| Column Name        | Supervised Results | UnSupervised Results |
| :----------------- | :----------------: | :-------------------:|
| OCCUPATION_TYPE    | 96391              | 96391                |
| FONDKAPREMONT_MODE | 210295             | 210295               |
| HOUSETYPE_MODE     | 154297             | 154297               |
| WALLSMATERIAL_MODE | 156341             | 156341               |
| EMERGENCYSTATE_MODE| 145755             | 145755               |

Time taken tabular data:  

| Time                 | Supervised Method   | UnSupervised Method  |
| :-----------------   | :----------------:  | :-------------------:|
| Minutes & Seconds    | 50 miutes 26 seconds| 1 minute 2 seconds   |  




# Load

### Data Export


## Results and Conclusions

  
## Dependencies

This project involves using various dependences listed below for data cleaning and predictions.

```bash


```
## Future Work


## Acknowledgments


## Author

Muhammad Kashif 

## References
[1] [Amazon Products data]([(https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products?resource=download])
