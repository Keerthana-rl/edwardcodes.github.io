---
title:  "Chapter-2 End-to-End Machine Learning"
permalink: /posts/mlchapter2/
excerpt: "This covers the chapter-2 from Hands-on Machine Learning with Scikit-learn and Tensorflow book. It discusses from framing the problem to deploying the model in production"
last_modified_at: 2021-10-15T16:00:11-04:00
header:
  image: assets/images/books.jpg
  teaser: assets/images/books.jpg
categories:
- tutorial
tags:
- book-notes
toc: true
toc_sticky: true
#classes: wide
---
> Dataset - California Housing Prices.

## Look at the big picture

### Frame the problem

Before jumping into the data, understand the business objective. The model selection, performance measure and everything depends on how we frame the problem. Whether it's a Supervised (Classification/ Regression) or Unsupervised learning or we should use batch or online training. Try to answer above questions before moving on.**Univariate Regression** - predicts single value

### Selecting performance measure

Performance measures give an idea of how much error the system typically makes in its predictions, with a higher weight of large errors. `Root Mean Square Error(RMSE)` is the preferred performance measure for regerssion tasks. If the datasets are with Outliers, `Mean Average Error(MAE)` is used due to distance measures.

RMSE follows `Euclidean Norm` or `l2 norm` and  MAE follows `Manhattan Norm` or `l1 norm` where it measures distance between two points Eg., distance a taxi has to drive in a rectangular street grid to get from the origin to the point x.

**which to use** - The RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

### Check Assumptions

Clarify the assumptions of data before working on. Whether the data has to be treated as classification or regression task. If we are working on *Loan Prediction Project*, predicting the exact numbers (Loan value) is not important, converting them into categories (Accepted/Rejected) matters. This can be clarified in earlier stage of project, instead of spending time on getting exact loan values.

## Get the Data

### Create the Workspace

Create an Isloated Environment - recommended if we work on different projects without conflicting each other

### Download the Data

**Code:**

```Python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path) # create directory if already not exists
tgz_path = os.path.join(housing_path, "housing.tgz")
urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()
```

### Loading the Data

Below function returns a Pandas DataFrame object containing all the data.

**Code:**

```Python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

### Take a Quick Look at the Data Structure

Below pandas keywords help to view and understand our data structure.

`head()` - display top 5 rows of dataframe
`info()` - quick description of the data
`describe()` - shows a summary of the numerical attributes
`hist()` - plot a histogram for the given attribute

### Create a Test set

#### Manual Method

**Data Snooping Bias** - to estimate the generalization error using the test set

**Code:**

```Python
import numpy as np

def split_train_test(data, test_ratio):
    np.random.seed(42) # always generates the same shuffled indices
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

**Hash Identifier** ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset, even though the dataset is changed.

**Code:**

```Python
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

#### Scikit-learn Method

`train_test_split` - pretty much does the same thing like our manual method
`random_state` - parameter that allows you to set the random generator seed

**Code:**

```Python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

If the dataset is not large enough, there's a risk of getting *Sampling bias*. To avoid this, we must make **stratified sampling** where population is divided into homogeneous subgroups called *strata*, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population.

**Code:**

```Python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

## Discover and Visualize the Data to Gain Insights

### Looking for Correlations

The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
there is a *strong positive correlation* whereas if the coefficient is close to –1, it means that there is a *strong negative correlation*. Coefficients close to zero mean that there is no linear correlation.

`.corr()` `scatter_matrix` from pandas - both used to compute the standard correlation
coefficient between every pair of attributes.

**Code:**

```Python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

### Experimenting with Attribute Combinations

- Try out various attribute combination
- Transform the existing attributes into more meaningful features - creating new features

## Prepare the Data for Machine Learning Algorithms

### Data Cleaning

Machine Learning algorithms cannot work with missing features. Use any one of the below options to handle missing values,

- Get rid of corresponding values
- Get rid of whole feature
- Set the values to some values (zero, mean, median, etc.)

**Using Pandas** - dropna() , drop() , and fillna()

```Python
dataframe.dropna(subset=["feature_name"]) # option 1
dataframe.drop("feature_name", axis=1) # option 2
median = dataframe["feature_name"].median() # option 3
dataframe["feature_name"].fillna(median, inplace=True)
```

**Using Sklearn** - SimpleImputer

```Python
from sklearn.impute import SimpleImputer
# replace each attribute’s missing values with the median of that attribute
imputer = SimpleImputer(strategy="median") 
```

#### Titbits of Scikit-learn design

- Consistency
  - Estimators - estimate some parameters based on a dataset - fit() mehod
  - Transformers - transform a dataset - transform() or fit_transform() method
  - Predictors - making predictions given a dataset - predict() method

### Handling Text and Categorical Attributes

Machine Learning algorithms prefer to work with numbers and we have to convert categories into numbers.

**Using Sklearn** - OrdinalEncoder() and OneHotEncoder()

If a categorical attribute has a large number of possible categories, then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance. To avoid that, we use `embeddings` which translate large sparse vectors into a lower-dimensional space that preserves semantic relationships.

### Feature Scaling

Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales. Scaling the target values is generally not required.

Popular feature scaling techniques are *min-max scaling / normalization* and *standardization*

**Min-Max Scaling** - values are shifted and rescaled so that they end up ranging from 0 to 1. Subtracting the minimum value from the value and dividing by the maximum minus the minimum. `MinMaxScaler` is used to do this in sklean library.

$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**Standardization** - Subtracts the mean value, and then it divides by the standard deviation so that the resulting distribution has unit variance. Standardization
does not bound values to a specific range, due to this nature, it can't be applied for Neural network as it expects input value ranging from 0 to 1. But it's less affected to Outliers.`StandardScaler` is used to do this in sklean library.

$x_{scaled} = \frac{x - x_{mean}}{std. deviation}$

### Transformation Pipelines

Data transformation steps that need to be executed in the right order. Scikit-Learn provides the `Pipeline` class to help with such sequences of transformations.

The `Pipeline` constructor takes a list of name/estimator pairs defining a sequence of
steps. All but the last estimator must be transformers. When the pipeline’s `fit()` method called, it calls `fit_transform()` sequentially on all transformers.

`ColumnTransformer` in sklearn has the capability to handle both numerical and categorical columns, and applies the appropriate transformations to each column.

## Select and Train a Model

### Training and Evaluating on the Training Set

**Linear Regression code:**

```Python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

Linear regression doesn't works well for this dataset, we can try `DecisionTreeRegressor` which is capable of finding complex nonlinear relationships in the data which gives results as *Zero Error* - maybe overfitted.

```Python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```

### Better Evaluation Using Cross-Validation

**K-fold cross-validation** - randomly splits the training set into *n folds*, then it
trains and evaluates the Decision Tree model *n times*, picking a different fold for
evaluation every time and training on the other *n-1 folds*. Cross-validation allows
you to get not only an estimate of the performance of your model, but also a measure
of how precise this estimate is (i.e., its standard deviation). But cross-validation comes at the cost of training the model several times, so it is not always possible.

**Code:**

```Python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```

**RandomForestRegressor** - training many Decision Trees on random subsets of the features, then averaging out their predictions. Building a model on top of many other models is called `Ensemble Learning`.

## Fine-Tune Your Model

### Grid Search

It will evaluate all the possible combinations of hyperparameter values using cross-validation with provided values of hyperparameters to experiment with.

**Code:**

```Python
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

### Randomized Search

It evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

**Benefits:**

- this approach will explore 'n' different values for n iterations for each hyperparameter rather than providing few values for hyperparameter in grid search
- More control over the computing budget you want to allocate to hyper‐
parameter search, simply by setting the number of iterations.

### Evaluate Your System on the Test Set

After tweaking the models which performs well, evaluate the final model on the test set. Get the predictors and the labels from our test set, run our full_pipeline to transform the data (call `transform()` , not `fit_transform()`, you do not want to fit the test set!), and evaluate the final model on the test set.

## Launch, Monitor, and Maintain Our System

- Write monitoring code to check our system’s live performance at regular intervals and trigger alerts when it drops. This is important to catch not only sudden breakage, but also performance degradation.
- Make sure we evaluate the system’s input data quality. Monitoring the inputs is particularly important for online learning systems.
- Train our models on a regular basis using fresh data and automate it. System’s performance may fluctuate severely over time if we don't refresh our model.
- If our system is an online learning system, we should make sure we save snapshots of its state at regular intervals, so we can easily roll back to a previously working state.
