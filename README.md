# **Model for California Housing Prices**

**Objective**
- I have built a Model using Random Forest Regressor of California Housing Prices Dataset to predict the price of the Houses in California.

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import sys, os, tarfile, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
```

**Getting the Data**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I will use California Housing Prices Dataset from the StatLib Repository for this Project. This Dataset was based on Data from the 1990 California Census. The Data has metrics such as Population, Median Income, Median House Price and so on for each block group in California. I will build a Model of Housing Prices in California using the California Census Dataset.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2010.PNG)

**Exploratory Data Analysis**
- The analysis report presents that the DataFrame has 20640 rows and each row represents one district. The report also presents that the DataFrame has 10 features where one is categorical and 9 are numerical. The info method is standard and useful to get the quick description of the Data, in particular the attributes type and the number of non null values.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2010b.PNG)

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2010a.PNG)

- The total bedrooms attribute has only 20433 non null values which means that 207 districts are missing here. The ocean proximity attribute is of object Data type and the values in this attribute is repetitive which means that it is probably categorical attribute. The value counts method can be used to find the categories and number of districts belonging to a particular category. The describe method shows the summary of the numerical attributes.

**Stratified Sampling**
- I have presented the Implementation of Stratified Sampling, Correlations using Scatter Matrix and Attribute combinations using Python here in the Snapshots:

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2011a.PNG)

**Visualizing the Geographical Data**
- Since, there is a Geographical Information latitude and longitude it will be a good idea to create the scatter plot of all districts to visualize the Data. I will use alpha option to 0.1 which makes it much easier to visualize the places where there is high density of data points. Now, let's look at the Housing prices. The radius of each circle represents the district's population option s, the color represents the price option c and I will use predefined color map option cmap called jet which ranges from blue low values and red high values.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2011b.PNG)

**Correlations**
- Now, I will compute the standard correlation coefficient betweeen every pair of the attributes using corr method. I will use Pandas scatter matrix function which plots every numerical attribute to other numerial attribute to check Correlations between attributes. I will focus only on those attributes who are strongly correlated to each other.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2011c.PNG)

**Preparing Data for Machine Learning Algorithms**
- Now, I will prepare the Data for Machine Learning Algorithms. Most of the Machine Learning Algorithms cannot work with missing values. The total bedrooms attribute has some missing values as obtained above in the analysis. Scikit Learn provides a handy class to take care of missing values: Simple Imputer. I will create the Imputer instance and replace the missing values with median of the attribute. Since the median can be calculated only on the numerical attributes, I will create the copy of Data without the text attribute ocean proximity. One issue with this representation is that ML Algorithms assume that the two nearby values are more similar than two distant values. So, I will apply the process of One Hot Encoding. Scikit Learn provides OneHotEncoder class to convert Categorical values into One Hot Vectors. The new attributes are called dummy attributes. 

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2012a.PNG)

**Training the Model**
- One of the most important transformations is to apply Feature Scaling to the Data. Machine Learning Algorithms don't perform well with the input numerical attributes having very different scales. The two common ways to get all attributes to have the same scale is Min max Scaling and Standardization. Scikit Learn provides Pipeline class to help with the sequence of Transformations. The Categorical columns and The Numerical columns are handled separately. It would be more convenient to have a single Transformer able to handle all columns, applying the appropriate transformations to each columns. Scikit Learn has ColumnTransformer class for this purpose. I will apply all the Transformations to the Housing Data using ColumnTransformer. 
  
  - **Decision Trees**: The score obtained using Linear Regression is not the best score. It is an example of Model underfitting the Training Data which means that the features don't provide enough information to make good predictions or that the Model is not powerful enough. The main ways to fix underfitting is to select more powerful Model or to feed the Training Algorithms with better features. Firstly, I will train the Model with Decision Trees which is more powerful Model. 
  
  - **Cross Validation**: I will use Scikit Learn's Cross Validation for Evaluating the Model. The result obtained is an array of a number of Evaluation scores. Scikit Learn's Cross Validation features expect a utility function greater is better rather than a cost function lower is better. So the scoring fucntion is actually the opposite of MSE which is a negative value. I will use minus sign before calculating the square root of scores.
  
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2012b.PNG)

**Random Forest Regressor**
- Random Forests work by training many Decision Trees on random subset of the features and then averaging out the predictions. Building a model on top of many other models is called Ensemble Learning and it is often a great way to push Machine Learning Algorithms even further. 

  - **Grid Search and Randomized Search**: I will use Scikit Learn's Grid Search to find the great combinations of hyperparameters values. It will evaluate all possible combinations of hyperparameters values using Cross Validation. The Grid Search Approach is fine when there is relatively few combinations. When the hyperparamters search space is large it is preferable to use Randomized Search. It evaluates the given number of random combinations by selecting a random value for each hyperparameter at every iteration.
  
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%2013.PNG)
  
