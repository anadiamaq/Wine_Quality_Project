<h1 align="center">
  Wine Quality Project
</h1>

Wine-drinkers usually agree that wines may be ranked by quality, but it's known wine-tasting is mainly subjective. There are many attempts to construct a more methodical way to find out the wine's quality. In this project, I propose a method of assessing wine quality using a neural network and test it against the wine-quality dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/wine.gif" width="576" height="417" border="10"/>
</p>

## Table of content

1. [Wine Data :wine_glass:](##-wine-data-:wine_glass)
2. [Data Approach](##-data-approach)
    * [Conclusion](###-conclusion)
3. [Regression Algorithms](##-regression-algorithms)
    * [Random Forest Regressor](###-random-forest-regressor)
    * [KNeighbors Regressor](###-kneighbors-regressor)
    * [Epsilon-Support Vector Regression](###-epsilon-support-vector-regression)
    * [Conclusion](###-conclusion)
4. [Neural Network](##-neural-network)
    * [Preprocessing](###-preprocessing)
    * [The Model](###-the-model)
    * [Compile and fit](###-compile-and-fit)
5. [Api service](##-api-service)
6. [References](##-references)

## Folder Structure

├── LICENSE
├── ML_models.ipynb
├── Multiple_Regression.ipynb     
├── Neural_Network.ipynb
├── Neural_Network_redwine.ipynb  
├── Neural_Network_whitewine.ipynb
├── README.md
├── data
│   ├── redwine-qa.csv
│   ├── whitewine-qa.csv
│   ├── wine-qa.csv
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── datasets_cleaning.ipynb
├── images
│   ├── Mult_regression_redwine.png
│   ├── Mult_regression_whitewine.png
│   ├── NN_error.png
│   ├── NNmodel_rep.png
│   ├── data_dist.png
│   ├── epoch_loss.png
│   ├── epoch_mae.png
│   ├── error_table_ML.png
│   ├── st-wine.jpg
│   └── wine.gif
├── logs
├── requirements.txt
├── src
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── app.cpython-38.pyc
│   │   ├── config.cpython-38.pyc
│   │   ├── fetch.cpython-38.pyc
│   │   ├── main.cpython-38.pyc
│   │   ├── prep_function.cpython-38.pyc
│   │   └── server.cpython-38.pyc
│   ├── app.py
│   ├── config.py
│   ├── encoder.pkl
│   ├── fetch.py
│   ├── model.h5
│   ├── pickles_prep.py
│   ├── prep_function.py
│   ├── scaler.pkl
│   ├── server.py
│   └── weights.h5
└── streamlit-app
    ├── app.py
    └── config.py

## Wine Data :wine_glass:

As I said, the dataset is publicly available for research purposes on UCI website. Here, there are two datasets, the red wine dataset contains 1599 instances with 11 features and the white wine dataset, 4898 instances and the same 11 feature.
The inputs are made from objective tests as pH, density or sulphates values. By contrast, the output is based on sensory data made by at least 3 evaluations wine experts who graded the wine quality between 0 (very bad) and 10 (very excellent). The two datasets are related to red and white variants of the Portuguese [“Vinho Verde”](https://www.vinhoverde.pt/en/) wine.
In order to increase the number of instances in a single data set, both data sets are unified. Previously, a column with the type of wine (red or white) is added to the dataset. Later, for a better visualization of the data, these are represented in different histograms shown below: 

![Data distribution](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/data_dist.png)

## Data Approach

For evaluation purposes, an ordinary least squares (OLS) regression is performed. This linear regression models the relationship between a dependent variable and two or more independent variables. In this case, there are a total of 12 variables. 11 variables are assumed as predictors o independent variables and one (quality) as dependent variable. This model helps to know which variables are most related to quality (dependent variable).

In the following tables, several metrics are evaluated, but only the p-value (p <0.05) is taken into account when selecting or not the different variables or inputs that will be used in the regression algorithms and in the neural network.

In the same way, the metric R-squared is a statistical measure that represents how a dependent variable is explained by a independent varibale or a group of independent variables in a regression model, as in this case. Thereby, for red wine dataset, the quality variable in red wine is explained in 0.36 points and, in the white wine dataset, by 0.28 points.

Other important value in theses tables is the regression coefficient (coef). This statistical measure estimates the change in the mean response per unit increase in X (independet variables) when the rest of variables are held constant. For instance, if volatile acidity  increases by 1, and all other variables don't change, quality variable (dependent variable) decreases by about 1.86 on average. Also, if p-value is less than 0.05 for that variable, the relationship between the predictor and the response is statistically significant, which means that variable has a role in the wine quality.

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/Mult_regression_redwine.png" width="761" height="576" border="10"/>
</p>

###### OLS Regression Results for red wines


<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/Mult_regression_whitewine.png" width="761" height="576" border="10"/>
</p>

###### OLS Regression Results for white wines

### Conclusion

These results show that not all independent variables are releated to wine quality. What's more, the releated independent variables are different in each dataset. For red wine, volatile acidity, chlorides, free sulfur dioxide, total sulfur dioxide, pH, sulphates and alcohol are the independent variables related to wine quality. While, fixed acidity, volatile acidity, residual sugar, free sulfur dioxide, pH, sulphates and alcohol are the independent variables realated to wine quality in white wines.

## Regression Algorithms

Due to these results, when selecting the variables to create a ML model or another, only variables with a p-value lower than 0.05 were taken into account. The regression algorithms applied on the data set, on the one hand the red wine and on the other the white wine, are Random Forest Regressor (RFR), KNeighbors Regressor (KNR) and Epsilon-Support Vector Regression (SVR). 
In all cases, the data was standardize by removing the mean and scaling to unit variance with the `StandardScaler()` from sklearn library.
After each regression algorithm, a cross validation analysis was carried out to extract two different errors, the mean squared error (mse) and r2 score or R-squared (explained above). MSE is the most used evaluation criterion for regression problems, especially in supervised machine learning.

### Random Forest Regressor
The Random Forest Regressor is an Emsemble method that combines predictions from multiple ML algorithms which means a more accurate predictions than any individual model.

### KNeighbors Regressor
KNeighbors Regressor can be used when data labels are continuous rather than discrete variables, as in this case. This simple algorithm stores all available cases and predict the numerical target based on a similarity measure.

### Epsilon-Support Vector Regression
Although Support Vector Machines aren't well known in regression problems but, in this case, the results aren't too different from other algorithms.

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/error_table_ML.png" width="374" height="338" border="10"/>
</p>

### Conclusion

Just looking at these results, it could be determined that for both the red wine and white wine datasets, the best regression algorithm is Random Forest Regressor. In both cases, mse has the lowest value, while r2 has the value closest to 1.

## Neural Network 

### Preprocessing

The final method to predict wine's quality is a Supervised Neural Network. Firstly, I tried two different NN's, one for the red wine dataset and another for the white one. But, how statistical metrics seem to be worse (how we can see in the table below), I decided to create just one model with both datasets together.

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/NN_error.png" width="312" height="83" border="10"/>
</p>

To start, a preprocessing of the data is necessary. In this way, a `LabelEncoder()` function was executed on the variable "type", created prior to the union of the two datasets. After that, all the data were standardized with the `StandardScaler()` function. This function of the Sklearn library standardizes the data by eliminating the mean and scaling the data so that its variance is equal to 1. Otherwise, the data would be too disparate for the neural network to train.

### The Model

```
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(12,),kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='relu'))
```

The code above is the Neural Network Model used to compile and fit the model which predicts the wine quality. It is a sequential type model. This kind of model is appropriate when layers go one after the other as in this case. Firstly, I added a `Dense` layer with 12 nodes, that correspond to the 12 inputs (the 12 independent variables). Usually, the number of nodes that makes up the input layer must be equal to the number of features (columns) in the data. Nevertheless, we can add one additional node for a bias term, but not in this model.

The output layer has a single node because this NN is a regressor and I just expect a value, the quality. The number of nodes in the hidden layer is the mean between the nodes in the input layer and in the output layer.

Returning to the use of the Dense layer, I decided use it in all the layers due to the lack of data. Beside I united two diffenrent datasets, the number of values are not enough to get good results. Dense layers have the quality of tightly joining their nodes with the nodes of the previous layer. The following image is a representation of how nodes are connected in this model.

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/NNmodel_rep.png" width="836" height="765" border="10"/>
</p>

Another characteristic of this model is the presence of regularizers in the input layer. Regularizers apply constraints to the parameters of the layer or activity during their optimization and add it to the loss function.

Finally, this model has in all its layers with ReLU-type activation functions. The rectified linear activation function or ReLU is a function that will output the input directly if it is positive, otherwise, it will output zero. This makes the model easier to train and often achieves better performance.

### Compile and fit

The `compile()` method takes a list of metrics to evaluate the model during de `fit()` phase. For this model, the loss function, which purpose is "to compute the quantity that a model should seek to minimize during training", is measured with mean squares error. The optimizer implements the Adam algorithm which is a stochastic gradient descent method and, the last argument is the metrics. In this case, I use the mean absolute error that quantifies the precision of a prediction technique.

```
model.compile(loss= 'mse',
              optimizer= 'adam',
              metrics=['mae'])
```
After fitting and using the `TensorBoard()` function as a callbacks argument inside `fit()` function, several graphs are obtained as a summary of the training that we see below:

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/epoch_loss.png" width="1274" height="510" border="10"/>
</p>

<p align="center">
<img src="https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/epoch_mae.png" width="1274" height="510" border="10"/>
</p>

Finally, to save the model weights, I use the `save_weights()` method. In this way, I can use it to deploy the model later in Flask.

## Api service

Before to run the API, it is necesary to run the `pickes_prep.py` file. This file creates two pickles files that are useful to transform the data that the user will enter, so that they are readable by the model.  

To view the app it is only necessary to run the following files:

- `python3 server.py`
- `streamlit run app.py`

In a python environment with the previous installation of the packages found in the requirements.txt file.

## References

1. [Comp.ai.neural-nets FAQ.](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html) 
2. Er, Y., & Atasoy, A. (2016). The classification of white wine and red wine according to their physicochemical qualities. International Journal of Intelligent Systems and Applications in Engineering, 23-26. Available in: [https://www.ijisae.org/IJISAE/article/view/914](https://www.ijisae.org/IJISAE/article/view/914)
3. Gupta, Y. (2018). Selection of important features and predicting wine quality using machine learning techniques. Procedia Computer Science, 125, 305-312. Available in: [https://www.sciencedirect.com/science/article/pii/S1877050917328053](https://www.sciencedirect.com/science/article/pii/S1877050917328053)
4. [Keras API reference](https://keras.io/api/)
5. Lee, S., Park, J., & Kang, K. (2015, September). Assessing wine quality using a decision tree. In 2015 IEEE International Symposium on Systems Engineering (ISSE) (pp. 176-178). IEEE. Available in: [https://ieeexplore.ieee.org/abstract/document/7302752](https://ieeexplore.ieee.org/abstract/document/7302752)
6. P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. Available in: [https://archive.ics.uci.edu/ml/datasets/wine+quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
7. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. Available in: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
