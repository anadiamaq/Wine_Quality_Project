# Wine Quality Project

Wine-drinkers usually agree that wines may be ranked by quality, but it's known wine-tasting is mainly subjective. There are many attempts to construct a more methodical way to find out the wine's quality. In this project, I propose a method of assessing wine quality using a neural network and test it against the wine-quality dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

![wine-gif](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/wine.gif)

## Wine Data :wine_glass:

As I said, the dataset is publicly available for research purposes on UCI website. Here, there are two datasets, the red wine dataset contains 1599 instances with 11 features and the white wine dataset, 4898 instances and the same 11 feature.
The inputs are made from objective tests as pH, density or sulphates values. By contrast, the output is based on sensory data made by at least 3 evaluations wine experts who graded the wine quality between 0 (very bad) and 10 (very excellent). The two datasets are related to red and white variants of the Portuguese [“Vinho Verde”](https://www.vinhoverde.pt/en/) wine.
In order to increase the number of instances in a single data set, both data sets are unified. Previously, a column with the type of wine (red or white) is added to the dataset. Later, for a better visualization of the data, these are represented in different histograms shown below: 

![Data distribution](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/data_dist.png)

## Data Approach

For evaluation purposes, an ordinary least squares (OLS) regression is performed. This linear regression models the relationship between a dependent variable and two or more independent variables. In this case, there are a total of 12 variables. 11 variables are assumed as predictors o independent variables and one (quality) as dependent variable. This model helps to know which variables are most related to quality (dependent variable).

In the following tables, several metrics are evaluated, but only the p-value (p <0.05) is taken into account when selecting or not the different variables or inputs that will be used in the regression algorithms and in the neural network.

![OLS Regression Results for red wines](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/Mult_regression_redwine.png)
The multiple regression for red wine variables shows that volatile acidity, chlorides, free sulfur dioxide, total sulfur dioxide, pH, sulphates and alcohol are de independent variables related to wine quality in red wines.

![OLS Regression Results for white wines](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/Mult_regression_whitewine.png)
The multiple regression for white wine variables shows that fixed acidity, volatile acidity, residual sugar, free sulfur dioxide, pH, sulphates and alcohol are de independent variables realated to wine quality in white wines.

## Regression Algorithms

Due to these results, when selecting the variables to create a ML model or another, only variables with a p-value lower than 0.05 were taken into account. The regression algorithms applied on the data set, on the one hand the red wine and on the other the white wine, are Random Forest Regressor (RFR), KNeighbors Regressor (KNR) and Epsilon-Support Vector Regression (SVR). 
In all cases, the data was standardize by removing the mean and scaling to unit variance with the `StandardScaler()` from sklearn.
After each regression algorithm, a cross validation analysis was carried out to extract two different errors, the mean squared error and r2 score or coefficient of determination.

### Random Forest Regressor
The Random Forest Regressor is a Emsemble method that combines predictions from multiple ML algorithms which means a more accurate predictions than any individual model.

### KNeighbors Regressor
KNeighbors Regressor can be used when data labels are continuous rather than discrete variables, as in this case. This simple algorithm stores all available cases and predict the numerical target based on a similarity measure.

### Epsilon-Support Vector Regression
Although Support Vector Machines aren't well known in regression problems but, in this case, the results aren't too different from other algorithms.

![Error table](https://github.com/anadiamaq/Wine_Quality_Project/blob/develop/images/error_table_ML.png)

Just looking at these results, it could be determined that for both the red wine and white wine datasets, the best regression algorithm is Random Forest Regressor.

## References

1. Er, Y., & Atasoy, A. (2016). The classification of white wine and red wine according to their physicochemical qualities. International Journal of Intelligent Systems and Applications in Engineering, 23-26.
2. Gupta, Y. (2018). Selection of important features and predicting wine quality using machine learning techniques. Procedia Computer Science, 125, 305-312.
3. Lee, S., Park, J., & Kang, K. (2015, September). Assessing wine quality using a decision tree. In 2015 IEEE International Symposium on Systems Engineering (ISSE) (pp. 176-178). IEEE.
4. P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
