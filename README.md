# **Project-Convex-Optimisation**

## **A. Logistic Regression**

Logistic regression models the probabilities for classification problems with two possible
outcomes. It is an extension of the linear regression model for classification problems.
We can use this model to classify each wine brand into the category “good wine” or “bad
wine”, a wine being considered “good” if its quality is superior or equal to 7/10 and “bad” in
the other case.

To solve a linear problem one cannot use a move function. One must use a cost function called
Cross-Entropy. The cross-entropy loss can be divided into two distinct cost functions.

## **B.Gradient Descent**

The ultimate target of gradient descent is to find the optimal parameters that lead to the
optimization of a particular function. In our study, we have a dataset that deals with a logistic
regression problem. Gradient descent is an iterative optimization algorithm to find the
minimum of a function.

However, there are different ways to implement this algorithm: Standard gradient descent
algorithm, Stochastic gradient descent, Gradient descent in mini-batch and Momentum
Gradient Descent.

Standard gradient descent algorithm is an iteration sequence that starts from any point. From
this current position, you search all around the point for the direction of where the slope goes
down the most. Once you have found this direction, you follow it for a distance and then
repeat the operation from step 1.
In this report we will show you several implementations of the gradient descent algorithm
and how this algorithm helps us to find the best wine according to these chemical
characteristics.

# **The Dataset**

## **A.Description**

We chose a dataset we found on Kaggle about red wine quality.
The dataset is composed of 12 columns and 1599 rows.
Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output)
variables are available (e.g., there is no data about grape types, wine brand, wine selling price,
etc.).

Each column represents a physicochemical variable (or a sensory variable for the last one)
used to determine the quality of wine, that is to say if it is “poor”, “normal” or “excellent”.
They are named as below:
1. fixed acidity: most acids involved with wine or fixed or non-volatile (do not
evaporate readily).
2. volatile acidity: the amount of acetic acid in wine, which at too high of levels can
lead to an unpleasant, vinegar taste.
3. citric acid: found in small quantities, citric acid can add 'freshness' and flavour to
wines.
4. residual sugar: the amount of sugar remaining after fermentation stops, it's rare
to find wines with less than 1 gram/litre and wines with greater than 45 grams/litre
are considered sweet.
5. chlorides: the amount of salt in the wine
6. free sulfur dioxide: the free form of SO2 exists in equilibrium between molecular
SO2 (as a dissolved gas) and bisulphite ion; it prevents microbial growth and the
oxidation of wine.
7. total sulfur dioxide: amount of free and bound forms of S02; in low concentrations,
SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm,
SO2 becomes evident in the nose and taste of wine.
8. density: the density of water is close to that of water depending on the percent
alcohol and sugar content.
9. pH: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14
(very basic); most wines are between 3-4 on the pH scale.
10. sulphates: a wine additive which can contribute to sulfur dioxide gas (S02) levels,
which acts as an antimicrobial and antioxidant.
11. alcohol: the percent alcohol content of the wine.
12. quality: output variable (based on sensory data, score between 0 and 10)
According to the information found on Kaggle, most of the wines composing the dataset
are “normal” and only a minority is considered “poor” or “excellent”.

## **B.Objective**

Our goal, with this dataset, is to be able, with the help of machine learning to determine
which physiochemical properties make a wine “Good”. 

# ** Python Implementation**  

## **A.Data Cleaning**

First, we check if there are any Null values in our data set. Then we arbitrary decide that if the quality is equal or above 7, the wine is “good”, if not it is
considered as “bad”. After that, we do the train set and test set. Keeping in mind that you must not evaluate the
performances of a model on the training data. If the training data is the same that the test
ones, the machine will already have the answers.

It is important to test the machine on data it has never seen. By doing so we will have an idea
on it's performance on other data. The test set is composed of 20% of the data and the 80%
left compose the train set.

Now comes the normalisation step. here we decided to use the RobustScaler regression. We use this type of regression because
it is way more optimised. This way we obtain data that is easier to use. The RobustScaler
allowed us to make a normalisation of our data without deforming it (because classic
regression can distort our data because of the outliers).

## **B.Confusion matrix**

A confusion matrix is a tool which allows us to measure a Machine Learning model’s
performance by verifying the frequency of correct predictions compared to reality in
classification issues.

We have chosen to match this matrix to estimate the dataset at first sight. It allows us to have
a first distribution that measures the quality of a classification system. For each training model we will bring up this wonderful tool to evaluate the preferences of
the models so we will have a concrete idea of which model works best for our dataset. 

# ** Results and Optimisation** 

For our prediction, we used 5 different models:
• Logistic Regression,
• Decision Tree,
• Random Forest,
• Support-vector machine (SVM),
• KNeighbors

We can see from the different learning models that two stand out and have the best matrix
confusion. I am talking about the RamdonForest model and the Kneighbors model. They
both have scores of 93% and 91% respectively. Even thought, RandomForest seems to have
a better accuracy, we have chosen to optimize the performance of the KNN algorithm.
Nevertheless, it was pointed out by Lin and Jeon in 2002 that RandomForest and the
Kneighbors algorithm can both be viewed as so-called weighted neighborhoods schemes.
Before we focus on our optimization, it is necessary to review the basics of the KNN
algorithm. 

The KNN algorithm selects the number K of neighbours and calculates the distance between 
the unclassified points. then it takes the K closest neighbours according to the calculated distance.

Then, we optimise our model, using the cross-validation which consist in training than
validating the models on different cut-outs of the train set. This is an estimation method of
the reliability of a model based on a sampling technique. We cut the date in K parts (this is
called folds) roughly equal. Each part K is used in a turn as the test data set. The rest is to be
used as training. The we do the average of the scores obtained and we choose the ones
which are the closest to the average. This is how we defined the best parameters. Here we
choose to split the data in 5 and we will focus on the KNeighborsclassifier.

