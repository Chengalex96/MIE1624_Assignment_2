# MIE1624_Assignment_2

**Objective:** To train, validate, and tune multi-class ordinary classification models that can classify, given a set of survey responses by a data scientist, what a survey respondent’s current yearly compensation bucket is.

**Cleaning the Data:**

**Objective: **To learn how to clean/preprocess the dataset

First cleaned the dataset using the KaggleSalary_Dataset.ipynb:

Dropped rows with missing target variable (Q24)- Salaries

Combined the salary buckets and label encoded them (Give the ranges of 0-999, 1000-1999... etc to 7500-9999 a value of 1 to group them)

This created two columns: Q24_encoded and Q24_buckets

**#Label Encoding the target variable**

Salaries.loc[1:,'Q24_Encoded'] = Salaries.loc[1:,'Q24'].map(salary_encode)

Salaries.loc[1:,'Q24_Encoded']=Salaries.loc[1:,'Q24_Encoded'].astype(int)

Salaries.Q24_Encoded.unique()

**#Combining the salary buckets**

Salaries.loc[1:,'Q24_buckets'] = Salaries.loc[1:,'Q24'].map(salary_buckets)

Salaries.Q24_buckets.unique()

**Definitions:**

**Classification** is a supervised machine learning approach used to assign a discrete value of one variable when given the values of others. Many types of machine learning models can be used for training classification problems, such as logistic regression, decision trees, kNN, SVM, random forest, gradient-boosted decision trees, and neural networks. In this assignment, I will use the ordinal logistic regression algorithm.

**Part 1: Data Cleaning**

Install libraries such as numpy, pandas, sklearn StandardScaler, LogisticRegression, test_train_split, metrics, matplotlib, seaborn

Convert the csv file to pandas dataframe, dropped the question row and the salary column since we only want categorical data (Q24)

Exclude columns with parts, then count number of null, must contain at least 5000 training examples
- Q1 - Age range (# years)
- Q2 - Gender
- Q3 - Residing country
- Q4 - Highest lv of education
- Q5 - Current job / position
- Q6 - Years in programming - range
- Q8 - Which programming language to learn first?
- Q11 - What computing platform do you use?
- Q13 - How many times have you used a TSU
- Q15 - How many years have you used ML methods?
- Q20 - Size of company
- Q21 - How many individuals deal with Data science at workplace
- Q22 - Does current employer use ML
- Q25 - How much money has your company spent on ML
- Q30 - Which of the following big data products do you use?
- Q32 - Which of the following business intelligence tools do you use?
- Q38 - Which is the primary tool you use to analyze data?

Using get dummies, we will later separate each unique value into a column, just like the columns with extra parts

df[['Q1', 'Q2','Q3','Q4','Q5','Q6','Q8','Q11','Q13','Q15','Q20','Q21','Q22','Q25','Q30','Q32','Q38']].isnull().sum(axis=0)

By listing out the values of each category, we can see how to handle each missing value for the column

Value_counts returns the proportions of responses: df['Q38'].value_counts(normalize=True)

This heatmap shows visually the number of null values in the dataset, red values represent null values:

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/064b83eb-d077-45db-86b0-d68b96af5658)

**Data Cleaning so it can be used with Scikit model:**

Substitute null values with the most common occurrences:
- For numbers 8, 11, and 13 we see that one value makes up the majority of the column's values, fill nans with the mode
- df['Q8'].fillna(df['Q8'].mode()[0],inplace=True)
- df['Q11'].fillna(df['Q11'].mode()[0],inplace=True)
- df['Q13'].fillna(df['Q13'].mode()[0],inplace=True)
- df['Q15'].fillna(df['Q15'].mode()[0],inplace=True)

Drop null values - only 1% of values:
- df = df.dropna(subset=['Q25'])

Drop columns - these values contain too many NAN values
- df = df.drop(columns = ['Q30', 'Q32', 'Q38'])

Any other null values are set to 0, and are dummy-encoded:
- df = df.fillna(0)
- df = pd.get_dummies(data=df, columns=['Q1', 'Q2','Q3','Q4','Q5','Q6','Q8','Q11','Q13','Q15','Q20','Q21','Q22','Q25'])

Dropped the time column and converted all data to 1's and 0's as a string, then int
- Then place the salary target buckets at the end
- Wanted to convert the categorical data into numerical data using dummy encoding making many extra columns but much easier to wokr with

The new heatmap shows no null values:

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/aa219fc1-71ea-4313-82ba-72f35d272e5f)

**Exploratory Data Analysis:**

Want to visualize the order of feature importance, we calculate the highest positive correlation of the features for Q24_encoded (Higher salary):

df_feature_selection['Q24_Encoded'].sort_values(ascending=False)[1:101].head(10)

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/9a220896-b396-4613-8743-211f5970b78e)

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/a1d9a96f-85bb-404f-9290-b52d9b34d7e6)

Highest negative correlation (Does not make a higher salary):

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/01429a97-b689-46bc-9dac-540451ae6b99)

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/a9b2f385-60c1-4d5a-a33d-7e651e3b8321)

I took the top 100 features with the highest positive and negative features and combined them: this reduced the number of features from 489 to 200 (60%):

df_clean_reduction = df_clean[positive_corr].join(df_clean[negative_corr]).join(df_clean['Q24_Encoded'])

This is called feature engineering: Feature engineering is a useful tool in machine learning because it is important to which features are used to train the model, if we use the wrong features, the model will rely on the wrong features to make its predictions. Dummy encoding is a feature engineering technique used to separate the unique values into different columns which makes it easier to process. How we handle missing values is also a feature engineering technique that may affect our model.

**Model Implementation**

Split the data between training and test data (70/30 split)

In ordinal logistic regression, there are 15 salary buckets (ranging from 0 - 14 as int values), so would need 14 binary classifications:
- First binary classification: any values > 0 are 1, rest is 0
- Second binary classification: any values > 1 is 1, rest is 0
- This creates 14 extra columns that indicate which rows belong to that salary bucket indicated by 1

Performing Ordinal Logistic Regression using 10-Fold Cross Validation:
- We want to find the best model coefficient for each bucket
- X is the top 200 feature data that is dummy encoded with 1's and 0's
- Y is the bucket column
- The model used is the LogisticRegression()
- For loop to go through each bucket, for loop to go through kfold.split(X) in which data is split between training and test
- Model is fitted for each kfold split, the F1 score is calculated using the predictions, and the average prediction is returned for each bucket
- The best logistic coefficient and intercept are stored for each binary classification

Results:
- The average accuracy gets better after each fold and the variance decreases after each fold
- The F1 score tends to decrease after each fold
- F1 is more accurate, for accuracy, if the data is skewed to low salaries and if the model says all feature provides low salaries, it is correct
- For example, if the model says all features will provide a low salary (dominant class), it is more likely than not will provide true positive and true negative results used in the accuracy calculation. This may be also the bias-variance trade-off, as we move to higher buckets, it seems to be overfitting the training sample to the lower salary target values. We will see if it’s overfitting if it provides a good score for the training data but not the test data. Scaling/ normalization of the features isn’t necessary because they are encoded as 1’s and 0’s using dummy encoding.

**Model Tuning:**

Hyperparameter - Grid Searching

C is a hyper parameter, that shows the inverse of regularization strength to be applied:
- coefficient of the cost function for the regularized logistic regression
- Values were picked such that the model converges  for C in [0.1,0.5,1,5,10]:

The type of solver used is also a hyperparameter, some methods will provide better results than others.
- These methods have different norms used for calculations
- Limited-memory Broyden–Fletcher–Goldfarb–Shanno - may not converge to anything
- Stochastic Average Gradient, Sags is impractical for large N (because it remembers the most recently computed values for approximately all gradients). - Long computation times
- Liblinear is used in 1 vs rest scenarios which is not useful here since this is a multi-class (Used to speed up computation time during coding session)
- for solver in ['newton-cg','liblinear']: # Use lbfgs which uses a L2 norm penalty

model = LogisticRegression(C=C, solver=solver, max_iter=500)

# Grid search for the best model, select the model with the best F1 score

          if np.mean(F1) > best_F1:
              best_model = model
              best_params = {'C':C, 'solver':solver}
              best_F1 = np.mean(F1)
              best_std = np.std(F1)

Prediction of random training data belonging to each salary bucket:

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/8375e256-e9c1-4cc8-9541-ba4bccc225fe)

Training Set Accuracy Score: 0.809
Training Set F1 Score: 0.221

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/a979a6ed-6e3b-40fe-ad4e-fb5644cf65b2)

Test Set Accuracy Score: 0.792
Test Set F1 Score: 0.142

![image](https://github.com/Chengalex96/MIE1624_Assignment_2/assets/81919159/2e3e6a16-6aaf-41ee-8ca7-537ad1ff116a)

Using the F1 score not only tells us which predictions we have right (accuracy), but it penalizes us against the ones we also got wrong (precision and recall). If out F1 score is close to 1, this means that our prediction matches our target value, if our F1 score is close to 0, this means that our model isn’t working very well.

For all 14 binary classifications, the best model coefficient and intercept is stored and used to solve for the probability (which inverse of regularization value and which solver provides the best F1 score). The probability of each survey participant belonging to each salary bracket is calculated using the logistic function. Since the brackets are cumulative, the bracket’s probability is subtracted from each other to get that specific salary range. The sum of the probability for each bracket should be equal to 1, the maximum probability for a bracket is then used to classify which bracket the survey participant belongs to. In

When we apply our optimal model to the test data, we obtain an accuracy of 79% and an F1 score of 0.16. Comparing that to our training data accuracy of 81.4% and F1 score of 0.243 we are slightly overfitting since it works better with training data. One method to increase the accuracy of the test and training set is to use different feature selection methods. By using other feature selection techniques such as PCA rather than selecting the features manually, we may be able to choose features that may be more useful in determining higher salaries. Other ways to improve the accuracy is to use different feature engineering methods since dummy encoding sets all the values to 0 and 1 and since the data is skewed, the F1_score will not be as good. Another method is to assign weights to certain features since some features have many more possible choices, and the choices will be spread thin among many different columns. There are multiple ways to improve the accuracy of our model, other methods are to use different regression algorithms, choose different hyperparameters to tune, etc. The distribution of the true target variable values and their predictions on both the training set and test set are plotted below.

Our model predicts that the salary range of $0-$9999 is the most frequent. Our model for both training and test data underestimates the number of participants in the other salary ranges. By trying to balance the data set with those who have a higher salary, more features can be used and extracted to determine the key features. There also seems to be a bump in the $100,00 – 124,999 range which may be the most common salary range for those in machine learning and data analytics, $0-9,999 may be the dominant range because people are uncomfortable sharing their salary in a survey.
