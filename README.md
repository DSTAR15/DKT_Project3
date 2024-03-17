Code - 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/dharm/Downloads/archive (5)/heart.csv")

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

Here is a detailed explanation of the provided code for heart disease prediction using logistic regression, along with the theoretical concepts behind it:

1. Importing Libraries:
   - `pandas`: Used for data manipulation and analysis.
   - `train_test_split`: Used to split the dataset into training and testing sets.
   - `StandardScaler`: Used to standardize features by removing the mean and scaling to unit variance.
   - `LogisticRegression`: The logistic regression model from scikit-learn.
   - `accuracy_score` and `classification_report`: Used to evaluate the performance of the model.

2. Loading Dataset:
   - The dataset is loaded using `pd.read_csv()` function, assuming it's stored in a CSV file named "heart_disease_dataset.csv".

3. Data Preprocessing:
   - Features (X) are extracted from the dataset excluding the target variable.
   - Target variable (y) is extracted.
   - Data is split into training and testing sets using `train_test_split()` function.

4. Feature Scaling:
   - Standardization is performed on the features using `StandardScaler()`.
   - This step ensures that each feature contributes equally to the model fitting by scaling them to have mean 0 and variance 1.

5. Model Training:
   - A logistic regression model is initialized.
   - The model is trained using the training data (X_train_scaled, y_train) using the `fit()` method.

6. Making Predictions:
   - Predictions are made on the testing set using the `predict()` method of the trained model.

7. Model Evaluation:
   - Accuracy of the model is calculated using `accuracy_score()` by comparing the predicted labels (`y_pred`) with the actual labels (`y_test`).
   - The classification report is generated using `classification_report()` which includes precision, recall, F1-score, and support for each class.

Theory Explained as below:

- Logistic Regression:
  - Logistic regression is a binary classification algorithm used to model the probability of a binary outcome.
  - It's based on the logistic function (sigmoid function) that maps any real-valued number into a value between 0 and 1.
  - In this case, it predicts the probability that a given patient has heart disease based on their features.
  
- Data Preprocessing:
  - Data preprocessing involves cleaning, transforming, and organizing data before feeding it into the machine learning model.
  - Common preprocessing steps include handling missing values, encoding categorical variables, and feature scaling.
  
- Feature Engineering:
  - Feature engineering involves creating new features or transforming existing ones to improve model performance.
  - It can include tasks like creating interaction terms, polynomial features, or transforming variables.
  
- Handling Missing Values and Outliers:
  - Missing values and outliers can significantly affect model performance.
  - Advanced techniques like imputation (e.g., filling missing values with mean/median) and outlier detection/removal are employed to handle them.
  
- Improving Healthcare Outcomes:
  - Accurate prediction of heart disease using logistic regression can lead to early detection and timely intervention, improving healthcare outcomes.
  - By incorporating patient demographics, medical history, and lifestyle factors, the model provides valuable insights for healthcare providers to make informed decisions.

Overall, this code demonstrates the application of logistic regression for heart disease prediction, showcasing the importance of data preprocessing, model training, and evaluation in building effective machine learning models for healthcare applications.
<img width="836" alt="image" src="https://github.com/DSTAR15/DKT_Project3/assets/128448451/be590c11-d09d-4107-bbfa-be3e4a13d40b">
