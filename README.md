# Credit Default Prediction

This project aims to estimate predictions of credit default using various machine learning models such as logistic regression, LDA (Linear Discriminant Analysis), QDA (Quadratic Discriminant Analysis), decision trees, and KNN (K-Nearest Neighbors).

## Dataset

The dataset used for this project is named "Credit." It contains information related to credit applications, including various attributes or features associated with each application. The dataset can be found with the name "credit.xlsx" in the repository.

### Data Dictionary

1. checking_balance: The balance in the applicant's checking account.
2. months_loan_duration: The duration of the loan in months.
3. credit_history: The credit history of the applicant.
4. purpose: The purpose of the loan.
5. amount: The loan amount requested by the applicant.
6. savings_balance: The balance in the applicant's savings account.
7. employment_duration: The duration of employment.
8. percent_of_income: The percentage of income dedicated to paying off the loan.
9. years_at_residence: The number of years the applicant has been at their current residence.
10. age: The age of the applicant.
11. other_credit: Indicates whether the applicant has other existing credit.
12. housing: The housing the applicant resides in.
13. existing_loans_count: The number of existing loans.
14. job: The job or occupation of the applicant.
15. dependents: The number of dependents the applicant has.
16. phone: Indicates whether the applicant has a phone.
17. default: Indicates whether the applicant defaulted on their loan (1 if defaulted, 0 if not).

## Analysis

The project involves the following steps:

1. Data Preparation: Load the dataset and perform necessary preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features.

2. Model Training: Split the dataset into training and testing sets. Train multiple machine learning models for each of the following methods:

   - Logistic Regression
   - LDA (Linear Discriminant Analysis)
   - QDA (Quadratic Discriminant Analysis)
   - Decision Trees
   - KNN (K-Nearest Neighbors)

   Each model will be trained on the training set using the corresponding algorithm.

3. Model Evaluation: Evaluate the performance of each model on the testing set using appropriate evaluation metrics such as accuracy, precision, recall, and F1 score. Compare the results of different models to determine their effectiveness in predicting credit default.

4. For this specific analysis, Recall & Accuracy, Precision & Recall have been used separately to calculate the benefit of the model of all the 100 models that have been run. The Excel function picks up the best model based on the given inputs by the banker.

5. Benefit Calculation: Based on the models' predictions, calculate the benefit for the bank in terms of the number of clients and the default rate. The Payments and Average Default values are assumed to begin with. These values can be adjusted based on the inputs given by the banker.

## File Dictionary

Credit_Bankruptcy_Presentation: A small presentation assuming you are a service provider approaching a bank for a tie-up with them.

Credit_Bankruptcy_Report: A brief report of the complete project.

MasterFile_AllCalculation: An Excel file that allows you to enter variables based on the bank's situation and shows instant outputs and visuals. It also has the underlying calculations used for the final benefit of the model.

R_Code_AllModels: R file with all the models run for each method and the results of accuracy, recall, and precision for every model

credit: The original dataset used to run the models.



