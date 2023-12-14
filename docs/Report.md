# PREDICTING CUSTOMER CHURN FOR TELECOM COMPANIES
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang 
- Author: Sahith Todupunuri
- GitHub : https://github.com/sahithtodupunuri
- Linkedin : https://www.linkedin.com/in/sahith-todupunuri

## 2. BACKGROUND
* The project aims to build a predictive model to identify customers who are most likely to churn in the near future.Understanding which customers are likely to churn can save a company significant amounts of money and allow them to take targeted actions to retain those customers.

* Research Questions :
   1. What features are most indicative of customer churn?
   2. How accurate can we predict customer churn?
   3. What recommendations can be made based on the model's outputs?


## 3. DATA

Description : 

1. Data Source : *[Kaggle](https://www.kaggle.com/datasets/priyankanavgire/telecom-churn)*. :link:

2. Data Size : 7 MB

3. Data Shape
   > - Number of columns =  21
   > - Number of rows    = 7043

4. What does each row represent?(a patient, a school, a crime, etc.)
The rows in the given table represent individual customer records in a telecom dataset. Each row would contain data about a single customer's mobile usage, billing, and other account-related metrics

5. Data dictionary:
   
| Column Name      | Description                          | Data Type           | Potential Values     |
|------------------|--------------------------------------|---------------------|----------------------|
| customerID       | Unique ID for each customer          | str                 | "7590-VHVEG"         |
| gender           | Gender of the customer               | str                 | "Female"             |
| SeniorCitizen    | Whether the customer is a senior     | int                 | 0                    |
| Partner          | Whether the customer has a partner   | str                 | "Yes"                |
| Dependents       | Whether the customer has dependents  | str                 | "No"                 |
| tenure           | Tenure in months                     | int                 | 1                    |
| PhoneService     | Whether the customer has phone svc.  | str                 | "No"                 |
| MultipleLines    | Whether multiple lines are available | str                 | "No phone service"   |
| InternetService  | Type of internet service             | str                 | "DSL"                |
| OnlineSecurity   | Whether online security is enabled   | str                 | "No"                 |
| OnlineBackup     | Whether online backup is enabled     | str                 | "Yes"                |
| DeviceProtection | Whether device protection is enabled | str                 | "No"                 |
| TechSupport      | Whether tech support is enabled      | str                 | "No"                 |
| StreamingTV      | Whether TV streaming is enabled      | str                 | "No"                 |
| StreamingMovies  | Whether movie streaming is enabled   | str                 | "No"                 |
| Contract         | Contract type                        | str                 | "Month-to-month"     |
| PaperlessBilling | Whether paperless billing is enabled | str                 | "Yes"                |
| PaymentMethod    | Method of payment                    | str                 | "Electronic check"   |
| MonthlyCharges   | Monthly charges for the customer     | float               | 29.85                |
| TotalCharges     | Total charges till now               | float               | 29.85                |
| Churn            | Whether the customer churned         | str                 | "No"                 |



6. Which variable/column will be your target/label in your ML model?
    - Churn                 

7. Which variables/columns may selected as features/predictors for your ML models?
   - tenure
   - TotalCharges
   - MonthlyCharges

## 4. EXPLORATORY DATA ANALYSIS
### 4.1 Understanding the Data
It is found that the column TotalCharges has few null values in it. So, I removed those rows without effecting the other data
### 4.2 Data Visualization
#### 4.2.1 Gender and Churn Distributions
 ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/ada94c06-1781-4513-a24f-b3b2951cb1e6)
 - 26.6 % of customers switched to another firm. Customers are 49.5 % female and 50.5 % male.

#### 4.2.3 Churn Distribution w.r.t gender
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/7440c972-475e-4991-9d9b-6820992c2093)
 - There is negligible difference in customer percentage/ count who changed the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.

#### 4.2.4 Customer Contract Distribution
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/a6b66257-ba4e-4313-b0c8-7278669bb00b)
- About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customers with One Year Contract and 3% with Two Year Contract

#### 4.2.5 Customer Payment Method distribution w.r.t. Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/3e9bf5aa-ea21-4d08-8507-26f2ab22b77d)
- Customers who chose Electronic Check as their payment method were the most likely to leave, while those who selected Credit Card automatic transfer or Bank Automatic Transfer, as well as Mailed Check, were more likely to stay with the service.

#### 4.2.6 Distribution of Monthly Charges by Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/bb4a5a10-e2bb-4ec8-b596-643393f7f7d7)
- Customers with higher Monthly Charges are also more likely to churn

#### 4.2.7 Distribution of Total Charges by Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/e327486a-1c61-4dac-bd0b-787249f29b5f)

#### 4.2.8 Heat Map
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/f8271990-8599-4755-93c1-36d8c325d91e)

## 5. MODEL TRAINING
 - Standardizing numerical features to understand the data more and ensured they are scaled properly for modeling purposes
 - Splitting the data into 70 - 30 for training and testing.

### 5.1 KNN
 - Used Random Forest Classifier for my initial prediction.
 - Obtained an accuracy of 0.732 using KNN
   ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/273f0011-d14d-4808-89a7-feda7c1ef7a2)
 - The KNN model is effective at identifying non-churn cases (Class 0) with a high precision and recall, indicating reliable predictions for this group.
 - The model shows limited effectiveness in accurately identifying churn cases (Class 1), with low precision and recall, suggesting a need for improvement in this area

### 5.2 Logistic Regression
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/e54fa1cd-6e79-4716-ad64-ce7f09daf1e5)
 - Balanced Performance: The Logistic Regression model shows a balanced performance with an overall accuracy of 81.09%, demonstrating good precision and recall for both classes, especially with a notably better ability to predict churn cases compared to the previous KNN model.
 - Improved Churn Prediction: The model achieves a significant improvement in identifying churn cases (Class 1) with a precision of 0.67 and recall of 0.58, indicating enhanced effectiveness in predicting customer churn compared to the KNN model.


### 5.3 Gradient Boosting Classifier
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/90ba550e-ec71-4bf7-b976-0727385f49d8)
 - The classifier shows strong performance for the majority class (non-churn), with both high precision and recall, indicating a high likelihood of accurate predictions for customers who will not churn.
 - Reasonable Detection of Churn: The classifier's ability to identify churn cases is reasonable, with moderate precision and over half of the actual churn cases correctly identified, suggesting that while it can distinguish churn cases to an extent, there is still potential for improvement in recognizing these instances

### 5.4 Random Forest Classifier
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/cd30a458-fa93-47a4-bb82-57f9ed637f08)
 - The model demonstrates strong predictive power for non-churn cases (Class 0) with high precision (0.83) and recall (0.93), suggesting it is quite reliable in identifying customers who will not churn.
 - Moderate Effectiveness for Churn Prediction: While the accuracy for churn cases (Class 1) is moderate with precision (0.71) and recall (0.49), there's room for improvement in correctly classifying actual churn cases, as indicated by almost half of the churn cases being misclassified as non-churn in the confusion matrix.


### 5.5 Voting Classifier
 - It uses multiple algorithms to obtain better predictive performance than could be obtained from any of the algorithms alone
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/609ade4a-8da0-416a-8b56-99995ee0bb1a)
 - The classifier exhibits a strong performance for the majority class (Class 0) with high precision and recall, indicating accurate predictions for non-churn instances.
 - For the minority class (Class 1), the results show moderate precision
 - There are total 1402+147=1549 actual non-churn values and the algorithm predicts 1402 of them as non churn and 147 of them as churn. While there are 248+313=561 actual churn values and the algorithm predicts 252 of them as non churn values and 309 of them as churn values.

## 6. CONCLUSIONS
 - The Logistic Regression model achieved a higher accuracy compared to the other individual models, indicating its robustness in handling the dataset used.
 - Across all models, there is a consistently strong performance on the majority class (Class 0 - non-churn), as indicated by high precision and recall.
 - All models showed moderate success in identifying the minority class (Class 1 - churn), with room for improvement in both precision and recall for this group.
 - The Voting Classifier, which combines predictions from multiple models, did not significantly outperform the Logistic Regression model in terms of accuracy but provided a balanced approach to both classes.
 - Enhancing customer retention is crucial for a company's financial health. To mitigate customer churn, it's imperative for businesses to deeply understand their clientele, particularly pinpointing those at risk of leaving and striving to heighten their contentment. Prioritizing exceptional customer service is a key tactic in addressing churn. Fostering customer loyalty by providing personalized experiences and tailored services can also play a significant role in decreasing churn rates. Additionally, conducting surveys with customers who have churned can offer valuable insights, enabling firms to proactively refine their strategies to prevent future churn.
   
## 7. REFERENCES
 - *[Customer Churn Prediction using Machine Learning: Main Approaches and Models](https://towardsdatascience.com/churn-prediction-770d6cb582a5)*. :link:

 - https://machinelearningmastery.com/machine-learning-with-python/
  
 - https://www.w3resource.com/machine-learning/scikit-learn/iris/index.php
  
 - https://www.learndatasci.com/tutorials/intro-feature-engineering-machine-learning-python/
 
 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

