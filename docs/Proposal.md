# Predicting Customer Churn for Telecom Companies
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang 
- Author: Sahith Todupunuri
- GitHub : https://github.com/sahithtodupunuri
- Linkedin : https://www.linkedin.com/in/sahith-todupunuri

## 2. Background
* The project aims to build a predictive model to identify customers who are most likely to churn in the near future.Understanding which customers are likely to churn can save a company significant amounts of money and allow them to take targeted actions to retain those customers.

* Research Questions :
   1. What features are most indicative of customer churn?
   2. How accurate can we predict customer churn?
   3. What recommendations can be made based on the model's outputs?


## 3. Data

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
   - Contract
   - MonthlyCharges

## 4. Exploratory Data Analysis
### 4.1 Understanding the Data
It is found that the column TotalCharges has few null values in it. So, I removed those rows without effecting the other data
### 4.2 Data Visualization
#### 4.2.1 Gender and Churn Distributions
 ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/ada94c06-1781-4513-a24f-b3b2951cb1e6)
#### 4.2.3 Churn Distribution w.r.t gender
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/7440c972-475e-4991-9d9b-6820992c2093)
#### 4.2.4 Customer Contract Distribution
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/a6b66257-ba4e-4313-b0c8-7278669bb00b)
#### 4.2.5 Customer Payment Method distribution w.r.t. Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/a5d202dd-c501-44ab-8c94-b5fcf3984334)
#### 4.2.6 Distribution of Monthly Charges by Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/bb4a5a10-e2bb-4ec8-b596-643393f7f7d7)
#### 4.2.7 Distribution of Total Charges by Churn
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/e327486a-1c61-4dac-bd0b-787249f29b5f)
#### 4.2.8 Heat Map
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/sahithtodupunuri/assets/114625950/f8271990-8599-4755-93c1-36d8c325d91e)

### Interpretations
-There is negligible difference in customer percentage/ count who changed the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.
-About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customers with One Year Contract and 3% with Two Year Contract
-Customers who chose Electronic Check as their payment method were the most likely to leave, while those who selected Credit Card automatic transfer or Bank Automatic Transfer, as well as Mailed Check, were more likely to stay with the service.
-Customers with higher Monthly Charges are also more likely to churn
## 5. Future Work
   -Splitting the data into training and testing sets
   -Standardizing numeric attributes
   -Divide the columns into 3 categories, one for standardisation, one for label encoding and one for one hot encoding
   -Model Evaluation

## 6. References

Customer Churn Prediction using Machine Learning: Main Approaches and Models
*[Customer Churn Prediction using Machine Learning: Main Approaches and Models](https://towardsdatascience.com/churn-prediction-770d6cb582a5)*. :link:
