# Predicting Customer Churn for Telecom Companies
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang 
- Author: Sahith Todupunuri
- GitHub : https://github.com/sahithtodupunuri
- Linkedin : https://www.linkedin.com/in/sahith-todupunuri

## Background
* The project aims to build a predictive model to identify customers who are most likely to churn in the near future.Understanding which customers are likely to churn can save a company significant amounts of money and allow them to take targeted actions to retain those customers.

* Research Questions :
   1. What features are most indicative of customer churn?
   2. How accurate can we predict customer churn?
   3. What recommendations can be made based on the model's outputs?


## Data

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


## References

Customer Churn Prediction using Machine Learning: Main Approaches and Models
*[Customer Churn Prediction using Machine Learning: Main Approaches and Models](https://towardsdatascience.com/churn-prediction-770d6cb582a5)*. :link:
