# Churn-Prediction-ANN
## Customer Churn Prediction using Artificial Neural Network

This project demonstrates how to use Python and Keras to build an artificial neural network (ANN) to predict customer churn.

## Dataset
The dataset used in this project is `Churn_Modelling.csv,` which contains information about bank customer churn.

### Columns:
- **RowNumber** (numerical): Index of the row.
- **CustomerId** (numerical): Unique identifier for the customer.
- **Surname** (categorical): Surname of the customer.
- **CreditScore** (numerical): Credit score of the customer.
- **Geography** (categorical): Country of the customer.
- **Gender** (categorical): Gender of the customer.
- **Age** (numerical): Age of the customer.
- **Tenure** (numerical): Number of years the customer has been with the bank.
- **Balance** (numerical): Account balance of the customer.
- **NumOfProducts** (numerical): Number of bank products the customer is using.
- **HasCrCard** (numerical): Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember** (numerical): Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary** (numerical): Estimated salary of the customer.
- **Exited** (numerical): Whether the customer left the bank (1 = Yes, 0 = No).

## Methodology
1. **Data Loading and Exploration:**
   - The dataset is loaded using pandas.
   - Initial exploration of the data to understand its structure and content.

2. **Data Preprocessing:**
   - Encoding categorical variables (Geography and Gender).
   - Concatenating encoded variables and dropping unnecessary columns.
   - Splitting the dataset into training and test sets.
   - Feature scaling using StandardScaler.

3. **Building the Neural Network:**
   - Initializing the ANN.
   - Adding input and hidden layers.
   - Adding the output layer.

4. **Model Training:**
   - Compiling the neural network.
   - Training the neural network on the training set.

5. **Evaluation:**
   - Plotting accuracy and loss during training and validation.
   - Making predictions on the test set.
   - Evaluating the model using the confusion matrix and accuracy score.
   - Visualizing the results using seaborn and matplotlib.

## Results
The model's accuracy and loss during training and validation are plotted. The confusion matrix and final accuracy score are displayed to evaluate the model's performance.

### Final Model Performance
- **Training Accuracy:** 86.07%
- **Validation Accuracy:** 84.74%
- **Test Accuracy:** 85.6%
- **Confusion Matrix:**
    - True Negatives: 1539
    - False Positives: 68
    - False Negatives: 220
    - True Positives: 173

## Conclusion
This project demonstrates the use of an artificial neural network to predict customer churn. The model is trained using the Keras library and evaluated using common metrics such as accuracy and confusion matrix.

## About Me
My Name is Sehaj Malhotra I'm a graduate student at Northeastern University, pursuing my master's in Data Analytics Engineering. I have a strong passion for Data Analytics, Data Visualization, and Machine Learning. I am an aspiring Data Analyst/Data Scientist. I love working with data to extract valuable insights.

MY LINKEDIN: https://www.linkedin.com/in/sehajmalhotra/

MY KAGGLE: https://www.kaggle.com/sehaj2001
