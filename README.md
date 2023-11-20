# NordeusChallenge
Submission for the Nordeus Data Science Challenge including the python notebook and the predictions for the league rank insinde 'league_rank_predictions.csv'.

# Nordeus Data Science Challenge

## Project Overview
This project is a submission for the Nordeus JobFair Data Science Challenge, dedicated to predicting league ranks for players playing Top Eleven. Using machine learning models, the aim to forecast the league position of each user at the end of a season. Presumably, the goal is to create a balanced and competitive experience for players.

## Environment
The project was realised inside Google Colab, utilising its resources, because of the limited computing perfornace of the local machine.

## Dataset
The project utilizes two datasets:
- `jobfair_train.csv` - Contains features like user activity, player statistics, and the target variable `league_rank`.
- `jobfair_test.csv` - Similar to the training dataset but without the target variable, for model prediction.

## Features
Features include user engagement metrics, player quality indicators, and other relevant game activity data.

## Machine Learning Models
We explore several models:
- RandomForestClassifier
- XGBoostClassifier
- LGBMClassifier
- DecisionTreeClassifier
- StackingClassifier (with Logistic Regression as the final estimator), however due to insufficient time, this models was abandoned

## Setup and Installation
Make sure Python is installed on your system. Dependencies include:
- pandas
- scikit-learn
- xgboost
- matplotlib

Install these using pip:
```bash
pip install pandas scikit-learn xgboost matplotlib

## Usage
To run the models and evaluate their performance, follow these steps:
1. Load the datasets `jobfair_train.csv` and `jobfair_test.csv`.
2. Preprocess the data as per the preprocessing steps outlined in the code.
3. Train the machine learning models using the preprocessed training data.
4. Evaluate the models using cross-validation techniques.
5. Use the trained models to make predictions on the preprocessed test data.
6. Analyze the results, and adjust the models or preprocessing steps as needed.

## Repository Structure
- `README.md`: This file, providing an overview and instructions.
- `NordeusChallenge.ipynb`: Contains code for the challenge.
- `league_rank_predictions.csv`: Includes all the predictions.

## License
This project is licensed under the [MIT License](LICENSE.txt).

