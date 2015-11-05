# Kaggle Springleaf Marketing Response Challenge
Code for 75th place (out of 2,225) in Kaggle Springleaf Marketing Response Challenge (private leaderboard).
* Link: https://www.kaggle.com/c/springleaf-marketing-response

# Feature engineering (not all are used in final ensemble)
* Derived the following from date columns
  - Year
  - Month
  - Day
  - Quarter
  - Day of Year
  - Week of Year
  - Day of Week
  - Week of Month
  - Hour of day (for one variable)
* One-Hot Encoding for variables likely to be categorical variables

# Models
* XGBoost

# Software
* Python 2.7
* Python libraries:
  - Pandas
  - Xgboost
  - numpy
