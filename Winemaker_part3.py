import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# Load the dataset
data_path = './vineyard_weather_1948-2017.csv'
vineyard_df = pd.read_csv(data_path)

# Convert 'DATE' to datetime and extract year and week
vineyard_df['DATE'] = pd.to_datetime(vineyard_df['DATE'])
vineyard_df['YEAR'] = vineyard_df['DATE'].dt.year
vineyard_df['WEEK'] = vineyard_df['DATE'].dt.isocalendar().week

# Filter data for weeks 35 to 40
filtered_df = vineyard_df[(vineyard_df['WEEK'] >= 35) & (vineyard_df['WEEK'] <= 40)]

# Aggregate data by year and week
aggregated_df = filtered_df.groupby(['YEAR', 'WEEK']).agg(
    total_precip=('PRCP', 'sum'),
    max_temp=('TMAX', 'max'),
    min_temp=('TMIN', 'min')
).reset_index()

# Create lagged features
aggregated_df['prev_week_precip'] = aggregated_df.groupby('YEAR')['total_precip'].shift(1)
aggregated_df['prev_week_max_temp'] = aggregated_df.groupby('YEAR')['max_temp'].shift(1)
aggregated_df['prev_week_min_temp'] = aggregated_df.groupby('YEAR')['min_temp'].shift(1)
aggregated_df['prev_2_week_precip'] = aggregated_df.groupby('YEAR')['total_precip'].shift(2)
aggregated_df['prev_2_week_max_temp'] = aggregated_df.groupby('YEAR')['max_temp'].shift(2)
aggregated_df['prev_2_week_min_temp'] = aggregated_df.groupby('YEAR')['min_temp'].shift(2)
aggregated_df['temp_range'] = aggregated_df['max_temp'] - aggregated_df['min_temp']

# Define the storm occurrence condition
aggregated_df['storm'] = ((aggregated_df['total_precip'] >= 0.35) & (aggregated_df['max_temp'] <= 80)).astype(int)

# Drop rows with missing values
aggregated_df.dropna(inplace=True)

# Split data into features and target
X = aggregated_df[['prev_week_precip', 'prev_week_max_temp', 'prev_week_min_temp',
                   'prev_2_week_precip', 'prev_2_week_max_temp', 'prev_2_week_min_temp',
                   'temp_range']]
y = aggregated_df['storm']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipelines
pipelines = {
    "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000))]),
    "Random Forest": Pipeline([('scaler', StandardScaler()), ('randomforestclassifier', RandomForestClassifier())]),
    "Gradient Boosting": Pipeline([('scaler', StandardScaler()), ('gradientboostingclassifier', GradientBoostingClassifier())]),
    "XGBoost": Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier())])
}

param_distributions = {
    "Random Forest": {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [None, 5],
        'randomforestclassifier__min_samples_split': [2, 5]
    },
    "Gradient Boosting": {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__max_depth': [3, 5],
        'gradientboostingclassifier__learning_rate': [0.1, 0.3]
    },
    "XGBoost": {
        'xgbclassifier__n_estimators': [100, 200],
        'xgbclassifier__max_depth': [3, 5],
        'xgbclassifier__learning_rate': [0.1, 0.3]
    }
}

# Train models with randomized search and store results
model_performance = {}
for name, pipeline in pipelines.items():
    if name in param_distributions:
        random_search = RandomizedSearchCV(pipeline, param_distributions[name], n_iter=8, cv=3, scoring='accuracy')
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
    else:
        best_model = pipeline.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    model_performance[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

# Train the best-performing model (e.g., XGBoost)
best_model_name = 'XGBoost'
best_model = pipelines[best_model_name].fit(X_train, y_train)

def evaluate_decision(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob):
    # Update probabilities based on user input
    updated_data = pd.DataFrame({
        'prev_week_precip': [aggregated_df['prev_week_precip'].mean()],
        'prev_week_max_temp': [aggregated_df['prev_week_max_temp'].mean()],
        'prev_week_min_temp': [aggregated_df['prev_week_min_temp'].mean()],
        'prev_2_week_precip': [aggregated_df['prev_2_week_precip'].mean()],
        'prev_2_week_max_temp': [aggregated_df['prev_2_week_max_temp'].mean()],
        'prev_2_week_min_temp': [aggregated_df['prev_2_week_min_temp'].mean()],
        'temp_range': [aggregated_df['temp_range'].mean()]
    })

    # Predict the probability of a storm occurrence
    storm_prob = best_model.predict_proba(updated_data)[0][1]

    # Calculate expected values for 'Harvest Now' and 'Wait'
    e_value_now = calculate_e_value_now(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob)
    e_value_wait = calculate_e_value_wait(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob, storm_prob)

    recommendation = 'Harvest Now' if e_value_now > e_value_wait else 'Wait'
    return e_value_now, e_value_wait, recommendation

def calculate_e_value_now(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob):
    # Define the payoff matrix
    payoff_matrix = {
        'Botrytis': -10000,
        'No Sugar Increase': 5000,
        'Typical Sugar Increase': 6000,
        'High Sugar Increase': 7000
    }

    # Calculate the expected value for harvesting now
    e_value_now = (botrytis_prob * payoff_matrix['Botrytis']) + \
                  (no_sugar_prob * payoff_matrix['No Sugar Increase']) + \
                  (typical_sugar_prob * payoff_matrix['Typical Sugar Increase']) + \
                  (high_sugar_prob * payoff_matrix['High Sugar Increase'])

    return e_value_now

def calculate_e_value_wait(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob, storm_prob):
    # Define the payoff matrix
    payoff_matrix = {
        'Botrytis': -10000,
        'No Sugar Increase': 5000,
        'Typical Sugar Increase': 6000,
        'High Sugar Increase': 7000,
        'Storm': -5000
    }

    # Calculate the expected value for waiting
    e_value_wait = (botrytis_prob * payoff_matrix['Botrytis']) + \
                   (no_sugar_prob * payoff_matrix['No Sugar Increase']) + \
                   (typical_sugar_prob * payoff_matrix['Typical Sugar Increase']) + \
                   (high_sugar_prob * payoff_matrix['High Sugar Increase']) + \
                   (storm_prob * payoff_matrix['Storm'])

    return e_value_wait

# Streamlit app
st.title("Decision Model for Vineyard Harvesting")
st.write("Adjust the sliders to see the recommended decision and its expected value.")

botrytis_prob = st.slider("Chance of Botrytis", 0.0, 1.0, 0.0)
no_sugar_prob = st.slider("Chance of No Sugar Increase", 0.0, 1.0, 0.0)
typical_sugar_prob = st.slider("Chance of Typical Sugar Increase", 0.0, 1.0, 0.0)
high_sugar_prob = st.slider("Chance of High Sugar Increase", 0.0, 1.0, 0.0)

if st.button("Evaluate"):
    e_value_now, e_value_wait, recommendation = evaluate_decision(botrytis_prob, no_sugar_prob, typical_sugar_prob, high_sugar_prob)
    st.write(f"E-Value Harvest Now: {e_value_now:.2f}")
    st.write(f"E-Value Wait: {e_value_wait:.2f}")
    st.write(f"Recommendation: {recommendation}")