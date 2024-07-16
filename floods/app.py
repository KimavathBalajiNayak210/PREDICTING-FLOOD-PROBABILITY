import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load the trained model
xgb_model = joblib.load('xgboost_model.pkl')

# Load the data
Flood_Prediction = pd.read_csv('flood.csv')

# Define independent and dependent variables
X = Flood_Prediction[['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
                      'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
                      'Siltation', 'AgriculturalPractices', 'Encroachments',
                      'IneffectiveDisasterPreparedness', 'DrainageSystems',
                      'CoastalVulnerability', 'Landslides', 'Watersheds',
                      'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                      'InadequatePlanning', 'PoliticalFactors']]
Y = Flood_Prediction['FloodProbability']

# Predict flood probabilities on the full dataset
predictions = xgb_model.predict(X)
Flood_Prediction['PredictedFloodProbability(%)'] = predictions

# Streamlit app
st.title("Flood Prediction Model")

# Evaluation metrics
test_mse = mean_squared_error(Y, predictions)
test_mae = mean_absolute_error(Y, predictions)
test_r2 = r2_score(Y, predictions)

st.header("Model Evaluation Metrics")
st.write(f'Mean Squared Error (MSE): {test_mse:.4f}')
st.write(f'Mean Absolute Error (MAE): {test_mae:.4f}')
st.write(f'R² Score (Accuracy): {test_r2:.4f}')



# User input for new predictions
st.header("Input New Data for Prediction")

new_input = {}
for feature in X.columns:
    new_input[feature] = st.selectbox(f'Select value for {feature}', range(16))

# Convert the input to DataFrame
new_input_df = pd.DataFrame([new_input])

# Predict flood probability for the input data
new_prediction = xgb_model.predict(new_input_df)
st.write(f'Predicted Flood Probability: {new_prediction[0]:.2f}%')


