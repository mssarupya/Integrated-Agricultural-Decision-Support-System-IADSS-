# 🌾 Integrated Agricultural Decision Support System (IADSS) - README 🌾

## Project Overview
📝 The Integrated Agricultural Decision Support System (IADSS) is designed to empower farmers with precise, location-specific insights for effective decision-making in agriculture. It integrates weather forecasting, crop prediction, and customized fertilizer recommendations to optimize farming practices.

## Data Files
📊 The project includes the following data files:
- crop_and_fertilizer_data.csv: This file contains information on crop types, soil conditions, and recommended fertilizers.
- weather_forecast.csv: This file provides historical and real-time weather data such as temperature, rainfall, humidity, and wind patterns.

## Data Fields
📋 The dataset contains the following data fields:

- District_Name: Name of the district where the crop is cultivated.
- Soil_color: Color of the soil in the cultivation area.
- Nitrogen, Phosphorus, Potassium: Levels of nutrients in the soil.
- pH: pH level of the soil.
- Rainfall, Temperature: Weather conditions at the time of cultivation.
- Crop: Type of crop cultivated.
- Fertilizer: Recommended fertilizer for the specific crop and soil conditions.
- Link: Link to additional information about the recommended fertilizer.

## Approach

### 1. Data Exploration and Preprocessing
🔍 Explore the data, handle missing values, and preprocess categorical features for model training.

python
# Load and preprocess the dataset
import pandas as pd

data = pd.read_csv('crop_and_fertilizer_data.csv')
# Data preprocessing steps...
2. Feature Selection and Engineering
🔍 Select relevant features and create new ones to improve model performance.

python
Copy code
# Feature selection and engineering...
3. Model Training and Comparisons
⚙️ Train and compare different models for predicting crop types and recommending fertilizers.

python
Copy code
# Model training and comparisons...

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
4. Model Evaluation
📊 Evaluate the trained models using suitable metrics to select the best-performing model.

# Model evaluation...

from sklearn.metrics import accuracy_score

# Model evaluation for Decision Tree Classifier
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Model evaluation for Random Forest Classifier
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
5. Making Predictions
🔮 Use the best-performing model to make predictions on new data.
predictions = rf_model.predict(new_data)
# Model Selection: Random Forest Classifier
## 🚀 The Random Forest Classifier is chosen over the Decision Tree Classifier for its ensemble learning approach, which improves prediction accuracy and handles overfitting.##







### Ensemble Learning: Random Forest combines multiple decision trees to enhance accuracy and generalizability.
Reduced Overfitting: By aggregating predictions from multiple trees, it mitigates overfitting compared to individual decision trees.
Feature Importance: It provides a feature importance score, aiding in understanding significant factors in crop prediction.
Usage

💻 To run the project, ensure the required libraries are installed:
pip install pandas scikit-learn
### Contact Details
📬 For inquiries or collaborations, reach out via:

Email: [mssarupya@gmail.com]
