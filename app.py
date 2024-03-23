# from flask import Flask, render_template, request
# import pickle
# import pandas as pd
# from TimeModule import Weather
# from sklearn.preprocessing import OneHotEncoder

# app = Flask(__name__)

# # Load the saved model and encoder
# model_filename = 'crop_prediction_model.pkl'
# encoder_filename = 'label_encoder.pkl'

# with open(model_filename, 'rb') as model_file:
#     model_crop = pickle.load(model_file)

# with open(encoder_filename, 'rb') as encoder_file:
#     encoder = pickle.load(encoder_file)

# # Load the dataset
# dataset_filename = 'crop and fertilizer(csv) - Sheet1.csv'  # Update with the actual path
# dataset = pd.read_csv(dataset_filename)

# # Render the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Handle the prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the input values from the form
#         district = request.form['district']
#         soil_color = request.form['soil_color']
#         nitrogen = float(request.form['nitrogen'])
#         phosphorus = float(request.form['phosphorus'])
#         potassium = float(request.form['potassium'])
#         pH = float(request.form['pH'])
#         rainfall = float(request.form['rainfall'])
#         temperature = float(request.form['temperature'])

#         # Get user input for city and state
#         city = request.form['city']
#         state = request.form['state']

#         # Create an instance of the Weather class
#         bttf = Weather(city_name=city, state_name=state)
#         bttf.api_caller()

#         # Perform one-hot encoding
#         input_data = pd.DataFrame(
#             [[district, soil_color, nitrogen, phosphorus, potassium, pH, rainfall, temperature]],
#             columns=['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
#         )

#         X_encoded = encoder.transform(input_data[['District_Name', 'Soil_color']])

#         # Make predictions
#         predicted_crop = model_crop.predict(X_encoded)

#         # Find the fertilizer associated with the recommended crop
#         recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]
#         link = dataset[(dataset['Crop'] == predicted_crop[0]) & (dataset['Fertilizer'] == recommended_fertilizer)]['Link'].values[0]

#         return render_template('result.html', crop=predicted_crop[0], fertilizer=recommended_fertilizer, link=link)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model and label encoders
model_filename = 'crop_prediction_model.pkl'
label_encoder_district_filename = 'label_encoder_district.pkl'
label_encoder_soil_color_filename = 'label_encoder_soil_color.pkl'

try:
    with open(model_filename, 'rb') as model_file:
        model_crop = pickle.load(model_file)

    with open(label_encoder_district_filename, 'rb') as encoder_file:
        label_encoder_district = pickle.load(encoder_file)

    with open(label_encoder_soil_color_filename, 'rb') as encoder_file:
        label_encoder_soil_color = pickle.load(encoder_file)

except FileNotFoundError:
    print("Model and/or encoders not found. Please make sure to train the model first.")

# Load the dataset
dataset_filename = 'crop and fertilizer(csv) - Sheet1.csv'  # Update with the actual path
dataset = pd.read_csv(dataset_filename)
# Extract unique values for District and Soil_color
district_values = dataset['District_Name'].unique()

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html', district_options=district_values)

# Define route for AJAX request to get soil colors based on selected district
@app.route('/get_soil_colors/<district>')
def get_soil_colors(district):
    soil_colors = dataset[dataset['District_Name'] == district]['Soil_color'].unique()
    return jsonify(soil_colors.tolist())

# Define route for AJAX request to get options for nitrogen, phosphorus, potassium, pH, rainfall, and temperature
@app.route('/get_options/<district>/<soil_color>')
def get_options(district, soil_color):
    options = {
        'nitrogen': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Nitrogen'].unique().tolist(),
        'phosphorus': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Phosphorus'].unique().tolist(),
        'potassium': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Potassium'].unique().tolist(),
        'ph': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['pH'].unique().tolist(),
        'rainfall': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Rainfall'].unique().tolist(),
        'temperature': dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Temperature'].unique().tolist()
    }
    return jsonify(options)

# Handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        district = request.form['district']
        soil_color = request.form['soil_color']
        nitrogen = request.form['nitrogen']
        phosphorus = request.form['phosphorus']
        potassium = request.form['potassium']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        temperature = request.form['temperature']

        # Convert original values to encoded values
        encoded_district = label_encoder_district.transform([district])[0]
        encoded_soil_color = label_encoder_soil_color.transform([soil_color])[0]

        input_data = pd.DataFrame(
            [[encoded_district, encoded_soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature]],
            columns=['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
        )

        # Make predictions
        predicted_crop_encoded = model_crop.predict(input_data)

        # Convert the predicted encoded values back to original values
        original_predicted_district = label_encoder_district.inverse_transform([encoded_district])[0]
        original_predicted_soil_color = label_encoder_soil_color.inverse_transform([encoded_soil_color])[0]

        # Find the fertilizer associated with the recommended crop
        recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop_encoded[0]]['Fertilizer'].values[0]
        link = dataset[
            (dataset['Crop'] == predicted_crop_encoded[0]) & (dataset['Fertilizer'] == recommended_fertilizer)
        ]['Link'].values[0]

        # Render the result page
        return render_template(
            'result.html',
            crop=predicted_crop_encoded[0],
            fertilizer=recommended_fertilizer,
            link=link,
           
        )

if __name__ == '__main__':
    app.run(debug=True)
