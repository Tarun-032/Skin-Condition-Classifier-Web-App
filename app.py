from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder
rf_model = joblib.load('dermatology_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Class descriptions
class_descriptions = {
    1: "Psoriasis is a chronic skin condition characterized by red, itchy, and scaly patches on the skin.",
    2: "Seborrheic dermatitis is a common skin condition that causes redness, scaly patches, and dandruff. It often affects the scalp but can also appear on other oily areas of the body.",
    3: "Lichen planus is an inflammatory skin condition that can cause an itchy rash, as well as white, lacy, raised areas on the skin.",
    4: "Pityriasis rosea is a benign skin rash that usually starts with a distinctive large, scaly, pink patch on the chest or back, followed by similar, smaller patches elsewhere on the body.",
    5: "Chronic dermatitis refers to long-term inflammation of the skin, which can result in redness, itching, and other skin changes.",
    6: "Pityriasis rubra pilaris is a rare skin disorder characterized by orange-red scaling plaques and keratotic follicular papules."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('Analyze.html')

@app.route('/facts')
def pcos():
    return render_template('facts.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        erythema = float(request.form['erythema'])
        scaling = float(request.form['scaling'])
        definite_borders = float(request.form['definiteBorders'])
        itching = float(request.form['Itching'])
        koebner_phenomenon = float(request.form['koebnerPhenomenon'])
        family_history = float(request.form['familyHistory'])

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'erythema': [erythema],
            'scaling': [scaling],
            'definite_borders': [definite_borders],
            'itching': [itching],
            'koebner_phenomenon': [koebner_phenomenon],
            'family_history': [family_history]
        })

        # Make predictions using the model
        prediction_code = rf_model.predict(user_input)[0]
        
        # Decode the predicted code to the corresponding class
        predicted_class = label_encoder.inverse_transform([prediction_code])[0]

        # Get the class description
        class_description = class_descriptions.get(prediction_code, "No description available.")

        # Pass the prediction and description to the result.html template
        return render_template('result.html', prediction=predicted_class, description=class_description)

    return render_template('analyze.html')

if __name__ == '__main__':
    app.run(debug=True)
