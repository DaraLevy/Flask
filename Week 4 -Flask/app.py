from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

with open("wine_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print('Received input:', int_features)
    final_features = [np.array(int_features)]
    print('Final input:', final_features)
    prediction = model.predict(final_features)

    wine_names = ['Class 1', 'Class 2', 'Class 3']
    predicted_name = wine_names[int(prediction[0])]

    return render_template('index.html', prediction_text= 'Predicted Wine Species: {}'.format(predicted_name))

app.run(port=5000)