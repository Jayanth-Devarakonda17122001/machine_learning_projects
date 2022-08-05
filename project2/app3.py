import pickle

import numpy as np
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('insurance.html')


@app.route('/predict', methods=['POST', 'GET'])
def results():
    age = float(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = request.form['smoker']

    # gender_type
    if sex == 'male':
        sex_male = 1
        sex_female = 0
    elif sex == 'female':
        sex_male = 0
        sex_female = 1
    else:
        sex_male = 0
        sex_female = 0

    # smoker_status

    if smoker == 'yes':
        smoker_yes = 1
        smoker_no = 0
    elif smoker == 'no':
        smoker_yes = 0
        smoker_no =1
    else:
        smoker_yes = 0
        smoker_no = 0

    x = np.array([[age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes]])

    sc = pickle.load(open('sc.pkl','rb'))

    x_std = sc.transform(x)

    insurance_model = pickle.load(open('insurance_charges_predictor.pkl', 'rb'))

    y_prediction = insurance_model.predict(x_std)

    return jsonify({'Model_prediction': float(y_prediction)})


if __name__ == '__main__':
    app.run(debug = True, port = 2020)

