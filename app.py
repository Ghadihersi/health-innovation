import numpy as np
from flask import Flask, request, render_template
import datetime
import pickle

app = Flask(__name__)
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('xgb_model.pkl', 'rb'))


def get_triage(triage):
    if triage == 2:
        return int(2)
    elif triage == 3:
        return int(3)
    elif triage == 4:
        return int(4)
    elif triage == 5:
        return int(5)
    else:
        return None


def get_hour(hour):
    morning = list(np.arange(0, 13))
    evening = list(np.arange(17, 24))
    if hour in morning:
        return int(7)
    elif hour in evening:
        return int(6)
    else:
        return None


def get_season(month):
    spring = list(np.arange(3, 7))
    summer = list(np.arange(7, 10))
    winter = list(np.arange(11, 3))
    if month in spring:
        return int(8)
    elif month in summer:
        return int(9)
    elif month in winter:
        return int(10)
    else:
        return None


def update_features(input_vals):
    """
    features = 'age', 'num_before_patient',
    'Triage Priority_2',Triage Priority_3','Triage Priority_4','Triage Priority_5','Arrival_period_evening',
    'Arrival_period_morning','Arrival_season_spring','Arrival_season_summer','Arrival_season_winter'
    :return:
    """
    now = datetime.datetime.now()
    features = np.zeros((1,11))

    to_sc = np.array([input_vals[0], input_vals[1]])
    inputs_sc = scaler.transform(to_sc.reshape(-1, 1))

    features[0,0] = inputs_sc[0][0]
    features[0,1] = inputs_sc[1][0]

    triage_value = get_triage(input_vals[2])
    if triage_value is not None:
        features[0,triage_value] = 1

    hour_value = get_hour(now.hour)
    if hour_value is not None:
        features[0,hour_value] = 1

    season_value = get_season(now.month)
    if season_value is not None:
        features[0,season_value] = 1

    return features


@app.route('/')
def home():
    """
    homepage layout
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    :return: rendered results on HDMI GUI
    """

    input_vals = [int(x) for x in request.form.values()] # contain age, num_patient_before, triage
    features = update_features(input_vals)

    prediction = model.predict(features)
    output = scaler.inverse_transform(prediction.reshape(-1, 1))
    output = int(np.expm1(output))

    current_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    return render_template('index.html',
                           prediction_text=
                           """ 
                           The current time is: {}
                           The estimated waiting time is  {} min
                           """.format(current_time,output))


if __name__ == "__main__":
    app.run(debug=True)
