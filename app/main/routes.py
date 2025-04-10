from flask import render_template, request, Blueprint, jsonify
from app.prediction_functions import route_accPrediction
import datetime as dt
import json
import traceback

main = Blueprint('main', __name__)

# home page
@main.route("/")
@main.route("/home")
def home():
    return render_template('index.html')


@main.route("/route_prediction")
def route_prediction():
    return render_template('route_prediction.html')

#API to get user inputs
@main.route('/prediction', methods=['POST'])
def prediction():
    try:
        req_data = request.get_json()
        origin = req_data['origin']
        destination = req_data['destination']
        date_time = req_data['datetime']
        print(date_time)
        #process time
        tm = dt.datetime.strptime(date_time,'%Y/%m/%d %H:%M').strftime('%Y-%m-%dT%H:%M')

        out = route_accPrediction(origin, destination, tm)

        return json.dumps(out)

    except:

        return jsonify({'trace': traceback.format_exc()})
