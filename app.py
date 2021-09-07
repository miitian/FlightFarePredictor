from flask import Flask, request, render_template
from flask_cors import cross_origin
import numpy as np
import pandas as pd
import sklearn
import pickle
import featureEngineering as fe

app = Flask(__name__)
model = pickle.load(open('FlightFarePrediction.pkl', 'rb'))

@app.route("/")
@cross_origin()

def home():
	return render_template("home.html")

@app.route("/predict", methods=['POST'])
@cross_origin()

def predict():
	if request.method == "POST":

		Airline = request.form["airline"]
		Source = request.form["source"]
		Destination = request.form["destination"]
		Total_Stops = request.form["stops"]
		dep_datetime = pd.to_datetime(request.form["dep_datetime"], format="%Y-%m-%dT%H:%M")
		arr_datetime = pd.to_datetime(request.form["arr_datetime"], format="%Y-%m-%dT%H:%M")

		#Date_of_Journey	= pd.to_datetime(dep_datetime, format="%Y-%m-%dT%H:%M").date()
		#Dep_Time = pd.to_datetime(dep_datetime, format="%Y-%m-%dT%H:%M").time()
		Date_of_Journey = dep_datetime.date()
		Dep_Time = dep_datetime.time()

		travel_time = (arr_datetime - dep_datetime).components
		Duration = "{}h {}m".format(travel_time.hours, travel_time.minutes)
		
		
		# adding Route, Additional_info and Arrival_Time keys to dictionary. These are reqiured as per feature engineering function
		# keeping values of these features as '' instead of np.nan as feature engineering function will delete observations with nan values
		df_dict = {"Airline":Airline,
					"Date_of_Journey":pd.to_datetime(Date_of_Journey),
					"Source":Source,
					"Destination":Destination,
					"Dep_Time":"{}:{}".format(Dep_Time.hour, Dep_Time.minute),
					"Duration":Duration,
					"Total_Stops":Total_Stops,
					#"Route":np.nan,
					#"Additional_Info":np.nan,
					#"Arrival_Time":np.nan
					"Route":'',
					"Additional_Info":'',
					"Arrival_Time":''
					}

		df = pd.DataFrame(df_dict, index=[0])
		df = fe.fetEngineering(df)

		prediction = model.predict(df)
		output=round(prediction[0],2)

		return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))

	return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)

