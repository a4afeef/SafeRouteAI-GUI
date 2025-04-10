import folium
from branca.utilities import none_max
from folium.plugins import MarkerCluster
from datetime import datetime

from pandas.core.interchange.dataframe_protocol import DataFrame
from pytz import timezone
import requests
import pandas as pd
import datetime as dt
import numpy as np
import itertools
import os
import joblib
from app.config import Keys

#Setting path to working directory
main_path = os.path.dirname(os.path.abspath( __file__ + "/../"))
accident_clustered = pd.read_csv(main_path + "/data/accident_clustered.csv")

state_geo = main_path + "/data/London_Borough_Excluding_MHW.geojson"

time_zone = timezone('Europe/London')

#Loading prediction model and model columns
model = joblib.load(main_path + "/ef_model/model.pkl")
model_columns = joblib.load(main_path + "/ef_model/model_columns.pkl")

#Setting keys for OpenWeather and Google
openweather_key = Keys['openweatherkey']
google_key = Keys['googlekey']

#Main function to predict accidents in every borrough
def borrough_accPrediction(date_time):
    #dat = accident_clustered
    datetime_object = dt.datetime.strptime(date_time, '%Y-%m-%dT%H:%M')

    new_dataset = accident_clustered.drop(columns=['hour', 'day_of_year', 'day_of_week', 'year'], axis=0)
    new_dataset['hour'] = datetime_object.hour
    day_of_year = (datetime_object - dt.datetime(datetime_object.year, 1, 1)).days + 1
    new_dataset['day_of_year'] = day_of_year
    day_of_week = datetime_object.date().weekday() + 1
    new_dataset['day_of_week'] = day_of_week
    new_dataset['year'] = datetime_object.year

    uArea = list(new_dataset['Area'].unique())
    areas = new_dataset[new_dataset['Area'].isin(uArea)].drop_duplicates(subset='Area', keep='first')
    weather = getBorWeather(areas, date_time)

    all_dataset = pd.merge(new_dataset, weather, how='left', on=['Area'])
    
    # Only drop 'time' column since other columns don't exist in OpenWeather data
    if 'time' in all_dataset.columns:
        all_dataset = all_dataset.drop(columns=['time'], axis=0)

    # run model predicition
    predicted_dataset = getBorProbability(all_dataset)

    print("left func")
    return predicted_dataset


#function to get weather information from OpenWeather
def getBorWeather(areas, date_time):
    weather_list = []  # Create a list to store weather data
    
    #convert datetime to timestamp for OpenWeather API
    timestamp = int(datetime.strptime(date_time, '%Y-%m-%dT%H:%M').timestamp())

    for index, row in areas.iterrows():
        lat = row["Latitude"]
        long = row["Longitude"]
        weather_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={long}&dt={timestamp}&appid={openweather_key}&units=metric"
        
        w_response = requests.get(weather_url)
        w_data = w_response.json()

        # OpenWeather API returns different structure than DarkSky
        weather_data = w_data['data'][0]  # Get first hour data
        weather_info = weather_data.get('weather', [{}])[0] if weather_data.get('weather') else {}
        
        # Map OpenWeather fields to match your model's expected fields
        weather_hour = {
            'time': timestamp,
            'temperature': weather_data.get('temp'),
            'apparentTemperature': weather_data.get('feels_like'),
            'humidity': weather_data.get('humidity'),
            'pressure': weather_data.get('pressure'),
            'windSpeed': weather_data.get('wind_speed'),
            'windGust': weather_data.get('wind_gust', 0),
            'windDeg': weather_data.get('wind_deg', 0),
            'cloudCover': weather_data.get('clouds', 0) / 100.0,  # Convert percentage to decimal
            'visibility': weather_data.get('visibility', 0) / 1000.0,  # Convert m to km
            'precipIntensity': weather_data.get('rain_1h', 0) if 'rain_1h' in weather_data else 0,
            'precipProbability': 1 if 'rain_1h' in weather_data else 0,
            'precipAccumulation': weather_data.get('snow_1h', 0) if 'snow_1h' in weather_data else 0,
            'dewPoint': weather_data.get('dew_point'),
            'uvi': weather_data.get('uvi', 0),
            'weather_id': weather_info.get('id'),
            'weather_main': weather_info.get('main'),
            'weather_description': weather_info.get('description'),
            'Area': row['Area']
        }
        
        weather_list.append(weather_hour)

    # Create final DataFrame using concat
    weather = pd.DataFrame(weather_list)
    return weather


#function to find probability of all accidents
def getBorProbability(all_dataset):
    #get prediction of given lat longs
    pred_ds = all_dataset[model_columns]
    prob = pd.DataFrame(model.predict_proba(pred_ds), columns=['No', 'probability'])
    prob = prob[['probability']]
    #merge with probability with lat longs
    output = prob.merge(all_dataset[['Latitude', 'Longitude', 'Borough_name']], how='outer', left_index=True, right_index=True)

    #rounding off lat long values for better filtering
    output["Latitude"] = round(output["Latitude"], 5)
    output["Longitude"] = round(output["Longitude"], 5)

    #droping duplicate lat longs
    output = output.drop_duplicates(subset=['Longitude', 'Latitude'], keep="last")

    print("total accident count:", len(output))

    return output


#Main funtion to accidents in a route
def route_accPrediction(origin, destination, date_time):
    print("entered func")
    #parse time
    datetime_object = dt.datetime.strptime(date_time, '%Y-%m-%dT%H:%M')

    # get route planning
    lats, longs, total_lat_long = getGoogleArea(origin, destination)
    # calculate distance between past accident points and route
    accident_dataset = getDistance(accident_clustered, lats, longs, total_lat_long)
    # filter for past accident points with distance <50m - route cluster
    new_dataset = accident_dataset[accident_dataset['distance'] < 0.003][['longitude', 'latitude', 'day_of_week', 'year', 'cluster',
                                               'day_of_year', 'hour', 'local_authority_ons_district']]
    print('new ds:', new_dataset.columns)
    #if no cluster, exit
    if len(new_dataset) == 0:
        return print("There is NO accident predicted on your way. You are safe!")

    else:
        # filter for accident points in route cluster
        new_dataset = new_dataset.drop(columns=['hour', 'day_of_year', 'day_of_week', 'year'], axis=0)
        new_dataset['hour'] = datetime_object.hour
        day_of_year = (datetime_object - dt.datetime(datetime_object.year, 1, 1)).days + 1
        new_dataset['day_of_year'] = day_of_year
        day_of_week = datetime_object.date().weekday() + 1
        new_dataset['day_of_week'] = day_of_week
        new_dataset['year'] = datetime_object.year
        new_dataset['day'] = datetime_object.day
        new_dataset['month'] = datetime_object.month

        #get weather prediction for unique cluster in past accident dataset
        unique_boroughs = list(new_dataset['local_authority_ons_district'].unique())
        boroughs = new_dataset[new_dataset['local_authority_ons_district'].isin(unique_boroughs)].drop_duplicates(subset='local_authority_ons_district', keep='first')
        weather = getWeather(boroughs, date_time)

        # merge with accident data - df with latlong and weather
        all_dataset = pd.merge(new_dataset, weather, how='left', on=['local_authority_ons_district'])
        # Only drop 'time' column since other columns don't exist in OpenWeather data
        if 'time' in all_dataset.columns:
            all_dataset = all_dataset.drop(columns=['time'], axis=0)

        print('mod col', model_columns)
        print('data col', all_dataset.columns)
        all_dataset = all_dataset[model_columns]
        print('bef pred')

        #run model predicition
        predicted_dataset = getProbability(all_dataset)

        final = {}
        final["accidents"] = predicted_dataset
        print("left func")
        return final


#function to get google Waypoints
def getGoogleArea(origin, destination):
    print('inside google area')
    print(origin, destination)
    
    # New Routes API endpoint
    URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    # New request format for Routes API
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': google_key,
        'X-Goog-FieldMask': 'routes.legs.steps.startLocation'
    }
    
    data = {
        "origin": {
            "address": origin
        },
        "destination": {
            "address": destination
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "computeAlternativeRoutes": False,
        "routeModifiers": {
            "vehicleInfo": {
                "emissionType": "GASOLINE"
            }
        },
        "languageCode": "en-GB",
        "units": "METRIC"
    }
    
    # Make POST request to Routes API
    results = requests.post(url=URL, headers=headers, json=data)
    output = results.json()
    print('Request output:', output)

    # Fetching all waypoints from new response format
    waypoints = output['routes'][0]['legs']

    lats = []
    longs = []
    total_lat_long = 0

    # Storing all lat separately
    for leg in waypoints:
        for step in leg['steps']:
            location = step['startLocation']['latLng']  # Updated to handle latLng nesting
            lats.append(location['latitude'])
            longs.append(location['longitude'])
            total_lat_long += 1

    lats = tuple(lats)
    longs = tuple(longs)
    print("Total points: " + str(total_lat_long))

    return lats, longs, total_lat_long


#function to filter related lat longs
def getDistance(accident_clustered, lats, longs, total_lat_long):
    accident_point_counts = len(accident_clustered.index)

    # approximate radius of earth in km
    R = 6373.0
    
    # Create a list of dataframes instead of using append
    dfs = [accident_clustered] * total_lat_long
    x_df = pd.concat(dfs, ignore_index=True)

    #create list equal to no. of accidents in dataset
    lats_r = list(itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in lats))
    longs_r = list(itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in longs))

    # append
    x_df['lat2'] = np.radians(lats_r)
    x_df['long2'] = np.radians(longs_r)

    # cal radiun50m
    x_df['lat1'] = np.radians(x_df['latitude'])
    x_df['long1'] = np.radians(x_df['longitude'])
    x_df['dlon'] = x_df['long2'] - x_df['long1']
    x_df['dlat'] = x_df['lat2'] - x_df['lat1']

    x_df['a'] = np.sin(x_df['dlat'] / 2) ** 2 + np.cos(x_df['lat1']) * np.cos(x_df['lat2']) * np.sin(x_df['dlon'] / 2) ** 2
    x_df['distance'] = R * (2 * np.arctan2(np.sqrt(x_df['a']), np.sqrt(1 - x_df['a'])))

    return x_df

#function to get weather information from OpenWeather
def getWeather(boroughs, date_time):
    weather_list = []  # Create a list to store weather data
    
    #convert datetime to timestamp for OpenWeather API
    timestamp = int(datetime.strptime(date_time, '%Y-%m-%dT%H:%M').timestamp())
    print("Total boroughs", boroughs.count())

    for index, row in boroughs.iterrows():
        print("Weather for borough: ", index)
        lat = row["latitude"]
        long = row["longitude"]
        weather_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={long}&dt={timestamp}&appid={openweather_key}&units=metric"

        print(weather_url)
        w_response = requests.get(weather_url)
        w_data = w_response.json()
        print(w_data)

        # OpenWeather API returns different structure than DarkSky
        weather_data = w_data['data'][0]  # Get first hour data
        weather_info = weather_data.get('weather', [{}])[0] if weather_data.get('weather') else {}
        
        # Map OpenWeather fields to match model's expected fields
        weather_hour = {
            'temp': weather_data.get('temp'),
            'feels_like': weather_data.get('feels_like'),
            'humidity': weather_data.get('humidity'),
            'pressure': weather_data.get('pressure'),
            'wind_speed': weather_data.get('wind_speed'),
            'wind_deg': weather_data.get('wind_deg', 0),
            'clouds': weather_data.get('clouds', 0),
            'visibility': weather_data.get('visibility', 0),
            'dew_point': weather_data.get('dew_point'),
            'weather_id': weather_info.get('id'),
            'local_authority_ons_district': row['local_authority_ons_district']
        }
        
        weather_list.append(weather_hour)

    # Create final DataFrame using concat
    weather = pd.DataFrame(weather_list)
    print('weather done')
    return weather


#get probability of accident points from EF model
def getProbability(all_dataset):
    #get prediction of given lat longs
    prob = pd.DataFrame(model.predict_proba(all_dataset), columns=['No', 'probability'])
    prob = prob[['probability']]
    #merge with probability with lat longs
    output = prob.merge(all_dataset[['latitude', 'longitude']], how='outer', left_index=True, right_index=True)

    #rounding off lat long values for better filtering
    output["latitude"] = round(output["latitude"], 5)
    output["longitude"] = round(output["longitude"], 5)

    #droping duplicate lat longs
    output = output.drop_duplicates(subset=['longitude', 'latitude'], keep="last")

    # to json
    processed_results = []
    for index, row in output.iterrows():
        lat = float(row['latitude'])
        long = float(row['longitude'])
        prob = float(row['probability'])

        result = {'lat': lat, 'lng': long, 'probability': prob}
        processed_results.append(result)

    print("total accident count:", len(output))

    return processed_results


