import pandas as pd
import pickle
import joblib

data = pd.read_csv('prediction_data/final_data.csv', sep = ';')

data['home_is_home'] = 1
data['away_is_home'] = 0

data_to_predict = data[['home_is_home', 
                        'B365H', 'B365D', 'B365A',

                        'home_rest_days', 'home_total_goals', 'home_total_xg',
                        'home_total_goals_against', 'home_total_xg_against', 
                        'home_last_5', 'home_PPG', 
                          
                        'away_is_home', 'away_rest_days',
                        'away_total_goals', 'away_total_xg',
                        'away_total_goals_against', 'away_total_xg_against', 
                        'away_last_5', 'away_PPG'
                        ]]



df = pd.read_csv('data/final_data.csv', sep=';')

X = df[['home_is_home', 'B365H', 'B365D', 'B365A',
        'home_rest_days','home_total_goals','home_total_xg',
        'home_total_goals_against','home_total_xg_against','home_last_5',
        'home_PPG','away_is_home','away_rest_days',
        'away_total_goals','away_total_xg',
        'away_total_goals_against','away_total_xg_against',
        'away_last_5','away_PPG']]

with open('model_v2.pkl', 'rb') as file:
    model = pickle.load(file)



scaler = joblib.load('scaler.pkl')


scaled_data = scaler.transform(data_to_predict)

predictions = model.predict(scaled_data)
print(len(predictions))
print(predictions)


