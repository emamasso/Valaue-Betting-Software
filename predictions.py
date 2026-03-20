import pandas as pd
import pickle
import joblib
import numpy as np
from data_engineering import *

data = df_new
#data = pd.read_csv('prediction_data/final_data.csv', sep = ';')

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





with open('model_v2.pkl', 'rb') as file:
    model = pickle.load(file)



scaler = joblib.load('scaler.pkl')


scaled_data = scaler.transform(data_to_predict)

predictions = list(model.predict(scaled_data))
probabilities =list(model.predict_proba(scaled_data))

games = []

for i in range(data.shape[0]):
    games.append('-'.join((data['home'].loc[i], data['away'].loc[i])))



dictionary = {'Game':games, 'Forecasted result':predictions}


final_data_frame = pd.DataFrame(dictionary)


mask = [final_data_frame['Forecasted result'] == 0,
        final_data_frame['Forecasted result'] == 1,
        final_data_frame['Forecasted result'] == 2]

final_data_frame['Forecasted result'] = np.select(mask, ['1', 'X', '2'], default='N/A')


prob = [max(x) for x in probabilities]

final_data_frame['Probability'] = prob


odds = pd.read_csv('prediction_data/games_odds.csv', sep = ';')

data_with_odds = pd.merge(data, odds[['id', 'H', 'D', 'A']], on='id', how='left')

final_data_frame['Home Win'] = data_with_odds['H']
final_data_frame['Draw'] = data_with_odds['D']
final_data_frame['Away Win'] = data_with_odds['A']

quote_cols = ['Away Win', 'Draw', 'Home Win']

quote_scelte = [final_data_frame['Home Win'],
    final_data_frame['Draw'],
    final_data_frame['Away Win']]

final_data_frame['Bet'] = np.select(mask, quote_scelte, default=np.nan)

final_data_frame['Expected Value'] = final_data_frame['Probability'] * final_data_frame['Bet']
final_data_frame['Expected Value'] = final_data_frame['Probability'] * final_data_frame[quote_cols].min(axis=1)

#print(final_data_frame[final_data_frame['Expected Value'] >= 1.05].sort_values(by=['Expected Value'], ascending=False))